# SPDX-License-Identifier: Apache-2.0
"""CPU checks for finished-image qualification and promotion failure paths."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e.memory import MemoryProbe, parse_memory
from e2e.run import image_overrides
from e2e.elastic import Usage

spec = importlib.util.spec_from_file_location('release_images', Path(__file__).parents[1] / 'tools/release/images.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_image_override_requires_finished_image():
    assert image_overrides(['vllm=local:v1', 'sglang=registry/x@sha256:123'], True, False) == {
        'vllm': 'local:v1', 'sglang': 'registry/x@sha256:123'}
    for values, no_install, local in [(['vllm=x'], False, False), (['vllm=x'], True, True),
                                      (['bad=x'], True, False), (['vllm='], True, False),
                                      (['vllm=x', 'vllm=y'], True, False)]:
        with pytest.raises(ValueError):
            image_overrides(values, no_install, local)


def test_nvml_counters():
    assert parse_memory('100, 200') == (100, 200)
    for value in ['[N/A], [N/A]', '200, 100', '-1, 100', '100, 0']:
        with pytest.raises(ValueError):
            parse_memory(value)


def test_cuda_fallback_preserves_selected_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr('e2e.memory.smi', lambda *args: '[N/A], [N/A]')
    class Cuda:
        def __init__(self, gpu): calls.append(gpu)
        def read(self): return 10, 100
        def close(self): calls.append('close')
    monkeypatch.setattr('e2e.memory.CudaMemory', Cuda)
    probe = MemoryProbe('GPU-selected')
    assert probe.source == 'cuda.cuMemGetInfo'
    assert probe.read() == (10, 100)
    probe.close()
    assert calls == ['GPU-selected', 'close']


def test_probe_failure_is_not_silently_skipped(monkeypatch):
    monkeypatch.setattr('e2e.memory.smi', lambda *args: '[N/A], [N/A]')
    def fail(gpu): raise RuntimeError('no driver')
    monkeypatch.setattr('e2e.memory.CudaMemory', fail)
    with pytest.raises(RuntimeError, match='no driver'):
        MemoryProbe('1')


def test_sampling_thread_failure_is_reported():
    def fail(): raise RuntimeError('lost GPU')
    usage = Usage(SimpleNamespace(read=fail))
    usage._thread.join(timeout=2)
    usage.stop()
    assert usage.error is not None
    with pytest.raises(RuntimeError, match='sampling failed'):
        usage.peak(SimpleNamespace(ipc='test'), 0)


def test_candidate_platform_and_revision():
    labels = {'org.opencontainers.image.revision': 'a' * 40,
              'org.opencontainers.image.version': '0.1.6', 'io.kvcached.engine': 'vllm',
              'io.kvcached.engine.version': '0.30.0'}
    info = {'Architecture': 'arm64', 'Os': 'linux', 'Config': {'Labels': labels}}
    release.validate_image(info, 'vllm', 'arm64', 'a' * 40, '0.1.6')
    for arch, revision in [('amd64', 'a' * 40), ('arm64', 'b' * 40)]:
        with pytest.raises(ValueError):
            release.validate_image(info, 'vllm', arch, revision, '0.1.6')


def test_missing_or_mismatched_gpu_evidence():
    evidence = {'revision': 'a' * 40, 'architecture': 'arm64', 'images': {'vllm': 'ref'},
                'summary': {'ok': True, 'cases': []}}
    with pytest.raises(ValueError, match='missing'):
        release.validate_evidence(evidence, 'vllm', 'arm64', 'a' * 40, 'ref')
    with pytest.raises(ValueError, match='match'):
        release.validate_evidence(evidence, 'vllm', 'amd64', 'a' * 40, 'ref')


def test_incomplete_candidates_never_publish(tmp_path, monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, 'argv', ['images.py', 'promote', '--candidates', str(tmp_path),
        '--evidence', str(tmp_path), '--revision', 'a' * 40, '--namespace', 'example', '--publish'])
    with pytest.raises(ValueError, match='four'):
        release.main()
    assert not calls


def test_failed_cache_reset_cannot_pass_reclaim():
    from e2e.elastic import drain
    srv = SimpleNamespace(client=SimpleNamespace(reset_prefix_cache=lambda: False))
    with pytest.raises(RuntimeError, match='reset failed'):
        drain(None, srv)


def test_memory_probe_initialization_failure_becomes_failed_case(monkeypatch):
    from e2e.elastic import run_elastic
    from e2e.matrix import ElasticPair
    def fail(gpu): raise RuntimeError('no counters')
    monkeypatch.setattr('e2e.elastic.MemoryProbe', fail)
    result = run_elastic(SimpleNamespace(gpu='1'), None,
                         ElasticPair('vllm', 'release05b', 'vllm', 'release05b'), 'test', 18000, 1)
    assert not result.ok
    assert result.checks[0].name == 'memory_probe'


def test_release_refuses_busy_gpu_before_starting_containers(tmp_path, monkeypatch):
    import sys
    from e2e import run
    monkeypatch.setattr(sys, 'argv', ['run.py', '--profile', 'release', '--out', str(tmp_path)])
    monkeypatch.setattr(run, 'gpu_processes', lambda gpus: ['1234'])
    def forbidden(*args, **kwargs): raise AssertionError('must not start containers')
    monkeypatch.setattr(run, 'Container', forbidden)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
