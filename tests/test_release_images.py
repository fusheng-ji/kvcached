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
              'io.kvcached.engine.version': '0.30.0',
              'org.opencontainers.image.base.name': 'vllm/vllm-openai:v0.30.0',
              'org.opencontainers.image.base.digest': 'sha256:' + 'c' * 64}
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


def image_info(engine, arch, revision='a' * 40):
    return {'Architecture': arch, 'Os': 'linux', 'Config': {'Labels': {
        'org.opencontainers.image.revision': revision,
        'org.opencontainers.image.version': '0.1.6',
        'io.kvcached.engine': engine,
        'io.kvcached.engine.version': release.ENGINES[engine],
        'org.opencontainers.image.base.name': f'upstream/{engine}:version',
        'org.opencontainers.image.base.digest': 'sha256:' + 'c' * 64,
    }}}


def candidate_refs():
    return {(engine, arch): f'ghcr.io/example/kvcached-{engine}@sha256:{i:064x}'
            for i, (engine, arch) in enumerate(
                ((e, a) for e in release.ENGINES for a in release.ARCHES), start=1)}


def passing_evidence(arch, candidates):
    cases, engines = [], {}
    for engine in release.ENGINES:
        for name, checks in {
            f'{engine}-setup': ['installed_package_verified', 'model_snapshots_ready'],
            f'{engine}-release05b-kv_c1': ['server_ready', 'server_alive_after_load',
                'prefix_cache_reset', 'no_error_signatures', 'kvcached_active', 'gen_seq_tokens_equal'],
            f'{engine}-release05b-native': ['server_ready', 'server_alive_after_load',
                'prefix_cache_reset', 'no_error_signatures'],
            f'elastic-{engine}-release05b+{engine}-release05b': ['a_requests_ok', 'b_requests_ok',
                'a_filled_memory', 'a_released_memory', 'b_reused_budget', 'a_physical_memory_released',
                'contended_requests_ok', 'memory_sampling', 'shared_budget_enforced'],
        }.items():
            if name.startswith('elastic-'):
                checks += ['no_gpu_process_left']
                checks += [f'elastic-{side}-{engine}-release05b-kv_c1_{c}'
                           for side in ('a', 'b') for c in ('ready', 'probe_unchanged', 'alive',
                               'clean_exit_on_signal', 'no_error_signatures', 'kvcached_active',
                               'shm_removed', 'socket_dirs_removed')]
            elif not name.endswith('-setup'):
                checks += ['clean_exit_on_signal', 'no_gpu_process_left']
                checks += [f'{phase}_ok' for phase in
                           ('gen_seq', 'gen_conc', 'apc_0', 'apc_1', 'apc_2', 'apc_conc')]
                if name.endswith('-kv_c1'):
                    checks += ['shm_removed', 'socket_dirs_removed']
                    checks += [f'apc_{i}_{check}' for i in range(3)
                               for check in ('tokens_equal', 'cached_tokens_equal')]
                else:
                    checks += ['kvcached_inactive']
            cases.append({'name': name, 'ok': True,
                          'checks': [{'name': c, 'ok': True} for c in checks]})
        engines[engine] = {'image': candidates[engine, arch], 'architecture': arch,
                           'labels': image_info(engine, arch)['Config']['Labels'],
                           'engine': release.ENGINES[engine], 'kvcached': '0.1.6',
                           'torch': '2.11.0', 'cuda': '13.0'}
    return {'revision': 'a' * 40, 'architecture': arch,
            'images': {e: candidates[e, arch] for e in release.ENGINES},
            'summary': {'ok': True, 'cases': cases},
            'setup': {'profile': 'release', 'kvcached_sha': 'a' * 40,
                      'engines': engines, 'gpus': [{'name': 'NVIDIA GB10', 'driver': '580.0'}],
                      'model_revisions': {release.MODEL: release.MODEL_REVISION}}}


@pytest.mark.parametrize('arch', release.ARCHES)
def test_complete_gpu_evidence_passes(arch):
    refs = candidate_refs()
    evidence = passing_evidence(arch, refs)
    for engine in release.ENGINES:
        release.validate_evidence(evidence, engine, arch, 'a' * 40, refs[engine, arch])


@pytest.mark.parametrize('mutation', ['case_duplicate', 'check_duplicate', 'case_truthy',
    'check_truthy', 'unrelated_failure', 'missing_budget_check', 'missing_setup', 'wrong_profile',
    'wrong_revision', 'wrong_image', 'wrong_arch', 'wrong_version', 'wrong_model', 'missing_driver'])
def test_contradictory_or_untraceable_evidence_cannot_pass(mutation):
    refs = candidate_refs()
    e = passing_evidence('arm64', refs)
    case = e['summary']['cases'][0]
    if mutation == 'case_duplicate':
        e['summary']['cases'].append(case)
    elif mutation == 'check_duplicate':
        case['checks'].append(case['checks'][0])
    elif mutation == 'case_truthy':
        case['ok'] = 'true'
    elif mutation == 'check_truthy':
        case['checks'][0]['ok'] = 'true'
    elif mutation == 'unrelated_failure':
        e['summary']['cases'].append({'name': 'extra', 'ok': False, 'checks': []})
    elif mutation == 'missing_budget_check':
        elastic = e['summary']['cases'][3]
        elastic['checks'] = [c for c in elastic['checks'] if c['name'] != 'shared_budget_enforced']
    elif mutation == 'missing_setup':
        e.pop('setup')
    elif mutation == 'wrong_profile':
        e['setup']['profile'] = 'smoke'
    elif mutation == 'wrong_revision':
        e['setup']['kvcached_sha'] = 'b' * 40
    elif mutation == 'wrong_image':
        e['setup']['engines']['vllm']['image'] = 'local:latest'
    elif mutation == 'wrong_arch':
        e['setup']['engines']['vllm']['architecture'] = 'amd64'
    elif mutation == 'wrong_version':
        e['setup']['engines']['vllm']['engine'] = '0.29.0'
    elif mutation == 'wrong_model':
        e['setup']['model_revisions'][release.MODEL] = 'b' * 40
    elif mutation == 'missing_driver':
        e['setup']['gpus'][0].pop('driver')
    with pytest.raises(ValueError):
        release.validate_evidence(e, 'vllm', 'arm64', 'a' * 40, refs['vllm', 'arm64'])


def index_for(engine, refs):
    return {'schemaVersion': 2, 'manifests': [
        {'platform': {'os': 'linux', 'architecture': arch},
         'digest': refs[engine, arch].split('@')[1]}
        for arch in release.ARCHES]}


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'wrong_os', 'wrong_digest'])
def test_staged_index_must_match_both_qualified_platforms(problem):
    refs = candidate_refs()
    index = index_for('vllm', refs)
    release.validate_index(index, 'vllm', refs)
    if problem == 'missing':
        index['manifests'].pop()
    elif problem == 'duplicate':
        index['manifests'].append(index['manifests'][0])
    elif problem == 'wrong_os':
        index['manifests'][0]['platform']['os'] = 'windows'
    else:
        index['manifests'][0]['digest'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError, match='staged manifest'):
        release.validate_index(index, 'vllm', refs)


def mock_registry(monkeypatch, refs, broken_engine=None):
    class Registry:
        def __init__(self, repo): self.engine = repo.rsplit('-', 1)[1]
        def fetch(self, kind, reference):
            index = index_for(self.engine, refs)
            if self.engine == broken_engine:
                index['manifests'].pop()
            return index, 'sha256:' + ('b' if self.engine == 'vllm' else 'c') * 64
    monkeypatch.setattr(release, 'Registry', Registry)


def test_promotion_uses_verified_digests_and_updates_latest_last(monkeypatch):
    refs, commands = candidate_refs(), []
    mock_registry(monkeypatch, refs)
    monkeypatch.setattr(release.subprocess, 'run', lambda args, **kw: commands.append(args))
    release.promote(refs, 'example', 'a' * 40, '0.1.6', True)
    assert len(commands) == 6
    assert all(':candidate-' in args[5] for args in commands[:2])
    assert all(':kvcached-v' in args[5] for args in commands[2:4])
    assert all(args[5].endswith(':latest') for args in commands[4:])
    assert commands[2][-1] == commands[4][-1] == 'ghcr.io/example/kvcached-vllm@sha256:' + 'b' * 64
    assert commands[3][-1] == commands[5][-1] == 'ghcr.io/example/kvcached-sglang@sha256:' + 'c' * 64


def test_bad_second_staged_manifest_changes_no_official_tags(monkeypatch):
    refs, commands = candidate_refs(), []
    mock_registry(monkeypatch, refs, broken_engine='sglang')
    monkeypatch.setattr(release.subprocess, 'run', lambda args, **kw: commands.append(args))
    with pytest.raises(ValueError, match='staged manifest'):
        release.promote(refs, 'example', 'a' * 40, '0.1.6', True)
    assert len(commands) == 2
    assert all(':candidate-' in args[5] for args in commands)


def test_version_publish_failure_never_updates_latest(monkeypatch):
    import subprocess
    refs, commands = candidate_refs(), []
    mock_registry(monkeypatch, refs)
    def run(args, **kwargs):
        commands.append(args)
        if len(commands) == 4:
            raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(release.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        release.promote(refs, 'example', 'a' * 40, '0.1.6', True)
    assert all(not args[5].endswith(':latest') for args in commands)


def test_promotion_dry_run_writes_nothing(monkeypatch, capsys):
    def forbidden(*args, **kwargs): raise AssertionError('dry run must not write registry tags')
    monkeypatch.setattr(release.subprocess, 'run', forbidden)
    release.promote(candidate_refs(), 'example', 'a' * 40, '0.1.6', False)
    assert len(capsys.readouterr().out.splitlines()) == 6


def write_candidate_files(directory, refs):
    import json
    directory.mkdir(parents=True, exist_ok=True)
    for (engine, arch), ref in refs.items():
        (directory / f'candidate-{engine}-{arch}.json').write_text(json.dumps({
            'engine': engine, 'architecture': arch, 'revision': 'a' * 40,
            'digest': ref.split('@')[1]}))


@pytest.mark.parametrize('problem', ['missing_arm64', 'bad_revision', 'failed_check'])
def test_invalid_evidence_never_reaches_registry_writes(tmp_path, monkeypatch, problem):
    import json
    import sys
    refs = candidate_refs()
    write_candidate_files(tmp_path / 'candidates', refs)
    evidence_dir = tmp_path / 'evidence'
    evidence_dir.mkdir()
    for arch in release.ARCHES:
        e = passing_evidence(arch, refs)
        if arch == 'arm64':
            if problem == 'missing_arm64':
                continue
            if problem == 'bad_revision':
                e['revision'] = 'b' * 40
            if problem == 'failed_check':
                e['summary']['cases'][0]['checks'][0]['ok'] = False
        (evidence_dir / f'evidence-{arch}.json').write_text(json.dumps(e))
    monkeypatch.setattr(release, 'inspect', lambda ref, arch: image_info(
        ref.split('@')[0].rsplit('-', 1)[1], arch))
    def forbidden(*args, **kwargs): raise AssertionError('must finish evidence validation first')
    monkeypatch.setattr(release.subprocess, 'run', forbidden)
    monkeypatch.setattr(sys, 'argv', ['images.py', 'promote', '--candidates',
        str(tmp_path / 'candidates'), '--evidence', str(evidence_dir), '--revision', 'a' * 40,
        '--namespace', 'example', '--publish'])
    with pytest.raises(ValueError):
        release.main()


def test_qualification_preserves_gpu_and_attaches_setup(tmp_path, monkeypatch):
    import json
    import sys
    refs = candidate_refs()
    write_candidate_files(tmp_path / 'candidates', refs)
    evidence_dir = tmp_path / 'evidence'
    evidence = passing_evidence('arm64', refs)
    monkeypatch.setattr('platform.machine', lambda: 'aarch64')
    monkeypatch.setattr(release, 'inspect', lambda ref, arch: image_info(
        ref.split('@')[0].rsplit('-', 1)[1], arch))
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        (evidence_dir / 'summary.json').write_text(json.dumps(evidence['summary']))
        (evidence_dir / 'setup.json').write_text(json.dumps(evidence['setup']))
    monkeypatch.setattr(release.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['images.py', 'qualify', '--candidates',
        str(tmp_path / 'candidates'), '--evidence', str(evidence_dir), '--revision', 'a' * 40,
        '--namespace', 'example', '--arch', 'arm64', '--gpus', 'GPU-1234-abcd'])
    release.main()
    assert len(commands) == 1
    command = commands[0]
    assert '--no-install' in command
    assert command[command.index('--gpus') + 1] == 'GPU-1234-abcd'
    saved = json.loads((evidence_dir / 'evidence-arm64.json').read_text())
    assert saved['setup'] == evidence['setup']
    # A retry cannot reuse an earlier passing summary or leave stale evidence.
    with pytest.raises(ValueError, match='must be empty'):
        release.main()
    assert len(commands) == 1


def test_registry_digest_mismatch_fails_before_parsing(monkeypatch):
    import io
    registry = object.__new__(release.Registry)
    registry.path, registry.token = 'example/kvcached-vllm', 'test-token'
    monkeypatch.setattr(release.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'{}'))
    with pytest.raises(ValueError, match='digest mismatch'):
        registry.fetch('manifests', 'sha256:' + 'a' * 64)


@pytest.mark.parametrize('missing_check', ['model_snapshots_ready', 'kvcached_inactive',
    'gen_seq_ok', 'apc_1_cached_tokens_equal',
    'elastic-a-vllm-release05b-kv_c1_probe_unchanged'])
def test_required_real_inference_and_recovery_checks_cannot_be_omitted(missing_check):
    refs = candidate_refs()
    evidence = passing_evidence('arm64', refs)
    removed = False
    for case in evidence['summary']['cases']:
        checks = case['checks']
        case['checks'] = [check for check in checks if check['name'] != missing_check]
        removed |= len(case['checks']) != len(checks)
    assert removed
    with pytest.raises(ValueError, match='missing required'):
        release.validate_evidence(evidence, 'vllm', 'arm64', 'a' * 40, refs['vllm', 'arm64'])


def test_release_model_pin_matches_qualification_matrix():
    from e2e.matrix import MODELS
    assert MODELS['release05b'].hf_id == release.MODEL
    assert MODELS['release05b'].revision == release.MODEL_REVISION
