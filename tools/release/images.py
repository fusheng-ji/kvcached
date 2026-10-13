#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate immutable candidate images and GPU evidence before promotion."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ENGINES = {'vllm': '0.30.0', 'sglang': '0.5.20'}
ARCHES = ('amd64', 'arm64')
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
MANIFEST_TYPES = ', '.join([
    'application/vnd.oci.image.index.v1+json',
    'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
    'application/vnd.docker.distribution.manifest.v2+json',
])


class Registry:
    """Read manifests/configs without pulling tens of GB of image layers."""

    def __init__(self, repository):
        if not repository.startswith('ghcr.io/'):
            raise ValueError('only GHCR repositories are supported')
        self.path = repository.removeprefix('ghcr.io/')
        query = urllib.parse.urlencode({'service': 'ghcr.io',
                                        'scope': f'repository:{self.path}:pull'})
        headers = {}
        if os.environ.get('GH_TOKEN'):
            credentials = f"{os.environ.get('GITHUB_ACTOR', 'token')}:{os.environ['GH_TOKEN']}"
            headers['Authorization'] = 'Basic ' + base64.b64encode(credentials.encode()).decode()
        request = urllib.request.Request('https://ghcr.io/token?' + query, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            self.token = json.load(response)['token']

    def fetch(self, kind, reference):
        request = urllib.request.Request(
            f'https://ghcr.io/v2/{self.path}/{kind}/{reference}', headers={
                'Authorization': f'Bearer {self.token}', 'Accept': MANIFEST_TYPES})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        digest = 'sha256:' + hashlib.sha256(payload).hexdigest()
        if reference.startswith('sha256:') and digest != reference:
            raise ValueError('registry content digest mismatch')
        return json.loads(payload), digest


def inspect(reference, arch):
    repository, digest = reference.split('@')
    registry = Registry(repository)
    manifest, _ = registry.fetch('manifests', digest)
    config, _ = registry.fetch('blobs', manifest['config']['digest'])
    return {'Architecture': config['architecture'], 'Os': config['os'],
            'Config': {'Labels': config['config'].get('Labels', {})}}


def validate_image(info, engine, arch, revision, version):
    labels = info['Config']['Labels']
    expected = {'org.opencontainers.image.revision': revision,
                'org.opencontainers.image.version': version,
                'io.kvcached.engine': engine,
                'io.kvcached.engine.version': ENGINES[engine]}
    if info['Architecture'] != arch or info['Os'] != 'linux':
        raise ValueError('candidate platform mismatch')
    if any(labels.get(k) != v for k, v in expected.items()):
        raise ValueError('candidate revision/version labels mismatch')
    if (not labels.get('org.opencontainers.image.base.name')
            or not re.fullmatch(r'sha256:[0-9a-f]{64}',
                                labels.get('org.opencontainers.image.base.digest', ''))):
        raise ValueError('candidate base image provenance missing')


def unique_named(items, what):
    selected = {}
    for item in items:
        name = item.get('name')
        if not isinstance(name, str) or name in selected:
            raise ValueError(f'duplicate or missing {what} name')
        selected[name] = item
    return selected


def validate_evidence(evidence, engine, arch, revision, reference, version='0.1.6'):
    if (evidence.get('revision') != revision or evidence.get('architecture') != arch
            or evidence.get('images', {}).get(engine) != reference):
        raise ValueError('GPU evidence does not match candidate digest/revision/platform')
    if evidence.get('summary', {}).get('ok') is not True:
        raise ValueError('GPU qualification did not pass')
    cases = unique_named(evidence.get('summary', {}).get('cases', []), 'case')
    required = {f'{engine}-setup', f'{engine}-release05b-kv_c1',
                f'{engine}-release05b-native', f'elastic-{engine}-release05b+{engine}-release05b'}
    if not required <= cases.keys():
        raise ValueError('missing GPU qualification cases')
    # Reject contradictory summaries, including failures outside the required set.
    for name, case in cases.items():
        if case.get('ok') is not True or not case.get('checks'):
            raise ValueError('missing or failing GPU qualification cases')
        checks = unique_named(case['checks'], 'check')
        if any(check.get('ok') is not True for check in checks.values()):
            raise ValueError('GPU qualification contains failing checks')
        if name not in required:
            continue
        if name.endswith('-setup'):
            mandatory = {'installed_package_verified', 'model_snapshots_ready'}
        elif name.startswith('elastic-'):
            mandatory = {'a_requests_ok', 'b_requests_ok', 'a_filled_memory', 'a_released_memory',
                         'b_reused_budget', 'a_physical_memory_released', 'contended_requests_ok',
                         'memory_sampling', 'shared_budget_enforced', 'no_gpu_process_left'}
            mandatory |= {f'elastic-{side}-{engine}-release05b-kv_c1_{check}'
                          for side in ('a', 'b') for check in (
                              'ready', 'probe_unchanged', 'alive', 'clean_exit_on_signal',
                              'no_error_signatures', 'kvcached_active', 'shm_removed',
                              'socket_dirs_removed')}
        else:
            mandatory = {'server_ready', 'server_alive_after_load', 'prefix_cache_reset',
                         'no_error_signatures', 'clean_exit_on_signal', 'no_gpu_process_left'}
            mandatory |= {f'{phase}_ok' for phase in
                          ('gen_seq', 'gen_conc', 'apc_0', 'apc_1', 'apc_2', 'apc_conc')}
            if name.endswith('-kv_c1'):
                mandatory |= {'kvcached_active', 'shm_removed', 'socket_dirs_removed',
                              'gen_seq_tokens_equal'}
                mandatory |= {f'apc_{i}_{check}' for i in range(3)
                              for check in ('tokens_equal', 'cached_tokens_equal')}
            else:
                mandatory |= {'kvcached_inactive'}
        if not mandatory <= checks.keys():
            raise ValueError('missing required GPU qualification checks')

    setup = evidence.get('setup', {})
    if setup.get('profile') != 'release' or setup.get('kvcached_sha') != revision:
        raise ValueError('GPU setup profile/revision mismatch')
    image = setup.get('engines', {}).get(engine, {})
    if (image.get('image') != reference or image.get('engine') != ENGINES[engine]
            or image.get('kvcached') != version or not image.get('torch') or not image.get('cuda')):
        raise ValueError('GPU setup image or installed versions mismatch')
    validate_image({'Architecture': image.get('architecture'), 'Os': 'linux',
                    'Config': {'Labels': image.get('labels', {})}}, engine, arch, revision, version)
    if setup.get('model_revisions', {}).get(MODEL) != MODEL_REVISION:
        raise ValueError('GPU setup is missing the tested model revision')
    if not setup.get('gpus') or any(not gpu.get('driver') for gpu in setup['gpus']):
        raise ValueError('GPU setup is missing driver provenance')


def validate_index(index, engine, candidates):
    """A release manifest must contain exactly the two qualified platforms."""
    platforms = {}
    for manifest in index.get('manifests', []):
        platform = manifest.get('platform', {})
        arch = platform.get('architecture')
        if platform.get('os') != 'linux' or arch not in ARCHES or arch in platforms:
            raise ValueError('staged manifest contains unexpected or duplicate platforms')
        platforms[arch] = manifest.get('digest')
    expected = {arch: candidates[engine, arch].split('@')[1] for arch in ARCHES}
    if platforms != expected:
        raise ValueError('staged manifest does not match both qualified candidate digests')


def promote(candidates, namespace, revision, version, publish):
    staged = {}
    # Assemble and verify BOTH engines before touching any official tag. Always
    # promote immutable index digests, so a tag race cannot change latest's input.
    for engine in ENGINES:
        repo = f'ghcr.io/{namespace}/kvcached-{engine}'
        staging_tag = f'candidate-{revision}'
        args = ['docker', 'buildx', 'imagetools', 'create', '--tag', f'{repo}:{staging_tag}',
                *(candidates[engine, arch] for arch in ARCHES)]
        if publish:
            subprocess.run(args, check=True)
            index, digest = Registry(repo).fetch('manifests', staging_tag)
            validate_index(index, engine, candidates)
            staged[engine] = f'{repo}@{digest}'
        else:
            print(json.dumps(args))
            staged[engine] = f'{repo}@sha256:<verified-candidate-index-digest>'
    for engine, engine_version in ENGINES.items():
        repo = f'ghcr.io/{namespace}/kvcached-{engine}'
        tag = f'{repo}:kvcached-v{version}-{engine}-v{engine_version}'
        args = ['docker', 'buildx', 'imagetools', 'create', '--tag', tag, staged[engine]]
        if publish:
            subprocess.run(args, check=True)
        else:
            print(json.dumps(args))
    # Registry writes across repositories cannot be transactional. latest only
    # moves after both version tags succeeded; retries use the same validated inputs.
    for engine in ENGINES:
        repo = f'ghcr.io/{namespace}/kvcached-{engine}'
        args = ['docker', 'buildx', 'imagetools', 'create', '--tag', f'{repo}:latest', staged[engine]]
        if publish:
            subprocess.run(args, check=True)
        else:
            print(json.dumps(args))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['qualify', 'promote'])
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--version', default='0.1.6')
    ap.add_argument('--arch', choices=ARCHES)
    ap.add_argument('--gpus', default='0', help='idle GPU index or UUID for native qualification')
    ap.add_argument('--namespace', required=True)
    ap.add_argument('--publish', action='store_true')
    a = ap.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.revision):
        ap.error('revision must be a full commit SHA')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', a.namespace):
        ap.error('invalid GHCR namespace')
    if not re.fullmatch(r'[0-9][0-9A-Za-z._-]*', a.version):
        ap.error('invalid release version')
    if not re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F-]+)', a.gpus):
        ap.error('qualification requires one GPU index or UUID')
    if a.mode == 'qualify' and a.publish:
        ap.error('--publish is only valid for promotion')
    candidates = {}
    for file in a.candidates.glob('**/candidate-*.json'):
        c = json.loads(file.read_text())
        key = (c['engine'], c['architecture'])
        if key in candidates or c['revision'] != a.revision:
            raise ValueError('duplicate candidate or revision mismatch')
        if key[0] not in ENGINES or key[1] not in ARCHES or not re.fullmatch(r'sha256:[0-9a-f]{64}', c['digest']):
            raise ValueError('invalid candidate')
        candidates[key] = f"ghcr.io/{a.namespace}/kvcached-{key[0]}@{c['digest']}"
    expected = {(e, arch) for e in ENGINES for arch in ARCHES}
    if set(candidates) != expected:
        raise ValueError('all four candidate images are required')
    if a.mode == 'qualify':
        import platform
        arch = {'aarch64': 'arm64', 'x86_64': 'amd64'}.get(platform.machine())
        if a.arch != arch:
            raise ValueError('qualification requires the native architecture')
        # A failed retry must not leave a previous passing evidence file behind.
        if a.evidence.exists() and any(a.evidence.iterdir()):
            raise ValueError('qualification evidence directory must be empty')
        a.evidence.mkdir(parents=True, exist_ok=True)
        images = {e: candidates[e, arch] for e in ENGINES}
        for e, ref in images.items():
            validate_image(inspect(ref, arch), e, arch, a.revision, a.version)
        args = ['python3', 'tests/e2e/run.py', '--profile', 'release', '--no-install',
                '--elastic-budget-mib', '2048', '--gpus', a.gpus, '--out', str(a.evidence)]
        for e, ref in images.items():
            args += ['--image', f'{e}={ref}']
        subprocess.run(args, check=True)
        summary = json.loads((a.evidence / 'summary.json').read_text())
        setup = json.loads((a.evidence / 'setup.json').read_text())
        evidence = {'revision': a.revision, 'architecture': arch, 'images': images,
                    'summary': summary, 'setup': setup}
        for e, ref in images.items():
            validate_evidence(evidence, e, arch, a.revision, ref, a.version)
        (a.evidence / f'evidence-{arch}.json').write_text(json.dumps(evidence, indent=2))
        return
    # Preflight every image and both evidence files BEFORE writing any registry tag.
    for (engine, arch), reference in candidates.items():
        validate_image(inspect(reference, arch), engine, arch, a.revision, a.version)
        files = list(a.evidence.glob(f'**/evidence-{arch}.json'))
        if len(files) != 1:
            raise ValueError(f'exactly one {arch} evidence file is required')
        validate_evidence(json.loads(files[0].read_text()), engine, arch, a.revision, reference,
                          a.version)
    promote(candidates, a.namespace, a.revision, a.version, a.publish)


if __name__ == '__main__':
    main()
