#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate immutable candidate images and GPU evidence before promotion."""
from __future__ import annotations

import argparse
import base64
import hashlib
import urllib.request
import urllib.parse
import json
import os
import re
import subprocess
from pathlib import Path

ENGINES = {'vllm': '0.30.0', 'sglang': '0.5.20'}
ARCHES = ('amd64', 'arm64')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def inspect(reference, arch):
    """Read manifest/config by digest without pulling tens of GB of layers."""
    registry_path, digest = reference.removeprefix('ghcr.io/').split('@')
    query = urllib.parse.urlencode({'service': 'ghcr.io', 'scope': f'repository:{registry_path}:pull'})
    headers = {}
    if os.environ.get('GH_TOKEN'):
        credentials = f"{os.environ.get('GITHUB_ACTOR', 'token')}:{os.environ['GH_TOKEN']}"
        headers['Authorization'] = 'Basic ' + base64.b64encode(credentials.encode()).decode()
    request = urllib.request.Request('https://ghcr.io/token?' + query, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.load(response)['token']

    def fetch(kind, sha):
        request = urllib.request.Request(f'https://ghcr.io/v2/{registry_path}/{kind}/{sha}', headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        if 'sha256:' + hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('registry content digest mismatch')
        return json.loads(payload)

    manifest = fetch('manifests', digest)
    config = fetch('blobs', manifest['config']['digest'])
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


def validate_evidence(evidence, engine, arch, revision, reference):
    if (evidence.get('revision') != revision or evidence.get('architecture') != arch
            or evidence.get('images', {}).get(engine) != reference):
        raise ValueError('GPU evidence does not match candidate digest/revision/platform')
    if evidence.get('summary', {}).get('ok') is not True:
        raise ValueError('GPU qualification did not pass')
    cases = evidence.get('summary', {}).get('cases', [])
    # Summary shape is deliberately checked, not just a caller-supplied passed flag.
    required = {f'{engine}-setup', f'{engine}-release05b-kv_c1',
                f'{engine}-release05b-native', f'elastic-{engine}-release05b+{engine}-release05b'}
    selected = {c['name']: c for c in cases if c.get('name') in required}
    if selected.keys() != required or not all(c.get('ok') and c.get('checks') for c in selected.values()):
        raise ValueError('missing or failing GPU qualification cases')
    for name, case in selected.items():
        checks = {c['name'] for c in case['checks']}
        if name.endswith('-setup'):
            mandatory = {'installed_package_verified'}
        elif name.startswith('elastic-'):
            mandatory = {'a_requests_ok', 'b_requests_ok', 'a_filled_memory', 'a_released_memory',
                         'b_reused_budget', 'a_physical_memory_released', 'contended_requests_ok', 'memory_sampling'}
        else:
            mandatory = {'server_ready', 'server_alive_after_load', 'prefix_cache_reset', 'no_error_signatures'}
            if name.endswith('-kv_c1'):
                mandatory |= {'kvcached_active', 'gen_seq_tokens_equal'}
        if not mandatory <= checks:
            raise ValueError('missing required GPU qualification checks')
    if not all(check.get('ok') is True for c in selected.values() for check in c['checks']):
        raise ValueError('GPU qualification contains failing checks')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['qualify', 'promote'])
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--version', default='0.1.6')
    ap.add_argument('--arch', choices=ARCHES)
    ap.add_argument('--namespace', required=True)
    ap.add_argument('--publish', action='store_true')
    a = ap.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.revision):
        ap.error('revision must be a full commit SHA')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', a.namespace):
        ap.error('invalid GHCR namespace')
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
        a.evidence.mkdir(parents=True, exist_ok=True)
        images = {e: candidates[e, arch] for e in ENGINES}
        for e, ref in images.items():
            validate_image(inspect(ref, arch), e, arch, a.revision, a.version)
        args = ['python3', 'tests/e2e/run.py', '--profile', 'release', '--no-install',
                '--elastic-budget-mib', '2048', '--gpus', '0', '--out', str(a.evidence)]
        for e, ref in images.items():
            args += ['--image', f'{e}={ref}']
        subprocess.run(args, check=True)
        summary = json.loads((a.evidence / 'summary.json').read_text())
        evidence = {'revision': a.revision, 'architecture': arch, 'images': images, 'summary': summary}
        for e, ref in images.items():
            validate_evidence(evidence, e, arch, a.revision, ref)
        (a.evidence / f'evidence-{arch}.json').write_text(json.dumps(evidence, indent=2))
        return
    # Preflight every image and both evidence files BEFORE writing any public tag.
    for (engine, arch), reference in candidates.items():
        validate_image(inspect(reference, arch), engine, arch, a.revision, a.version)
        files = list(a.evidence.glob(f'**/evidence-{arch}.json'))
        if len(files) != 1:
            raise ValueError(f'exactly one {arch} evidence file is required')
        validate_evidence(json.loads(files[0].read_text()), engine, arch, a.revision, reference)
    for engine, version in ENGINES.items():
        tag = f'ghcr.io/{a.namespace}/kvcached-{engine}:kvcached-v{a.version}-{engine}-v{version}'
        args = ['docker', 'buildx', 'imagetools', 'create', '--tag', tag,
                *(candidates[engine, arch] for arch in ARCHES)]
        if a.publish:
            subprocess.run(args, check=True)
        else:
            print(json.dumps(args))
    if a.publish:
        # latest only after both immutable version manifests were created.
        for engine, version in ENGINES.items():
            repo = f'ghcr.io/{a.namespace}/kvcached-{engine}'
            subprocess.run(['docker', 'buildx', 'imagetools', 'create', '--tag', f'{repo}:latest',
                            f'{repo}:kvcached-v{a.version}-{engine}-v{version}'], check=True)


if __name__ == '__main__':
    main()
