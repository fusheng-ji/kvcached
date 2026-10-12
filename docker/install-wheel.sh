#!/usr/bin/env bash
set -euo pipefail
# Large package indexes can exceed pip's 15-second default on remote builders.
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
# Resolve the toolkit's driver stub on both SBSA and x86_64 images. Never put
# it in LD_LIBRARY_PATH: runtime must resolve the driver's real libcuda.
stub=$(find -L /usr/local/cuda -name libcuda.so -path '*/stubs/*' -print -quit)
test -n "$stub"
export LIBRARY_PATH="$(dirname "$stub")${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
python3 - <<'SNAPSHOT'
import importlib.metadata as md, json
versions = {}
for package in ('numpy', 'posix-ipc', 'wrapt'):
    try:
        versions[package] = md.version(package)
    except md.PackageNotFoundError:
        pass
with open('/tmp/base-runtime-versions.json', 'w') as f:
    json.dump(versions, f)
SNAPSHOT
python3 -m pip check > /tmp/dependencies-before.txt 2>&1 || true
python3 -m pip install --no-cache-dir 'setuptools>=64' wheel ninja
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/wheels /opt/kvcached-src
python3 -m pip install --no-cache-dir /tmp/wheels/kvcached-*.whl
python3 -m pip check > /tmp/dependencies-after.txt 2>&1 || true
python3 - <<'CHECK'
from pathlib import Path
before = set(Path('/tmp/dependencies-before.txt').read_text().splitlines())
after = set(Path('/tmp/dependencies-after.txt').read_text().splitlines())
new = after - before - {'No broken requirements found.'}
if before - {'No broken requirements found.'}:
    print('Pre-existing base-image dependency findings:', *sorted(before), sep='\n')
if new:
    raise SystemExit('New dependency failures: ' + '\n'.join(sorted(new)))
CHECK

# Carry only added/changed dependencies: packages already in the pinned base
# need no download, and some engine distributions are unavailable on PyPI.
mapfile -t added < <(python3 - <<'RESOLVED'
import importlib.metadata as md, json
before = json.load(open('/tmp/base-runtime-versions.json'))
for package in ('numpy', 'posix-ipc', 'wrapt'):
    version = md.version(package)
    if before.get(package) != version:
        print(f'{package}=={version}')
RESOLVED
)
if (( ${#added[@]} )); then
  python3 -m pip download --only-binary=:all: --no-deps --dest /tmp/wheels "${added[@]}"
fi
