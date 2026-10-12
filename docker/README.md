# Try kvcached Using Docker Images

This directory contains Dockerfiles for running **kvcached** together with either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/lm-sys/sglang). Both images are built upon existing SGLang or vLLM images, and installed with needed patched and kvcached.

We also provide a development image that has both vLLM and SGLang.

---

## 1. Image names & tags

| Engine | Public image | Default tag |
| ------ | ------------ | ----------- |
| vLLM   | `ghcr.io/ovg-project/kvcached-vllm`   | `latest`, `kvcached-v0.1.6-vllm-v0.30.0` |
| SGLang | `ghcr.io/ovg-project/kvcached-sglang` | `latest`, `kvcached-v0.1.6-sglang-v0.5.20` |
| vLLM+SGLang | `ghcr.io/ovg-project/kvcached-dev` | `latest`, `v0.1.6` |

## 2. Pulling a pre-built image

```bash
# vLLM engine
docker pull ghcr.io/ovg-project/kvcached-vllm:latest

# SGLang engine
docker pull ghcr.io/ovg-project/kvcached-sglang:latest

# vLLM+SGLang kvcached development
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

## 3. Running the containers

We use vLLM as an example.

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --env "HF_TOKEN=<secret>" \
  -v /dev/shm:/shm \
  --ipc=host \
  --network=host \
  --privileged \
  --name kvcached-vllm \
  ghcr.io/ovg-project/kvcached-vllm \
  bash
```

## 4 Performing a benchmark

Attach to the container first.

```bash
docker exec -it kvcached-vllm bash
```

Then, you can use it as a normal vLLM container, e.g., running benchmarks.

For example, you can run the following command to start a vLLM server and run benchmarks.

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
vllm serve meta-llama/Llama-3.2-1B --no-enable-prefix-caching --port=12346 --tensor-parallel-size=1
vllm bench serve --model meta-llama/Llama-3.2-1B --request-rate 10 --num-prompts 1000 --port 12346
```

NOTE: If installed correctly, you should see that kvcached patches the vLLM:

```
[kvcached][INFO][xxxx-xx-xx xx:xx:xx][patch_base.py:98] Applying 16 patches for vllm
[kvcached][INFO][xxxx-xx-xx xx:xx:xx][version_utils.py:199] Detected vllm version: 0.30.0
[kvcached][INFO][xxxx-xx-xx xx:xx:xx][patch_base.py:178] Successfully patched vllm: nixl_connector_compat, elastic_block_pool, engine_core, mp_client, core_engine_proc_manager, model_runner_v2, kv_layout_v2, gpu_worker, hybrid_block_size_align, kv_cache_coordinator, allocate_slots, mamba_partial_tail
```

Another way to verify is to check the memory consumption using `nvidia-smi`. With kvcached, you should see that the memory usage is closer to model weight size when there are no requests.

## 5. Building the image locally (optional)

If you have modified the source code or want to build for a different base CUDA version you can rebuild the image yourself:

```bash
# Build vLLM image
docker build -f docker/Dockerfile.vllm -t kvcached-vllm .

# Build SGLang image
docker build -f docker/Dockerfile.sglang -t kvcached-sglang .

# Build development image
docker build -f docker/Dockerfile.dev -t kvcached-dev .
```

## Native ARM64 and AMD64 release images

The engine Dockerfiles install a wheel built from **this checkout**, not the
latest package on PyPI. The upstream vLLM 0.30.0 and SGLang 0.5.20 multi-platform
base images are pinned by digest. The extension is compiled natively on each
architecture; no QEMU or cross-architecture GPU emulation is used. CUDA driver
stubs are used only during linking. The runtime uses the host NVIDIA driver.
The combined development image is outside this release workflow. The build
records pre-existing dependency conflicts in the upstream images and rejects
new conflicts introduced by installing kvcached; it does not silently upgrade
the engine dependency stack to repair unrelated upstream issues.

Build on the target architecture (use `amd64` on an x86_64 host):

```bash
revision=$(git rev-parse HEAD)
for engine in vllm sglang; do
  docker build --platform linux/arm64 -f "docker/Dockerfile.$engine" \
    --build-arg SOURCE_REVISION="$revision" \
    --build-arg SOURCE_URL=https://github.com/fusheng-ji/kvcached \
    -t "kvcached-$engine:qualification" .
done
```

### Qualify the actual installed images

Use an idle NVIDIA GPU, a compatible NVIDIA container runtime and Python 3.10+
on the host. The public Qwen2.5-0.5B-Instruct model downloads into the Hugging Face
cache on the first run. Do not run qualification alongside a training job.

```bash
python3 tests/e2e/run.py --profile release --gpus 0 --no-install \
  --image vllm=kvcached-vllm:qualification \
  --image sglang=kvcached-sglang:qualification \
  --elastic-budget-mib 2048 --out /tmp/kvcached-qualification
```

Use the same command on AMD64. `--engines vllm` or `--engines sglang` selects one
engine while debugging; full release qualification requires both. Image overrides
require `--no-install` and reject local-host mode so tests cannot silently replace
the installed package with the checkout. Installation checks load the native
extension, check its ELF architecture and verify automatic patch hooks. Serving
checks compare greedy tokens and prefix-cache behavior against the native engine.

On GB10, `nvidia-smi` may report `N/A` for memory. The sampler then keeps a CUDA
primary context on the GPU selected by PCI identity and reads `cuMemGetInfo`.
Errors in initialization or the sampling thread fail qualification. These CUDA
counters on unified-memory systems are **not** Linux `MemAvailable`.

The bounded elastic test sets each instance's KV limit to 2 GiB, tests allocation,
release of at least half the observed growth after cache reset (including a
driver-visible memory decrease, with 64 MiB accounting tolerance), subsequent growth
by the second instance, concurrent requests and unchanged greedy probes. It uses
no memory ballast. Its `b_reused_budget` assertion measures reuse within a logical
KV budget, **not proof of device-wide exhaustion or reuse of the same physical
pages**. The existing unbounded test retains its device-pressure assertion.
Use the latter only on a dedicated GPU with enough load to fill it. Logs, versions,
model revisions, memory source, time series and checks are written to the output
directory. A missing or failed GPU run is not a passing release qualification.

### Candidate build → GPU qualification → promotion

All three workflows must run from the **same commit**. Repository owners are
lowercased for GHCR, so forks publish to their own namespace automatically.

1. Configure `IMAGE_AMD64_RUNNER` and `IMAGE_ARM64_RUNNER` for native runners with
   at least 100 GiB free disk and sufficient RAM (16 GiB minimum recommended).
   Defaults are `ubuntu-24.04` and `ubuntu-24.04-arm`; standard runners may fail
   the explicit disk preflight. Use larger hosted runners for PR builds; never
   expose a privileged self-hosted runner to untrusted PR code. PR runs build
   only and do not log into GHCR or push packages.
2. Dispatch **Container candidates**. All four engine/architecture jobs must
   succeed. Each publishes a commit/architecture candidate tag and uploads its
   immutable digest. Candidates are not official version tags.
3. Configure dedicated native NVIDIA GPU runners with labels
   `[self-hosted, Linux, ARM64, kvcached-release-gpu]` and
   `[self-hosted, Linux, X64, kvcached-release-gpu]`, or override
   `IMAGE_ARM64_GPU_RUNNER` / `IMAGE_AMD64_GPU_RUNNER` with JSON label arrays.
   They need Docker, the NVIDIA runtime, Python, `gh`, model-network access and
   disk space for both images. Dispatch **Qualify container candidates** once
   per architecture using the successful candidate run ID. Each tests exact
   digests, without reinstalling kvcached, and uploads GPU evidence and logs.
4. Inspect both results. Configure the `container-release` GitHub environment
   with required reviewers for human approval. Dispatch **Promote qualified
   containers** with the candidate run ID and both qualification run IDs.
   It verifies successful workflow provenance, matching commits, four image
   platforms/version labels and complete passing evidence bound to each digest.
   Only then does it create the two multi-platform version manifests and update
   `latest`. Metadata validation does not pull image layers.

Tags retain the existing format, for example
`kvcached-v0.1.6-vllm-v0.30.0`. Update Dockerfile version labels, the qualification
matrix and `tools/release/images.py` together for a future engine/package release.
Preflight failures change no official tags. Registry writes across two image
repositories are not transactional: if the registry fails during promotion,
re-run the same validated inputs; `latest` is updated only after both version
manifests have been created. Artifacts expire after 30 days, so rebuild/requalify
when evidence is no longer available.

For a promotion dry run, download the candidate and evidence artifacts and run:

```bash
python3 tools/release/images.py promote --candidates candidates --evidence evidence \
  --revision "$(git rev-parse HEAD)" --namespace fusheng-ji
```

Set `GH_TOKEN` for private GHCR packages. Publishing requires the explicit
`--publish` option; do not use it for local qualification.
