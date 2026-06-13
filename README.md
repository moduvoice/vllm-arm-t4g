# vllm-arm-t4g

**English** · [한국어](README.ko.md)

**Build & run [vLLM](https://github.com/vllm-project/vllm) on AWS `g5g` (AWS Graviton2 **arm64** + NVIDIA **T4G**, compute capability **sm_75 / Turing**).**

`g5g` is the cheapest NVIDIA-GPU instance family on AWS, but it sits at an awkward
intersection that mainstream vLLM does not target: an **arm64 (aarch64)** host **and** a
**Turing (sm_75)** GPU. Upstream prebuilt wheels are x86_64-first, and modern vLLM increasingly
assumes **sm_80+** (Ampere) for its fast paths. This repo documents a **reproducible source build**
that works, plus the runtime settings Turing actually needs.

> Scope: this is a build/run recipe — **one small upstream patch + pinned versions + the right
> runtime flags**, not a fork. Verified on a live `g5g.2xlarge`. To serve **VoxCPM2** (audio/omni)
> on top of this build, see the companion repo **omni-vllm-voxcpm2-arm-t4g**.

## TL;DR

```bash
# on an arm64 g5g box (Ubuntu 24.04)
git clone https://github.com/<you>/vllm-arm-t4g && cd vllm-arm-t4g
./build.sh          # clones vLLM v0.20.0, applies the patch, builds for sm_75
# ... or build the container:
docker build -t vllm-arm-t4g .
```

## Why it doesn't "just work" on g5g / T4G

| Problem | Cause | Fix in this repo |
|---|---|---|
| `pip install vllm` gives no usable wheel | no arm64 wheel for this combo | build from source against an **arm64 CUDA PyTorch** wheel |
| Source build fails compiling FA3 | vLLM tries to build `_vllm_fa3_C` (FlashAttention-3), which is **Hopper-only**; nvcc errors on sm_75 | **`patches/0001-disable-fa3-build-for-sm75.patch`** — skip the FA3 extension |
| Runtime: FlashInfer crashes | FlashInfer JIT kernels are not compatible with **sm_75** | run with **`attention_backend: TRITON_ATTN`** (see companion repo's deploy config) |
| `bfloat16` errors | Turing has **no bf16** | use **`dtype: float16`** |

## Verified environment (AWS g5g.2xlarge, 2026-06-13)

| | |
|---|---|
| Instance / GPU | `g5g.2xlarge` · **NVIDIA T4G 15 GB** · driver **595.64** |
| OS / kernel / arch | Ubuntu 24.04.4 LTS · `6.17.0-1015-aws` · **aarch64** |
| Python | **3.13** |
| PyTorch | **2.11.0+cu130** (arm64 wheel, see below) + matching torchaudio/torchvision |
| CUDA toolkit (nvcc, for the build) | **12.0** (`CUDA_HOME`); sm_75 is fully supported by CUDA 12.x |
| Compiler | **g++-12** (`CC=CXX=/usr/bin/g++-12`) |
| vLLM | **v0.20.0** (`88d34c6`) + the FA3 patch |
| Build arch flag | `TORCH_CUDA_ARCH_LIST=7.5` |

**arm64 CUDA PyTorch wheels** (AWS-published, `cu130`, cp313):

```
torch        https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torch-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_aarch64.whl
torchaudio   https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torchaudio-2.11.0%2Bcu130-cp313-cp313-linux_aarch64.whl
torchvision  https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torchvision-0.26.0%2Bcu130-cp313-cp313-linux_aarch64.whl
```

`torch.cuda.get_arch_list()` on the resulting build includes `sm_75` ✓.

## Build from source (bare metal)

See [`build.sh`](build.sh). It:
1. installs build deps (`g++-12`, `cmake`, `ninja`, `git`, `numactl`),
2. creates a venv (Python 3.13) and installs the arm64 `cu130` torch wheels,
3. clones vLLM `v0.20.0`, runs `python use_existing_torch.py` (build against the installed torch),
4. applies `patches/0001-disable-fa3-build-for-sm75.patch`,
5. builds with `TORCH_CUDA_ARCH_LIST=7.5 CC=g++-12 CXX=g++-12 pip install -e . --no-build-isolation`.

A full source build takes ~30–60 min on `g5g.2xlarge` (4 vCPU). Use `MAX_JOBS`/`NVCC_THREADS` to tune.

## Container

[`Dockerfile`](Dockerfile) encodes the same steps from an arm64 CUDA base. Build it **on an arm64
host** (a g5g box, or `--platform linux/arm64` with emulation — emulation is very slow for a CUDA build).

## What works / what doesn't on Turing (sm_75)

- ✅ Core vLLM engine, `TRITON_ATTN` attention backend, fp16 models, `enforce_eager`.
- ⚠️ **No bf16** — load models in `float16`.
- ⚠️ **No FlashInfer / FA2 / FA3** fast-attention paths (sm_80+); use `TRITON_ATTN`.
- ⚠️ CUDA graphs can be brittle on this stack for some models — `enforce_eager: true` is the safe default.
- Pin to **vLLM v0.20.0**: newer vLLM keeps raising the floor toward sm_80; this recipe is dated and version-pinned on purpose.

## The patch

```diff
--- a/setup.py
+++ b/setup.py
@@ if _is_cuda():
         # FA3 requires CUDA 12.3 or later
-        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
+        pass  # DISABLED FOR sm_75: FA3 Hopper-only
```

One line: vLLM unconditionally queues the FlashAttention-3 CUDA extension when `CUDA >= 12.3`, but
FA3 targets Hopper (sm_90). On Turing the nvcc compile fails and aborts the whole build. Skipping the
extension lets the rest of vLLM build; FA3 is never usable on sm_75 anyway.

## License

vLLM is Apache-2.0; this recipe (patch + scripts + docs) is released under **Apache-2.0** — see
[`LICENSE`](LICENSE). vLLM, FlashInfer, and PyTorch are the property of their respective authors;
this repo redistributes **no** third-party binaries — everything is fetched from its upstream source
at build time.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) by the vLLM project.
- AWS for the arm64 CUDA PyTorch builds (`framework-binaries` S3).
