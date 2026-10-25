# vllm-arm-t4g

[English](README.md) · **한국어**

**AWS `g5g`(AWS Graviton2 **arm64** + NVIDIA **T4G**, compute capability **sm_75 / Turing**)에서
[vLLM](https://github.com/vllm-project/vllm)을 빌드하고 구동합니다.**

`g5g`는 AWS에서 가장 저렴한 NVIDIA GPU 인스턴스 계열이지만, 주류 vLLM이 겨냥하지 않는 까다로운
조합 위에 있습니다 — **arm64(aarch64) 호스트** + **Turing(sm_75) GPU**. 업스트림 사전빌드 휠은
x86_64 우선이고, 최신 vLLM은 빠른 경로에 점점 **sm_80+(Ampere)** 를 가정합니다. 이 레포는 실제로
작동하는 **재현 가능한 소스 빌드**와, Turing에 실제로 필요한 런타임 설정을 문서화합니다.

> 범위: 이건 빌드/구동 레시피입니다 — **작은 업스트림 패치 1개 + 버전 핀 + 올바른 런타임 플래그**,
> 포크가 아닙니다. 실제 `g5g.2xlarge`에서 검증됨. 이 빌드 위에서 **VoxCPM2**(오디오/omni)를 서빙하려면
> 동반 레포 **omni-vllm-voxcpm2-arm-t4g** 를 참고하세요.

## TL;DR

```bash
# arm64 g5g 박스(Ubuntu 24.04)에서
git clone https://github.com/moduvoice/vllm-arm-t4g && cd vllm-arm-t4g
./build.sh          # vLLM v0.20.0 클론 → 패치 적용 → sm_75로 빌드
# ... 또는 컨테이너 빌드:
docker build -t vllm-arm-t4g .
```

## 왜 g5g / T4G에서 "그냥" 안 되나

| 문제 | 원인 | 이 레포의 해결 |
|---|---|---|
| `pip install vllm`에 쓸만한 휠 없음 | 이 조합용 arm64 휠 부재 | **arm64 CUDA PyTorch** 휠 기반으로 소스 빌드 |
| 소스 빌드가 FA3 컴파일에서 실패 | vLLM이 `_vllm_fa3_C`(FlashAttention-3, **Hopper 전용**)를 빌드 시도 → sm_75에서 nvcc 오류 | **`patches/0001-disable-fa3-build-for-sm75.patch`** — FA3 확장 건너뜀 |
| 런타임: FlashInfer 크래시 | FlashInfer JIT 커널이 **sm_75** 비호환 | **`attention_backend: TRITON_ATTN`** 로 구동(동반 레포 deploy config 참고) |
| `bfloat16` 오류 | Turing엔 **bf16 없음** | **`dtype: float16`** 사용 |

## 검증 환경 (AWS g5g.2xlarge, 2026-06-13)

| | |
|---|---|
| 인스턴스 / GPU | `g5g.2xlarge` · **NVIDIA T4G 15 GB** · 드라이버 **595.64** |
| OS / 커널 / 아키텍처 | Ubuntu 24.04.4 LTS · `6.17.0-1015-aws` · **aarch64** |
| Python | **3.13** |
| PyTorch | **2.11.0+cu130** (arm64 휠, 아래 참고) + 맞는 torchaudio/torchvision |
| CUDA 툴킷 (빌드용 nvcc) | **12.0** (`CUDA_HOME`); sm_75는 CUDA 12.x에서 완전 지원 |
| 컴파일러 | **g++-12** (`CC=CXX=/usr/bin/g++-12`) |
| vLLM | **v0.20.0** (`88d34c6`) + FA3 패치 |
| 빌드 아키 플래그 | `TORCH_CUDA_ARCH_LIST=7.5` |

> 공식 PyTorch aarch64 CUDA 휠에는 **`sm_75` 커널이 없어** T4G에서 `no kernel image is available`로 실패합니다.
> 아래 AWS 빌드에는 `sm_75`가 들어 있습니다. 전체 버전·검사 결과·pip 인덱스:
> **[pytorch-arm64-t4g](https://github.com/moduvoice/pytorch-arm64-t4g)**.

**arm64 CUDA PyTorch 휠** (AWS 제공, `cu130`, cp313):

```
torch        https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torch-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_aarch64.whl
torchaudio   https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torchaudio-2.11.0%2Bcu130-cp313-cp313-linux_aarch64.whl
torchvision  https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130/torchvision-0.26.0%2Bcu130-cp313-cp313-linux_aarch64.whl
```

빌드 결과의 `torch.cuda.get_arch_list()`에 `sm_75`가 포함됩니다 ✓.

## 소스 빌드 (베어메탈)

[`build.sh`](build.sh) 참고. 동작:
1. 빌드 의존성 설치(`g++-12`, `cmake`, `ninja`, `git`, `numactl`),
2. venv(Python 3.13) 생성 + arm64 `cu130` torch 휠 설치,
3. vLLM `v0.20.0` 클론, `python use_existing_torch.py`(설치된 torch 기준 빌드),
4. `patches/0001-disable-fa3-build-for-sm75.patch` 적용,
5. `TORCH_CUDA_ARCH_LIST=7.5 CC=g++-12 CXX=g++-12 pip install -e . --no-build-isolation` 빌드.

전체 소스 빌드는 `g5g.2xlarge`(4 vCPU)에서 ~30–60분. `MAX_JOBS`/`NVCC_THREADS`로 조절.

## 컨테이너

[`Dockerfile`](Dockerfile)이 동일한 단계를 arm64 CUDA 베이스에서 인코딩합니다. **arm64 호스트**(g5g
박스, 또는 `--platform linux/arm64` 에뮬레이션 — CUDA 빌드는 에뮬레이션이 매우 느림)에서 빌드하세요.

## Turing(sm_75)에서 되는 것 / 안 되는 것

- ✅ vLLM 코어 엔진, `TRITON_ATTN` 어텐션 백엔드, fp16 모델, `enforce_eager`.
- ⚠️ **bf16 없음** — 모델을 `float16`으로 로드.
- ⚠️ **FlashInfer / FA2 / FA3** 빠른 어텐션 경로 불가(sm_80+); `TRITON_ATTN` 사용.
- ⚠️ 일부 모델에서 CUDA graphs가 불안정할 수 있음 — `enforce_eager: true`가 안전한 기본값.
- **vLLM v0.20.0에 핀 고정**: 최신 vLLM은 sm_80 쪽으로 하한을 계속 올림 → 이 레시피는 의도적으로 날짜·버전 고정.

## 패치

```diff
--- a/setup.py
+++ b/setup.py
@@ if _is_cuda():
         # FA3 requires CUDA 12.3 or later
-        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
+        pass  # DISABLED FOR sm_75: FA3 Hopper-only
```

단 한 줄: vLLM은 `CUDA >= 12.3`이면 FlashAttention-3 CUDA 확장을 무조건 큐잉하는데, FA3는
Hopper(sm_90) 전용입니다. Turing에선 nvcc 컴파일이 실패해 전체 빌드가 중단됩니다. 이 확장을
건너뛰면 나머지 vLLM이 빌드되고, 어차피 sm_75에서 FA3는 쓸 수 없습니다.

## 라이선스

vLLM은 Apache-2.0이며, 이 레시피(패치 + 스크립트 + 문서)도 **Apache-2.0** 으로 배포합니다
([`LICENSE`](LICENSE) 참고). vLLM, FlashInfer, PyTorch는 각 저작자 소유이며, 이 레포는 서드파티
바이너리를 **재배포하지 않습니다** — 전부 빌드 시 업스트림에서 가져옵니다.

## 감사의 글

- [vLLM](https://github.com/vllm-project/vllm) — vLLM 프로젝트.
- arm64 CUDA PyTorch 빌드(`framework-binaries` S3) — AWS.
