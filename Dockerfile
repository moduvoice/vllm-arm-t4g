# vLLM v0.20.0 for AWS g5g — arm64 (Graviton2) + NVIDIA T4G (sm_75).
# Build ON an arm64 host (a g5g box). Encodes build.sh; CUDA 12.x toolkit is only
# needed for nvcc (sm_75 builds with any CUDA 12.x). PyTorch is the arm64 cu130 wheel.
#
#   docker build -t vllm-arm-t4g .
#
# NOTE: the reference build was verified bare-metal on g5g.2xlarge (CUDA 12.0 toolkit,
# Python 3.13, torch 2.11.0+cu130). This container has not yet been re-verified from
# scratch in CI — treat the bare-metal build.sh as the source of truth.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG VLLM_REF=v0.20.0
ARG TORCH_CUDA_ARCH_LIST=7.5
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CC=/usr/bin/g++-12 CXX=/usr/bin/g++-12 \
    TORCH_BASE=https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential g++-12 gcc-12 cmake ninja-build numactl \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.13 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install -U pip wheel setuptools && \
    pip install \
      "${TORCH_BASE}/torch-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_aarch64.whl" \
      "${TORCH_BASE}/torchaudio-2.11.0%2Bcu130-cp313-cp313-linux_aarch64.whl" \
      "${TORCH_BASE}/torchvision-0.26.0%2Bcu130-cp313-cp313-linux_aarch64.whl"

COPY patches/ /tmp/patches/
RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git /opt/vllm && \
    cd /opt/vllm && \
    python use_existing_torch.py && \
    git apply /tmp/patches/0001-disable-fa3-build-for-sm75.patch && \
    MAX_JOBS=$(nproc) pip install -e . --no-build-isolation -v

RUN python -c "import vllm, torch; print('vLLM', vllm.__version__, '| torch arch', torch.cuda.get_arch_list())"
WORKDIR /opt/vllm
