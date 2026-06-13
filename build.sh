#!/usr/bin/env bash
# Build vLLM v0.20.0 for AWS g5g — arm64 (Graviton2) + NVIDIA T4G (sm_75 / Turing).
# Reconstructed from a verified g5g.2xlarge build (Ubuntu 24.04, Python 3.13, torch 2.11.0+cu130).
# Run ON an arm64 g5g box.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

VLLM_REF="${VLLM_REF:-v0.20.0}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"          # T4G = sm_75
WORK="${WORK:-$HOME/vllm-arm-t4g-build}"
TORCH_BASE="https://framework-binaries.s3.us-west-2.amazonaws.com/pytorch/v2.11.0/arm64/cu130"

[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: this recipe targets arm64 (g5g). uname=$(uname -m)"; exit 1; }
mkdir -p "$WORK"

echo "== [1/5] system build deps (sudo) =="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates build-essential g++-12 gcc-12 cmake ninja-build numactl \
  python3.13 python3.13-venv python3.13-dev

echo "== [2/5] venv + arm64 cu130 PyTorch =="
python3.13 -m venv "$WORK/.venv"
. "$WORK/.venv/bin/activate"
pip install -U pip wheel setuptools
pip install \
  "$TORCH_BASE/torch-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_aarch64.whl" \
  "$TORCH_BASE/torchaudio-2.11.0%2Bcu130-cp313-cp313-linux_aarch64.whl" \
  "$TORCH_BASE/torchvision-0.26.0%2Bcu130-cp313-cp313-linux_aarch64.whl"
python -c "import torch; print('torch', torch.__version__, '| arch_list', torch.cuda.get_arch_list())"

echo "== [3/5] clone vLLM $VLLM_REF =="
rm -rf "$WORK/vllm"
git clone --depth 1 --branch "$VLLM_REF" https://github.com/vllm-project/vllm.git "$WORK/vllm"
cd "$WORK/vllm"

echo "== [4/5] build against installed torch + apply sm_75 patch =="
python use_existing_torch.py                         # strip torch pins; build vs the installed wheel
git apply "$HERE/patches/0001-disable-fa3-build-for-sm75.patch"

echo "== [5/5] compile vLLM for sm_75 =="
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export CC=/usr/bin/g++-12 CXX=/usr/bin/g++-12
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export VLLM_FLASH_ATTN_VERSION=""                    # do not request FA2/FA3 on Turing
pip install -e . --no-build-isolation -v             # ~30-60 min on g5g.2xlarge (4 vCPU)

python -c "import vllm; print('vLLM build OK:', vllm.__version__)"
echo "Done. Activate with:  source $WORK/.venv/bin/activate"
