#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROLL_DIR="${1:-$ROOT/../ROLL}"
REV="$(cat "$ROOT/configs/roll_revision.txt" | tr -d '\n\r')"
if [ ! -d "$ROLL_DIR" ]; then
  git clone https://github.com/alibaba/ROLL.git "$ROLL_DIR"
fi
git -C "$ROLL_DIR" checkout "$REV"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVTE_FRAMEWORK=pytorch
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-2}"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil pybind11
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 vllm==0.11.0 transformers==4.57.0 'numpy<2'
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export CUDNN_PATH="$SITE/nvidia/cudnn"
python -m pip install --no-build-isolation flash-attn==2.8.1 'transformer-engine[pytorch]==2.8.0'
python -m pip install -r "$ROLL_DIR/requirements_torch280_vllm.txt" -r "$ROOT/configs/requirements_roll_runtime.txt" tensorboard
python -m pip install --no-deps -e "$ROLL_DIR" -e "$ROLL_DIR/mcore_adapter"
ray --version
python -c 'import torch, transformers, vllm, roll, mcore_adapter; print(torch.__version__, transformers.__version__, vllm.__version__)'
