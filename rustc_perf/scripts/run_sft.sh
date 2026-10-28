#!/bin/bash
# run_sft (rustc-perf): cold-start SFT (step 1/4). Teaches Qwen3-1.7B-Base to emit, from a given
#   per-change state, one format-valid rustc compile-speedup idea covering the 6 categories (B1..B6),
#   used as the GRPO pretrain.
#
# Data (local data/, no external deps):
#   rustc is built "fully hold-out clean" -- sft_train.jsonl(608) / sft_val.jsonl(127) use only train
#   changes (before the last 32 weeks); not a single test (last 32 weeks) gold idea ever enters SFT,
#   so no _clean variant is needed and SPOD_DATA_TAG defaults to empty. Only strong positives are used,
#   with class up-sampling to balance (minority classes capped at 3x, majority classes not truncated).
#
# Output is a standard HF bf16 dir ./output_spod_sft/sft_hf, usable directly as the GRPO pretrain.
#
# WARNING: base must be the -Base suffixed version. The non-suffixed Qwen3 is the instruct/thinking
#   variant and spontaneously emits <think>...</think>, polluting the raw-text spod_raw concatenation.
#   Full-param SFT of 1.7B measures ~27GB/GPU, safe on 2 GPUs.
#
# Usage: NGPU=2 bash scripts/run_sft.sh
set -e
cd "$(dirname "$0")/.."

NGPU="${NGPU:-2}"
SFT_BASE="${SPOD_SFT_BASE:-Qwen/Qwen3-1.7B-Base}"   # rustc base
MODEL_REVISION="${SPOD_MODEL_REVISION:-ea980cb0a6c2ae4b936e82123acc929f1cec04c1}"
DATA_TAG="${SPOD_DATA_TAG:-}"                        # rustc is naturally train-only, default empty
EPOCHS="${SPOD_SFT_EPOCHS:-6}"
BATCH="${SPOD_SFT_BATCH:-2}"
LR="${SPOD_SFT_LR:-1e-5}"
MAX_LEN="${SPOD_SFT_MAX_LEN:-768}"
SEED="${SPOD_SFT_SEED:-42}"

echo "[SFT] base=${SFT_BASE}  revision=${MODEL_REVISION}  ngpu=${NGPU}  data_tag='${DATA_TAG}'  epochs=${EPOCHS}"

torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29500}" \
    entry/start_spod_sft_train.py \
    --pretrain "${SFT_BASE}" \
    --revision "${MODEL_REVISION}" \
    --world_size "${NGPU}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --max_len "${MAX_LEN}" \
    --seed "${SEED}" \
    --data_tag "${DATA_TAG}" \
    --output_dir ./output_spod_sft
