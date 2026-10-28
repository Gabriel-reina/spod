#!/bin/bash
# run_sft: cold-start SFT (step 1/4). Makes Qwen2.5-0.5B able to emit, for a given baseline state,
#   a format-valid speedup idea covering the 5 categories, to serve as GRPO's pretrain.
#
# Data (local data/, no external dependency):
#   SPOD_DATA_TAG=_clean (default, used by open-source): sft_train_clean.jsonl / sft_val_clean.jsonl
#     -- all items attached to hold-out test28 states are removed; together with grpo_prompts_test28
#        this forms a full-pipeline hold-out.
#   SPOD_DATA_TAG="" (full): sft_train.jsonl / sft_val.jsonl -- includes test-state examples; use _clean for held-out evaluation.
# Uses positive samples only, categories already upsampled to balance (minority cap 3x, majority not cut).
#
# Output is a standard HF bf16 dir ./output_spod_sft/sft_hf, usable directly as GRPO's pretrain.
#
# Usage: NGPU=2 bash scripts/run_sft.sh
set -e
cd "$(dirname "$0")/.."

NGPU="${NGPU:-2}"
SFT_BASE="${SPOD_SFT_BASE:-Qwen/Qwen2.5-0.5B}"   # base model, default 0.5B; for Qwen3 use the -Base suffixed version
MODEL_REVISION="${SPOD_MODEL_REVISION:-060db6499f32faf8b98477b0a26969ef7d8b9987}"
DATA_TAG="${SPOD_DATA_TAG:-_clean}"
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
