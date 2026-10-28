#!/bin/bash
# run_rm: proxy Reward Model (step 2/4). Bradley-Terry pairwise, produces a standard HF
#   token-classification head (num_labels=1, last-token logit as the response score).
#
# Data (local data/, no external dependency):
#   SPOD_DATA_TAG=_clean (default, used by open-source): rm_items_prompted_clean.jsonl (148 items) + 5280/108 pairs
#     -- test28 leakage removed.
#   SPOD_DATA_TAG="" (full): rm_items_prompted.jsonl (190 items) + 8229/284 pairs -- use _clean for held-out evaluation.
#
# WARNING RM never does category rebalancing: the category bias (C1 33.6% vs C5 4.8%) is exactly the
#   disease SPOD is meant to measure and correct.
# Recipe: unfreeze last 4 layers + BT loss + grouped lr (head 1e-3 / llm 1e-5) + cosine + 3ep + maxlen768.
# Output ./output_spod_rm/rm_hf, consumed directly by GRPO-side hf_infer.
#
# Usage: NGPU=2 bash scripts/run_rm.sh
set -e
cd "$(dirname "$0")/.."

NGPU="${NGPU:-2}"
RM_BASE="${SPOD_RM_BASE:-Qwen/Qwen2.5-0.5B}"   # same base model as SFT
MODEL_REVISION="${SPOD_MODEL_REVISION:-060db6499f32faf8b98477b0a26969ef7d8b9987}"
DATA_TAG="${SPOD_DATA_TAG:-_clean}"
EPOCHS="${SPOD_RM_EPOCHS:-3}"
PAIR_BATCH="${SPOD_RM_PAIR_BATCH:-8}"
MAX_LEN="${SPOD_RM_MAX_LEN:-768}"
SEED="${SPOD_RM_SEED:-42}"

echo "[RM] base=${RM_BASE}  revision=${MODEL_REVISION}  ngpu=${NGPU}  data_tag='${DATA_TAG}'  epochs=${EPOCHS}"

torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29501}" \
    entry/start_spod_rm_train.py \
    --pretrain "${RM_BASE}" \
    --revision "${MODEL_REVISION}" \
    --world_size "${NGPU}" \
    --epochs "${EPOCHS}" \
    --pair_batch "${PAIR_BATCH}" \
    --max_len "${MAX_LEN}" \
    --seed "${SEED}" \
    --data_tag "${DATA_TAG}" \
    --output_dir ./output_spod_rm
