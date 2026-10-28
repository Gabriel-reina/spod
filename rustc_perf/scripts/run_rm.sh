#!/bin/bash
# run_rm (rustc-perf): proxy Reward Model (step 2/4). Bradley-Terry pairwise, produces a standard HF
#   token-classification head (num_labels=1, last-token logit as the response score).
#
# Data (local data/, no external deps):
#   rustc is fully hold-out clean -- rm_items_prompted.jsonl(1414 train-only: strong 493 / null 574
#   / reject 347) + rm_pairs_slim_train.jsonl(49523) / rm_pairs_slim_val.jsonl(7728).
#   No test-change item ever enters a pair (0 leakage, with positive control), so no _clean variant is
#   needed and DATA_TAG defaults to empty.
#
# WARNING: the RM never does class rebalancing. Class bias is exactly the disease SPOD measures and
#   corrects, so it must remain in the RM.
# Recipe: unfreeze last 4 layers + BT loss + grouped lr (head 1e-3 / llm 1e-5) + cosine (warmup 1ep)
#   + 3ep + maxlen768. Output ./output_spod_rm/rm_hf, consumed directly by GRPO-side hf_infer.
#
# WARNING: base is Qwen/Qwen3-1.7B-Base (-Base suffix). Qwen3ForTokenClassification inherits the same
#   implementation as Qwen2 (verified on transformers 4.57), architecture-compatible. 1.7B unfreezes
#   only the last 4 layers, safe on 2 GPUs.
#
# Usage: NGPU=2 bash scripts/run_rm.sh
set -e
cd "$(dirname "$0")/.."

NGPU="${NGPU:-2}"
RM_BASE="${SPOD_RM_BASE:-Qwen/Qwen3-1.7B-Base}"   # same base as SFT
MODEL_REVISION="${SPOD_MODEL_REVISION:-ea980cb0a6c2ae4b936e82123acc929f1cec04c1}"
DATA_TAG="${SPOD_DATA_TAG:-}"                     # rustc is naturally train-only, default empty
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
