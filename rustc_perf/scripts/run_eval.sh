#!/bin/bash
# run_eval (rustc-perf): offline evaluation of an arm's GRPO output (single GPU). Generation +
#   RM scoring, written per-item; aggregation and cross-arm win rate are left to spod_analyze.py.
#   Requires: this arm has finished GRPO and been merged to HF via scripts/run_convert.sh.
#   Special case arm=sft: evaluate the SFT cold-start model directly (the pre-GRPO starting point),
#   as the common reference for all arms.
#
# Eval set: default is the 177 hold-out items (grpo_prompts_test.jsonl, changes in the last 32 weeks,
#   never seen by any arm during training). To go back to the train-set self-eval convention, pass
#   SPOD_EVAL_PROMPTS=grpo_prompts_train.jsonl.
#
# Usage: bash scripts/run_eval.sh <arm> [step]
#   e.g.: bash scripts/run_eval.sh bhap 49
set -e
cd "$(dirname "$0")/.."

ARM="${1:-${SPOD_ARM:-v}}"
STEP="${2:-${SPOD_CKPT_STEP:-49}}"

if [ "${ARM}" = "sft" ]; then
    ACTOR="${SPOD_ACTOR:-./output_spod_sft/sft_hf}"
    EVAL_TAG="sft"
else
    ACTOR="${SPOD_ACTOR:-./output/grpo_hf/${ARM}_step${STEP}}"
    EVAL_TAG="${ARM}_step${STEP}"
fi
# the scoring RM must be identical across arms, otherwise cross-arm scores are not comparable
RM="${SPOD_RM:-./output_spod_rm/rm_hf}"
K="${SPOD_EVAL_K:-8}"
PROMPTS="${SPOD_EVAL_PROMPTS:-grpo_prompts_test.jsonl}"

echo "[EVAL] actor=${ACTOR}"
echo "[EVAL] rm=${RM}  tag=${EVAL_TAG}  k=${K}  set=${PROMPTS}"

python entry/start_spod_eval.py \
    --actor "${ACTOR}" \
    --rm "${RM}" \
    --tag "${EVAL_TAG}" \
    --k "${K}" \
    --seed "${SPOD_SEED:-42}" \
    --prompts "${PROMPTS}" \
    --out_dir ./output/spod_eval
