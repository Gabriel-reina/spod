#!/bin/bash
# run_convert: synthesize a given arm's GRPO Megatron ckpt into HF bf16 (single GPU).
#   The GRPO side writes Megatron distributed ckpts (checkpoint-{step}) that AutoModelForCausalLM
#   cannot read, so this step is required before evaluation. Uses the mcore_adapter conversion tool
#   shipped with the public ROLL framework.
#
# Usage: bash scripts/run_convert.sh <arm> [step]
#   e.g.: bash scripts/run_convert.sh bhap 49
#   step = 49 (last step, default) or 25 (mid-run); corresponds to save_steps=25 in the yaml.
set -e
cd "$(dirname "$0")/.."

ARM="${1:-${SPOD_ARM:-v}}"
STEP="${2:-${SPOD_CKPT_STEP:-49}}"

CKPT_DIR="${SPOD_CKPT_DIR:-./output/grpo/${ARM}/ckpt/checkpoint-${STEP}}"
OUT_DIR="${SPOD_CONVERT_OUT:-./output/grpo_hf/${ARM}_step${STEP}}"

echo "[CONVERT] arm=${ARM} step=${STEP}"
echo "[CONVERT] in =${CKPT_DIR}"
echo "[CONVERT] out=${OUT_DIR}"

# Uses the installed public mcore_adapter package; no checkout-relative tool path.
python entry/convert_checkpoint.py \
    --checkpoint_path "${CKPT_DIR}" \
    --output_path "${OUT_DIR}" \
    --bf16
