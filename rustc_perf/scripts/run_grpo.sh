#!/bin/bash
# run_grpo (rustc-perf): SPOD online GRPO (step 3/4). Both arms share configs/spod_grpo.yaml; arm
#   differences all go through environment variables, the yaml is untouched -- the premise of a
#   single-variable comparison is that every byte other than the SPOD switch is identical.
#
# Unified operator (in-group debiasing at the reward layer, SPOD always runs MIN=0):
#   d_i  = beta * std(r_group) * phi(c_i)
#   r'_i = r_i                if i is the in-class top1 (SPOD_EXEMPT=top1, peak preservation)
#        = r_i - d_i          otherwise (cold-tail items may be raised)
#
# Two arms (same base, single-variable comparable):
#   v      pure GRPO baseline (SPOD off), the target
#   bhap   SPOD main method: rmscore / MIN=0 / EXEMPT=top1 / beta=1.0
#
# Output is a Megatron distributed ckpt (checkpoint-{step}); before eval it must pass through
#   scripts/run_convert.sh to synthesize HF.
#
# Usage: bash scripts/run_grpo.sh <arm>
#   e.g.: bash scripts/run_grpo.sh bhap
set -e
cd "$(dirname "$0")/.."

ARM="${1:-${SPOD_ARM:-v}}"

# Training outputs (local HF dirs). Default to the outputs of run_sft.sh / run_rm.sh; override via env.
PRETRAIN="${SPOD_PRETRAIN:-./output_spod_sft/sft_hf}"
REWARD_PRETRAIN="${SPOD_REWARD_PRETRAIN:-./output_spod_rm/rm_hf}"

# Per-arm output dir (local): category distribution / phi / per-item detail
OUT_DIR="${SPOD_OUT_DIR:-./output/spod_grpo/${ARM}}"
BETA="${SPOD_BETA:-1.0}"
GROUP="${SPOD_GROUP:-8}"

CONFIG_NAME="spod_grpo"      # single yaml, shared by both arms

case "${ARM}" in
    v)     SPOD_SET="SPOD_ENABLE=0" ;;
    bhap)  SPOD_SET="SPOD_ENABLE=1,SPOD_POP=rmscore,SPOD_MIN=0,SPOD_EXEMPT=top1"; BETA=1.0 ;;
    *) echo "[FATAL] unknown arm ARM=${ARM}, choose v/bhap"; exit 1 ;;
esac

# ---- assemble environment variables (reward worker reads os.environ) ----
# baseline arm short-circuits the whole SPOD module, passing only the switch; the method arm (bhap)
# additionally carries beta etc.
case "${ARM}" in
    v) ENV_VARS="${SPOD_SET}" ;;
    *) ENV_VARS="${SPOD_SET},SPOD_BETA=${BETA},SPOD_GROUP=${GROUP},SPOD_OUT_DIR=${OUT_DIR},SPOD_TAG=${ARM},SPOD_LOG_DETAIL=${SPOD_LOG_DETAIL:-1}" ;;
esac

# phi table convention: the rustc scenario uses the 'rustc' table (Qwen3-1.7B-Base RM, baked into
# spod_reward.py).
ENV_VARS="${ENV_VARS},SPOD_PHI_SRC=${SPOD_PHI_SRC:-rustc}"

echo "[GRPO] arm=${ARM}  config=${CONFIG_NAME}.yaml"
echo "[GRPO] pretrain=${PRETRAIN}  reward_pretrain=${REWARD_PRETRAIN}"
echo "[GRPO] env=${ENV_VARS}"

# export the comma-separated KEY=VAL entries one by one, for the reward worker / actor worker to read
IFS=',' read -ra _KVS <<< "${ENV_VARS}"
for kv in "${_KVS[@]}"; do
    export "${kv?}"
done

python entry/start_rlvr_pipeline.py \
    --config_path ../configs \
    --config_name "${CONFIG_NAME}" \
    --pretrain "${PRETRAIN}" \
    --reward_pretrain "${REWARD_PRETRAIN}" \
    --num_gpus_per_node "${SPOD_NUM_GPUS_PER_NODE:-${NGPU:-2}}" \
    --output_dir "${SPOD_GRPO_OUTPUT_DIR:-./output/grpo/${ARM}}" \
    --seed "${SPOD_SEED:-42}"
