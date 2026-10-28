#!/bin/bash
# Run on a Linux GPU host over SSH after installing the public ROLL environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENARIO="${1:-nanogpt}"
case "$SCENARIO" in
  nanogpt) cd "$ROOT" ;;
  rustc) cd "$ROOT/rustc_perf" ;;
  *) echo 'Usage: run_reproduction.sh nanogpt|rustc'; exit 2 ;;
esac
export SPOD_SEED=42 SPOD_SFT_SEED=42 SPOD_RM_SEED=42 PYTHONHASHSEED=42
export NGPU="${NGPU:-2}"
ray --version
bash scripts/run_sft.sh
bash scripts/run_rm.sh
bash scripts/run_rm_phi.sh
bash scripts/run_eval.sh sft
for ARM in v bhap; do
  bash scripts/run_grpo.sh "$ARM"
  bash scripts/run_convert.sh "$ARM" 49
  bash scripts/run_eval.sh "$ARM" 49
done
python spod_metrics.py --dir ./output/spod_eval --arms sft,v_step49,bhap_step49 --base v_step49
python spod_analyze.py --dir ./output/spod_eval --base v_step49
