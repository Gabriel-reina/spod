#!/bin/bash
# run_rm_phi: use the trained RM to re-score the library items, producing the per-category mean
#   table in the SPOD rmscore convention (the raw material for phi).
#   This table is the source of the spod2clean _RM_MEAN in roll_spod/utils/spod_reward.py: phi(c) is
#   derived from "the z-score of each category's RM mean", which SPOD uses to deduct points, by category,
#   from the high-scoring items of popular classes.
#   The open-source spod_reward.py already bakes in this table; this step is for verification / re-measuring
#   after swapping the RM or the base model.
#   1 GPU, forward over 148 items, minute-scale.
#
# Usage: SPOD_RM=./output_spod_rm/rm_hf bash scripts/run_rm_phi.sh
set -e
cd "$(dirname "$0")/.."

RM="${SPOD_RM:-./output_spod_rm/rm_hf}"    # local dir of the RM checkpoint used for scoring
DATA_TAG="${SPOD_DATA_TAG:-_clean}"        # _clean = the 148 hold-out items
TAG="${SPOD_PHI_TAG:-spod2clean}"

echo "[PHI] rm=${RM}  data_tag='${DATA_TAG}'  tag=${TAG}"

python entry/start_spod_rm_phi.py \
    --rm "${RM}" \
    --tag "${TAG}" \
    --data_tag "${DATA_TAG}" \
    --out_dir ./output_spod_phi
