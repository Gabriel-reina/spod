#!/bin/bash
# run_rm_phi (rustc-perf): re-score the library items with the trained RM to produce the per-category
#   mean-score table under the SPOD rmscore convention (the raw material for phi).
#   This table is the source of the 'rustc' rm_mean block in roll_spod/utils/spod_reward.py: phi(c) is
#   the z-score of each category's mean RM score, which SPOD uses to penalize high-scoring items of
#   over-represented categories. Direction is the same family as speedrun -- the rarest
#   B6_infra_tooling has the highest mean (phi=+1.722), the most common B1_backend_codegen is low
#   (phi=-0.587), confirming RM popularity bias (freq-rm_mean Pearson r=-0.475).
#   The open-source spod_reward.py already ships this table (SPOD_PHI_SRC=rustc works out of the box);
#   this step is for validation / re-measuring after swapping the RM or the base. 1 GPU, 1414 forward
#   passes, minute-scale.
#
# Data (local data/): rustc is fully hold-out clean, rm_items_prompted.jsonl is already 1414 train-only
#   items, no test-change item ever entered, so no _clean variant is needed and SPOD_DATA_TAG defaults
#   to empty.
#
# Usage: SPOD_RM=./output_spod_rm/rm_hf bash scripts/run_rm_phi.sh
set -e
cd "$(dirname "$0")/.."

RM="${SPOD_RM:-./output_spod_rm/rm_hf}"    # local dir of the RM checkpoint used for scoring
DATA_TAG="${SPOD_DATA_TAG:-}"              # rustc is naturally train-only, default empty
TAG="${SPOD_PHI_TAG:-rustc}"

echo "[PHI] rm=${RM}  data_tag='${DATA_TAG}'  tag=${TAG}"

python entry/start_spod_rm_phi.py \
    --rm "${RM}" \
    --tag "${TAG}" \
    --data_tag "${DATA_TAG}" \
    --out_dir ./output_spod_phi
