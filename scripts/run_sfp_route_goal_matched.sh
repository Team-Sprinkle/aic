#!/usr/bin/env bash
# Matched full-action comparison; both arms use the same frozen 3D route probe.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
gpu="${1:-7}"
output="${2:-/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_matched_v1}"
if [[ -e "$output" ]]; then
  echo "Preserve existing output: $output" >&2
  exit 2
fi
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES="$gpu"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
cache=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/route_handoff_divergent_round2_v1
base=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
initial=/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_matched_v2/control/best.pt
goal=outputs/experiments/2026-10-07_sfp_route_branch_support_v1/visual_pose_task_ridge_v2.npz

common=(.pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py
  --cache "$cache" --checkpoint "$base" --device cuda
  --full-state-skip --long-history --no-route-heads --balance-sfp-targets
  --paired-sfp-mass .5 --init-temporal "$initial"
  --route-goal-model "$goal" --frozen-reference "$initial"
  --reference-old-sfp-weight .5 --updates 2000 --min-updates 1000
  --budget-minutes 75 --seed 20261007)

for arm in control candidate; do
  command=("${common[@]}" --output "$output/$arm")
  if [[ "$arm" == control ]]; then
    command+=(--route-goal-zero-control)
  fi
  printf '%q ' "${command[@]}" > "$output/$arm.command.txt"
  printf '\n' >> "$output/$arm.command.txt"
  "${command[@]}" > "$output/$arm.log" 2>&1
  tail -n 2 "$output/$arm.log"
done
