#!/usr/bin/env bash
# Matched CPU fallback: same frozen init, data, seed, and update count;
# the only experimental difference is excluding audited sustained stalls.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
out=$(realpath -m "${1:?new result directory}")
[[ ! -e "$out" ]] || { echo "Preserve existing result: $out" >&2; exit 2; }
mkdir -p "$out"
cache=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/route_handoff_divergent_round2_v1
pairs=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/paired_action_manifest_round2_v1.json
prior=/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/control/best.pt
filter=/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/stall_filter_v1.json
base=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
common=(--device cpu --cache "$cache" --checkpoint "$base"
        --full-state-skip --long-history --no-route-heads
        --balance-sfp-targets --paired-sfp-mass .5
        --init-temporal "$prior" --frozen-reference "$prior"
        --reference-old-sfp-weight 1 --paired-contrast-manifest "$pairs"
        --paired-contrast-weight .2 --updates 2000 --min-updates 2000
        --budget-minutes 90 --seed 20261007)
for variant in baseline stall_filtered; do
  extra=()
  [[ "$variant" == baseline ]] || extra=(--excluded-train-indices "$filter")
  printf '%q ' .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
    "${common[@]}" "${extra[@]}" --output "$out/$variant" > "$out/$variant.command.txt"
  printf '\n' >> "$out/$variant.command.txt"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
    "${common[@]}" "${extra[@]}" --output "$out/$variant" \
    > "$out/$variant.log" 2>&1
done
