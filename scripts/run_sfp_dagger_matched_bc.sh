#!/usr/bin/env bash
# Equal-budget SFP-only native BC continuation, with/without causal teacher takeovers.
set -euo pipefail
cd "$(dirname "$0")/.."
archive=/home/nas/chmin/aic_archives/2026-10-02_two_family_perception
tag=${AIC_SFP_DAGGER_TAG:-v1}
base=$archive/sfp_native_pair_relabel_matched_v2
corrections=$archive/sfp_dagger_pair_bc_$tag
checkpoint=$archive/sfp_actor_pair_matched_v1/best.pt
control=$archive/sfp_actor_dagger_control_$tag
corrected=$archive/sfp_actor_dagger_corrected_$tag
test -s "$corrections/rows.jsonl"
test -f "$corrections/summary.json"
test ! -e "$corrected"
if [[ ${AIC_SFP_DAGGER_REUSE_V1_CONTROL:-0} != 1 ]]; then
  test ! -e "$control"
fi
common=(--dataset "$base" --sfp-native-only
  --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3
  --imagenet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
  --resume-local "$checkpoint" --selection-metric sfp_native_near
  --updates 1000 --minutes 60 --batch-size 24 --workers 4
  --learning-rate .00003 --orientation-loss-weight .1
  --near-lateral-loss-weight .8 --near-phase-sample-weight 3
  --near-phase-threshold-m .08 --lateral-buffer-m .08)
if [[ ${AIC_SFP_DAGGER_REUSE_V1_CONTROL:-0} == 1 ]]; then
  test -f "$archive/sfp_actor_dagger_control_v1/summary.json"
  control_pid=
else
  CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_predicted_port_local_actor.py \
    "${common[@]}" --output "$control" \
    > "$archive/sfp_actor_dagger_control_$tag.log" 2>&1 &
  control_pid=$!
fi
CUDA_VISIBLE_DEVICES=2 .pixi/envs/default/bin/python scripts/train_predicted_port_local_actor.py \
  "${common[@]}" --extra-rows "$corrections" --onpolicy-boost 4 \
  --output "$corrected" \
  > "$archive/sfp_actor_dagger_corrected_$tag.log" 2>&1 &
corrected_pid=$!
control_rc=0;corrected_rc=0
if [[ -n $control_pid ]]; then wait "$control_pid" || control_rc=$?; fi
wait "$corrected_pid" || corrected_rc=$?
printf 'control_exit=%s corrected_exit=%s\n' "$control_rc" "$corrected_rc"
test "$control_rc" -eq 0 && test "$corrected_rc" -eq 0
