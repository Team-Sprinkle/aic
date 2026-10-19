#!/usr/bin/env bash
# Matched, bounded native-SFP actor continuation. Invoke only after relabel v2 completes.
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/experiments/2026-10-02_two_family_perception
data=/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_pair_relabel_matched_v2
archive=/home/nas/chmin/aic_archives/2026-10-02_two_family_perception
test -f "$data/summary.json"
test -f "$data/rows.jsonl"
test -f "$data/matched_old_rows.jsonl"
test ! -e "$archive/sfp_actor_pair_matched_v1"
test ! -e "$archive/sfp_actor_oldframe_matched_v1"
mkdir -p "$archive/sfp_native_old_matched_v1"
ln -sfn "$data/matched_old_rows.jsonl" "$archive/sfp_native_old_matched_v1/rows.jsonl"
common=(--sfp-native-only
  --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3
  --imagenet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
  --resume-local "$root/local_actor_sfp_native39_bodyconverted_v1/best.pt"
  --selection-metric sfp_native_near
  --updates 600 --minutes 60 --batch-size 24 --workers 4
  --learning-rate 0.00003 --orientation-loss-weight 0.1
  --near-lateral-loss-weight 0.8 --near-phase-sample-weight 3
  --near-phase-threshold-m 0.08 --lateral-buffer-m 0.08)
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_predicted_port_local_actor.py \
  --dataset "$archive/sfp_native_old_matched_v1" \
  --output "$archive/sfp_actor_oldframe_matched_v1" "${common[@]}" \
  > "$archive/sfp_actor_oldframe_matched_v1.log" 2>&1 &
old_pid=$!
CUDA_VISIBLE_DEVICES=2 .pixi/envs/default/bin/python scripts/train_predicted_port_local_actor.py \
  --dataset "$data" \
  --output "$archive/sfp_actor_pair_matched_v1" "${common[@]}" \
  > "$archive/sfp_actor_pair_matched_v1.log" 2>&1 &
pair_pid=$!
old_rc=0;pair_rc=0
wait "$old_pid" || old_rc=$?
wait "$pair_pid" || pair_rc=$?
printf 'oldframe_exit=%s pair_exit=%s\n' "$old_rc" "$pair_rc"
test "$old_rc" -eq 0 && test "$pair_rc" -eq 0
