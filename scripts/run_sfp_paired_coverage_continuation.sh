#!/usr/bin/env bash
# Train-only paired-target admission, frozen-camera cache, and bounded actor fit.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
data_root=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005
old=outputs/experiments/2026-10-05_transport_identity_audit/paired_merged_train_v3/sfp_labels.jsonl
wide=/home/nas/chmin/aic_archives/2026-10-05_sfp_paired_coverage_v2
sparse=/home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_alternates_v1
merged=$data_root/paired_coverage_merged_v1
cache=$data_root/temporal_cache_v3
route=$data_root/route_handoff_cache_v2
actor=outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v1
handoff=outputs/experiments/2026-10-05_transport_identity_audit/sfp_frozen_handoff_v2
base=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
warm=outputs/experiments/2026-10-05_transport_identity_audit/route_handoff_action_only_v1/best.pt
cal=configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json
gpu=${1:-1}
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU index must be 0..7' >&2; exit 2; }
[[ ! -e "$merged" && ! -e "$cache" && ! -e "$route" && ! -e "$actor" && ! -e "$handoff" ]] || {
  echo 'A continuation output already exists; preserve it and resume manually' >&2; exit 2;
}
for folder in "$wide" "$sparse"; do
  [[ -f "$folder/progress.json" ]] || { echo "Missing completed collection $folder" >&2; exit 2; }
  .pixi/envs/default/bin/python - "$folder" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);planned=json.loads((p/'plan.json').read_text())['scenes']
done=json.loads((p/'progress.json').read_text())['results']
if set(done)!={r['name'] for r in planned}:raise SystemExit(f'Incomplete collection: {p}')
PY
done
.pixi/envs/default/bin/python scripts/admit_sfp_paired_coverage.py \
  --old "$old" --collection "$wide" --collection "$sparse" \
  --calibration "$cal" --output "$merged" \
  > "$data_root/paired_coverage_admission_v1.log" 2>&1
bash scripts/cache_sfp_balanced_temporal_features.sh "$merged/sfp_labels.jsonl" "$cache" "$gpu"
.pixi/envs/default/bin/python scripts/prepare_temporal_route_handoff_cache.py \
  --cache "$cache" --sfp-labels "$merged/sfp_labels.jsonl" --output "$route" \
  > "$data_root/route_handoff_cache_v2.log" 2>&1
CUDA_VISIBLE_DEVICES="$gpu" .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache "$route" --checkpoint "$base" --output "$actor" \
  --full-state-skip --no-route-heads --long-history --task-family both \
  --balance-sfp-targets --paired-sfp-mass .5 --init-temporal "$warm" \
  --updates 3000 --min-updates 1000 --budget-minutes 60 \
  > "$actor.log" 2>&1
# This matched head isolates the effect of extra SFP arrival labels. It never
# changes the transport actor's action weights and remains shadow-only.
CUDA_VISIBLE_DEVICES="$gpu" .pixi/envs/default/bin/python scripts/train_sfp_frozen_handoff.py \
  --cache "$route" --base "$base" --actor "$warm" \
  --prior "$route/handoff_prior.json" --output "$handoff" --updates 800 \
  > "$handoff.log" 2>&1
printf '%s\n' "merged=$merged" "cache=$cache" "route=$route" "actor=$actor" "handoff=$handoff"
