#!/usr/bin/env bash
# Admit the completed bounded collection, cache images, and fit a matched pair.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
gpu_cache=${1:-1}
gpu_control=${2:-1}
gpu_diverse=${3:-2}
[[ "$gpu_cache" =~ ^[0-7]$ && "$gpu_control" =~ ^[0-7]$ && "$gpu_diverse" =~ ^[0-7]$ ]] || exit 2
[[ "$gpu_control" != "$gpu_diverse" ]] || { echo 'Matched fits need distinct GPUs' >&2; exit 2; }
root=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005
admitted=$root/sfp_diverse_final_v1
cache=$root/temporal_cache_diverse_v1
pause=$root/pause_cache_diverse_v1
out=outputs/experiments/2026-10-06_sfp_diverse_transport
base=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
warm=outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v2/best.pt
[[ -f "$admitted/manifest.json" && -f "$admitted/sfp_labels.jsonl" ]] || {
  echo 'Complete causal admission is required before training' >&2; exit 2;
}
python3 - /home/nas/chmin/aic_archives/2026-10-06_sfp_diverse_teacher_v1 \
  /home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_retry_v1 <<'PY'
import json,sys
from pathlib import Path
for arg in sys.argv[1:]:
    p=Path(arg);plan=json.loads((p/'plan.json').read_text())
    progress=json.loads((p/'progress.json').read_text())['results']
    if set(progress)!={r['name'] for r in plan['scenes']}:
        raise SystemExit(f'Collection incomplete: {p}')
    if len(plan['scenes'])>=20:
        pilot=plan['scenes'][:20]
        if sum(progress[x['name']]['official_score'].get('tier3',0)>=75 for x in pilot)<16:
            raise SystemExit('Pilot score gate failed')
    for x in plan['scenes']:
        if progress[x['name']]['official_score'].get('tier3',0)>=75 and not \
           (Path(progress[x['name']]['audit'])/'scored_stream.json').exists():
            raise SystemExit('Scored bag extraction incomplete')
PY
[[ ! -e "$cache" && ! -e "$pause" &&
   ! -e "$out/control_matched_v1" && ! -e "$out/diverse_matched_v1" ]] || {
  echo 'Matched output already exists; preserve and resume manually' >&2; exit 2;
}
bash scripts/cache_sfp_balanced_temporal_features.sh "$admitted/sfp_labels.jsonl" "$cache" "$gpu_cache"
.pixi/envs/default/bin/python scripts/prepare_temporal_route_handoff_cache.py \
  --cache "$cache" --sfp-labels "$admitted/sfp_labels.jsonl" --output "$pause" \
  --sfp-axial-min-m -.13 --sfp-axial-max-m -.06 --sfp-lateral-max-m .03 \
  > "$out/pause_cache_diverse_v1.log" 2>&1
common=(--cache "$pause" --checkpoint "$base" --full-state-skip --no-route-heads
        --long-history --task-family both --balance-sfp-targets --paired-sfp-mass .5
        --init-temporal "$warm" --updates 3000 --min-updates 3000 --budget-minutes 180)
CUDA_VISIBLE_DEVICES="$gpu_control" .pixi/envs/default/bin/python \
  scripts/train_temporal_transport_probe.py "${common[@]}" \
  --exclude-train-episode-min 2400000 --output "$out/control_matched_v1" \
  > "$out/control_matched_v1.log" 2>&1 &
pid_control=$!
CUDA_VISIBLE_DEVICES="$gpu_diverse" .pixi/envs/default/bin/python \
  scripts/train_temporal_transport_probe.py "${common[@]}" \
  --output "$out/diverse_matched_v1" \
  > "$out/diverse_matched_v1.log" 2>&1 &
pid_diverse=$!
wait "$pid_control"
wait "$pid_diverse"
echo "matched_fit_done control=$out/control_matched_v1 diverse=$out/diverse_matched_v1"
