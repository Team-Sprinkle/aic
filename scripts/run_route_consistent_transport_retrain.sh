#!/usr/bin/env bash
set -euo pipefail
cd /data1/chmin/yj/ws_aic/src/aic

# Active, route-checked dataset: SC contains only Low Lateral Bypass teachers.
export CUDA_VISIBLE_DEVICES="${AIC_TRAIN_GPU:-4}"
export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
union=/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4
cache="${AIC_ROUTE_CACHE:-/home/nas/chmin/aic_archives/2026-10-12_route_consistent_cache_v1}"
output="${AIC_ROUTE_OUTPUT:-$PWD/outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1}"
sc_spec="$PWD/outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json"
checkpoint="$PWD/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
init="$PWD/outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/fit_pair_ft/best.pt"

.pixi/envs/default/bin/python - "$union" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads((root/'manifest.json').read_text())
for family in ('sfp','sc'):
    p=root/f'{family}_labels.jsonl'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==m['families'][family]['output_sha256']
assert m['families']['sc']['route_policy'].startswith('llb_only')
assert m['complete_reset_split'] and not m['final_scenes_opened']
PY

if [[ ! -d "$cache" ]]; then
 .pixi/envs/default/bin/python -u scripts/cache_temporal_transport_features.py \
  --sfp-labels "$union/sfp_labels.jsonl" --sc-labels "$union/sc_labels.jsonl" \
  --sc-spec "$sc_spec" \
  --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
  --checkpoint "$checkpoint" --output "$cache" --feature-source imagenet_views \
  --imagenet-resnet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth --batch-size 48
fi
.pixi/envs/default/bin/python - "$union" "$cache" <<'PY'
import hashlib,json,sys
from pathlib import Path
union,cache=map(Path,sys.argv[1:])
manifest=json.loads((cache/'manifest.json').read_text())
for family in ('sfp','sc'):
    path=union/f'{family}_labels.jsonl'
    assert manifest['sources_sha256'][str(path)]==hashlib.sha256(path.read_bytes()).hexdigest()
assert manifest['feature_source']=='imagenet_views'
PY
if [[ ! -f "$cache/camera_offsets.npy" ]]; then
 .pixi/envs/default/bin/python scripts/make_temporal_camera_offsets.py \
  --cache "$cache" --sc-rows "$union/sc_labels.jsonl"
fi

test ! -e "$output" || { echo "Preserve existing training output: $output" >&2; exit 1; }
.pixi/envs/default/bin/python -u scripts/train_route_aware_diffusion_transport.py \
 --cache "$cache" --checkpoint "$checkpoint" --output "$output" --init "$init" \
 --learning-rate .00003 --updates 6000 --budget-minutes 150 \
 --batch-size 48 --sample-steps 12 --seed 20261014
