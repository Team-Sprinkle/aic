#!/usr/bin/env bash
set -euo pipefail
# Historical reproducer only. The October 11 union mixed CheatCode SC routes
# with Low Lateral Bypass SC routes and is retired from active training.
if [[ "${AIC_REPRODUCE_RETIRED_MIXED_SC:-0}" != 1 ]]; then
 echo "Retired mixed-SC training recipe. Use the route-consistent v2 union; set AIC_REPRODUCE_RETIRED_MIXED_SC=1 only to reproduce the frozen experiment." >&2
 exit 2
fi
cd /data1/chmin/yj/ws_aic/src/aic

export CUDA_VISIBLE_DEVICES=4
export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"

general=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_coverage_v1
exact=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_sc_exact_v1
labels_general=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_labels_v1
labels_exact=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_sc_exact_labels_v1
union=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_union_v1
cache=/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_cache_v1
output="$PWD/outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/official_anchor_retrain"
base_sfp=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl
base_sc="$PWD/outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/transport_chunks.jsonl"
sc_spec="$PWD/outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json"
checkpoint="$PWD/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
init="$PWD/outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/fit_pair_ft/best.pt"

if [[ ! -d "$labels_general" ]]; then
 .pixi/envs/default/bin/python -u scripts/prepare_official_anchor_route_labels.py \
  --collection "$general" --output "$labels_general" --episode-index-start 4000000
fi
if [[ ! -d "$labels_exact" ]]; then
 .pixi/envs/default/bin/python -u scripts/prepare_official_anchor_route_labels.py \
  --collection "$exact" --output "$labels_exact" --episode-index-start 4100000
fi

.pixi/envs/default/bin/python - "$labels_general" "$labels_exact" <<'PY'
import json,sys
from collections import Counter
counts=Counter()
for folder in sys.argv[1:]:
    manifest=json.load(open(folder+'/manifest.json'))
    for row in manifest['admitted']:
        counts[(row['family'],row['split'])]+=1
minimum={('sfp','train'):12,('sfp','validation'):3,
         ('sc','train'):12,('sc','validation'):3}
print({'admitted':dict(counts),'minimum':minimum})
if any(counts[key]<value for key,value in minimum.items()):
    raise SystemExit('Coverage gate failed: inspect scenes and causal labels before fitting')
PY

if [[ ! -d "$union" ]]; then
 .pixi/envs/default/bin/python scripts/merge_official_anchor_route_labels.py \
  --base-sfp "$base_sfp" --base-sc "$base_sc" \
  --anchor-labels "$labels_general" --anchor-labels "$labels_exact" --output "$union"
fi
if [[ ! -d "$cache" ]]; then
 .pixi/envs/default/bin/python -u scripts/cache_temporal_transport_features.py \
  --sfp-labels "$union/sfp_labels.jsonl" --sc-labels "$union/sc_labels.jsonl" \
  --sc-spec "$sc_spec" \
  --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
  --checkpoint "$checkpoint" --output "$cache" --feature-source imagenet_views \
  --imagenet-resnet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth --batch-size 48
fi
if [[ ! -f "$cache/camera_offsets.npy" ]]; then
 .pixi/envs/default/bin/python scripts/make_temporal_camera_offsets.py \
  --cache "$cache" --sc-rows "$union/sc_labels.jsonl"
fi

test ! -e "$output" || { echo "Preserve existing training output: $output" >&2; exit 1; }
.pixi/envs/default/bin/python -u scripts/train_route_aware_diffusion_transport.py \
 --cache "$cache" --checkpoint "$checkpoint" --output "$output" --init "$init" \
 --learning-rate .00003 --anchor-mass .25 --updates 6000 --budget-minutes 150 \
 --batch-size 48 --sample-steps 12 --seed 20261014
