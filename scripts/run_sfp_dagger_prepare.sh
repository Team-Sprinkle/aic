#!/usr/bin/env bash
# Convert only full-score post-takeover commands into paired-RGB actor targets.
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/experiments/2026-10-02_two_family_perception
archive=/home/nas/chmin/aic_archives/2026-10-02_two_family_perception
tag=${AIC_SFP_DAGGER_TAG:-v1}
collection=${AIC_SFP_DAGGER_COLLECTION:-$archive/sfp_onpolicy_takeover20_fivecard_v1}
selection=$root/sfp_dagger_selection_$tag
projection=$archive/sfp_dagger_projection_$tag
chunks=$archive/sfp_dagger_teacher_chunks_$tag.jsonl
dataset=$archive/sfp_dagger_pair_bc_$tag
min_episodes=${AIC_SFP_DAGGER_MIN_EPISODES:-6}
min_rows=${AIC_SFP_DAGGER_MIN_ROWS:-300}
min_early=${AIC_SFP_DAGGER_MIN_EARLY_ROWS:-50}
test -f "$collection/summary.json"
test ! -e "$selection"
test ! -e "$projection"
test ! -e "$chunks"
test ! -e "$dataset"
.pixi/envs/default/bin/python scripts/prepare_sfp_stage_dagger_selection.py \
  --collection "$collection" --output "$selection" --min-episodes "$min_episodes"
.pixi/envs/default/bin/python scripts/build_sfp_native_projection_labels.py \
  --selection "$selection" \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output "$projection" > "$archive/sfp_dagger_projection_$tag.log" 2>&1
.pixi/envs/default/bin/python scripts/build_sfp_native_teacher_local_chunks.py \
  --labels "$projection" --post-teacher-takeover-only \
  --output "$chunks" --summary "$archive/sfp_dagger_teacher_chunks_${tag}_summary.json" \
  > "$archive/sfp_dagger_teacher_chunks_$tag.log" 2>&1
CUDA_VISIBLE_DEVICES=2 .pixi/envs/default/bin/python scripts/build_sfp_native_cascade_bc_dataset.py \
  --source "$chunks" --labels "$projection" \
  --coarse "$root/joint_locator_native_sfp_v5_heatmap/best.pt" \
  --broad "$root/sfp_broad_crop480_v2/best.pt" \
  --fine "$root/sfp_fine_crop160_v2_inside_selection/best.pt" \
  --orientation "$root/sfp_native_orientation_v2_highnative/best.pt" \
  --orientation-secondary "$root/sfp_native_orientation_v3_runtime_matched/best.pt" \
  --pair-orientation \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --selection-manifest "$root/sfp_dagger_alltrain_split.json" \
  --sampling-boost 8 --max-supervision-position-error-mm 3 \
  --output "$dataset" > "$archive/sfp_dagger_pair_bc_$tag.log" 2>&1
AIC_SFP_DAGGER_TAG="$tag" AIC_SFP_DAGGER_MIN_EPISODES="$min_episodes" \
AIC_SFP_DAGGER_MIN_ROWS="$min_rows" AIC_SFP_DAGGER_MIN_EARLY_ROWS="$min_early" \
.pixi/envs/default/bin/python - <<'PY'
import json
import os
from pathlib import Path
root=Path('/home/nas/chmin/aic_archives/2026-10-02_two_family_perception')
dataset=root/('sfp_dagger_pair_bc_'+os.environ['AIC_SFP_DAGGER_TAG'])
report=json.loads((dataset/'summary.json').read_text())
rows=[json.loads(line) for line in (dataset/'rows.jsonl').open()]
episodes={r['episode'] for r in rows}
onpolicy=sum(r.get('onpolicy_teacher_match',False) for r in rows)
if (len(episodes)<int(os.environ['AIC_SFP_DAGGER_MIN_EPISODES']) or
    len(rows)<int(os.environ['AIC_SFP_DAGGER_MIN_ROWS']) or
    onpolicy<int(os.environ['AIC_SFP_DAGGER_MIN_EARLY_ROWS'])):
    raise ValueError(f'Corrective dataset gate: {len(episodes)} episodes, {len(rows)} rows, {onpolicy} near-takeover rows')
print(json.dumps({'admitted_episodes':len(episodes),'rows':len(rows),
                  'near_takeover_rows':onpolicy,'pose_error':report['accepted_lateral_error_mm']},indent=2))
PY
