#!/usr/bin/env bash
# Six bounded same-scene autonomous SC transport repeats. Preserve each run.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
root="$repo/outputs/experiments/2026-10-01_transport_reliability"
export AIC_TRANSPORT_EVAL_GPU=1 AIC_TRANSPORT_EVAL_POLICY_GPU=2
export AIC_TRANSPORT_EVAL_SCENE_DIR="$repo/outputs/experiments/2026-09-30_sc_llb_collection/pilot30/scenes/llb_0009"
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.RunTemporalTransport
export AIC_TEMPORAL_TRANSPORT_CHECKPOINT="$root/history_no_previous_6000/best.pt"
export AIC_TEMPORAL_TRANSPORT_IMAGENET=/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
export AIC_SC_VISUAL_STANDOFF_CHECKPOINT="$root/actor_native_dual_port_locator/best.pt"
export AIC_SC_VISUAL_STANDOFF_CALIBRATION="$repo/configs/hierarchical_recovery/sc_visual_standoff_20261001.json"
export AIC_SC_VISUAL_STANDOFF_STIFFNESS=1200 AIC_SC_VISUAL_STANDOFF_DAMPING=80
export AIC_SC_VISUAL_STANDOFF_ORIENTATION=1
export AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT="$root/onpolicy_goal_adaptation/best.pt"
export AIC_SC_VISUAL_STANDOFF_TRIGGER_M=.08
checkpoint="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
plan="$repo/outputs/experiments/2026-09-30_sc_llb_collection/pilot30/plans/llb_0009_insert/moveit_joint_trajectory.json"
for specification in control_1:0 guard_1:.18 guard_2:.18 control_2:0 control_3:0 guard_3:.18; do
  name=${specification%%:*};export AIC_SC_TRANSPORT_Z_FLOOR_M=${specification#*:}
  audit="$root/matched_zfloor_${name}"
  if [[ -e "$audit" ]]; then
    echo "Preserve existing run: $audit" >&2
    exit 1
  fi
  echo "BEGIN $name $(date -u +%FT%TZ)" >&2
  bash "$repo/scripts/run_clean_transport_eval.sh" llb_0009 "$checkpoint" "$audit"
  "$repo/.pixi/envs/default/bin/python" "$repo/scripts/measure_transport_teacher_path.py" "$audit" --plan "$plan" >/dev/null
  echo "END $name $(date -u +%FT%TZ)" >&2
done
