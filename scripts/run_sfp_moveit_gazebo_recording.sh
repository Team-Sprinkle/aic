#!/usr/bin/env bash
# One bounded, nonfinal Gazebo replay of a saved SFP MoveIt path after actor prefix.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
plan=$(realpath "${1:?saved case plan JSON}")
output=$(realpath -m "${2:?new Gazebo output directory}")
gpu=${3:-3}
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'Invalid GPU' >&2; exit 2; }
variant=${AIC_SFP_MOVEIT_VARIANT:-retreat_align_approach}
[[ "$variant" == direct || "$variant" == align_then_approach || "$variant" == retreat_align_approach || "$variant" == side_down_then_center || "$variant" == side_sweep_high ]] || {
 echo "Invalid route variant: $variant" >&2; exit 2;
}
test ! -e "$output" || { echo "Preserve existing output: $output" >&2; exit 2; }
readarray -t details < <(.pixi/envs/default/bin/python - "$plan" "$variant" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text());a=Path(p['audit'])
assert a.joinpath('eval_config.yaml').exists()
assert hashlib.sha256(a.joinpath('eval_config.yaml').read_bytes()).hexdigest()==p['config_sha256']
assert p['route_variants'][sys.argv[2]]['success']
assert p['scene_group']==json.loads(a.joinpath('scored_stream.json').read_text())['scene_group_id']
print(a)
print(f"sfp_moveit_{p['scene_group'][:10]}")
PY
)
scene=${details[0]};name=${details[1]}
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
export AIC_TRANSPORT_EVAL_GPU=$gpu AIC_TRANSPORT_EVAL_POLICY_GPU=$gpu
export AIC_TRANSPORT_EVAL_SCENE_DIR=$scene
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.RunTemporalThenMoveItDiagnostic
export AIC_TRANSPORT_EVAL_TEACHER_PREFIX=true
export AIC_TEMPORAL_TRANSPORT_CHECKPOINT=/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt
export AIC_TEMPORAL_TRANSPORT_IMAGENET=/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
export AIC_SFP_CORRECTIVE_ACTOR_DURATION_S=5
export AIC_SFP_CORRECTIVE_TEACHER_DURATION_S=30
export AIC_SFP_MOVEIT_PLAN_SOURCE=$plan
export AIC_SFP_MOVEIT_VARIANT=$variant
export AIC_SFP_MOVEIT_LIVE_REPLAN=1
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06
export AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=40
export AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC=240
export AIC_SFP_DIAGNOSTIC_STOP=0
base=$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
bash scripts/run_clean_transport_eval.sh "$name" "$base" "$output" > "$output.launcher.log" 2>&1 &
launcher=$!
ready=0
for _ in $(seq 1 180); do
 if [[ -f "$output/moveit_plan_request.json" ]]; then ready=1; break; fi
 if ! kill -0 "$launcher" 2>/dev/null; then break; fi
 sleep 1
done
if (( ready )); then
 set +u
 source .pixi/envs/default/setup.bash
 set -u
 export ROS_DOMAIN_ID=95 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
 export PYTHONPATH="$repo/aic_teacher_official:$repo/scripts:${PYTHONPATH:-}"
 side_flag=()
 if [[ "$variant" == side_down_then_center || "$variant" == side_sweep_high ]]; then
   side_flag=(--enable-cable-side-variants)
 fi
 if ! .pixi/envs/default/bin/python scripts/plan_sfp_actor_state_corridor.py \
     --audit "$scene" --start-json "$output/moveit_plan_request.json" \
     --output "$output/live_moveit_plan.json" --buffer-mm 85 "${side_flag[@]}" \
     > "$output/live_moveit_planner.log" 2>&1; then
   echo failed > "$output/live_moveit_plan.error"
 fi
else
 echo 'Gazebo policy did not request a live plan' >&2
fi
wait "$launcher"
