#!/usr/bin/env bash
# Rootless official-Gazebo replay of an actual MoveIt joint trajectory.
set -euo pipefail
lock_gpu=${AIC_MOVEIT_PILOT_GPU:-3}
[[ "$lock_gpu" =~ ^[0-7]$ ]] || exit 2
case ",${AIC_PROJECT_GPU_LOCKS_HELD:-}," in
 *",$lock_gpu,"*) ;;
 *) export AIC_PROJECT_GPU_LOCKS_HELD="${AIC_PROJECT_GPU_LOCKS_HELD:+$AIC_PROJECT_GPU_LOCKS_HELD,}$lock_gpu"
    exec flock -w 3600 "/var/tmp/chmin_aic_gpu${lock_gpu}.lock" bash "$0" "$@" ;;
esac

repo="${AIC_MOVEIT_PILOT_REPO:-$(realpath "$(dirname "$0")/..")}"
if [[ "${AIC_MOVEIT_PILOT_LAUNCH_SNAPSHOT:-0}" != 1 ]]; then
  export AIC_MOVEIT_PILOT_REPO="$repo" AIC_MOVEIT_PILOT_LAUNCH_SNAPSHOT=1
  mkdir -p "$repo/outputs/experiments/2026-09-30_transport_continuation/launcher_snapshots"
  snapshot=$(mktemp "$repo/outputs/experiments/2026-09-30_transport_continuation/launcher_snapshots/teacher_XXXXXX.sh")
  cp "$0" "$snapshot"
  exec bash "$snapshot" "$@"
fi
plan_dir="$(realpath "${1:?MoveIt plan directory}")"
audit_dir="$(realpath -m "${2:?new audit output directory}")"
scene_dir="$(realpath "${AIC_MOVEIT_PILOT_SCENE_DIR:-/var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926}")"
gpu="${AIC_MOVEIT_PILOT_GPU:-3}"
online_insertion="${AIC_MOVEIT_PILOT_ONLINE_INSERTION:-false}"
replay_action_mode="${AIC_MOVEIT_PILOT_ACTION_MODE:-joint_position_then_cheatcode}"
case "$replay_action_mode" in
 joint_position_then_cheatcode|absolute_cartesian_pose_base_link) ;;
 *) echo "Unsupported explicit replay interface: $replay_action_mode" >&2; exit 1 ;;
esac
tracking_gate_force_fraction="${AIC_MOVEIT_PILOT_TRACKING_GATE_FORCE_FRACTION:-1.0}"
force_confirm_threshold="${AIC_MOVEIT_PILOT_FORCE_CONFIRM_THRESHOLD_N:-1.0}"
sc_contact_recovery="${AIC_MOVEIT_PILOT_SC_CONTACT_RECOVERY:-false}"
expert_mode="${AIC_MOVEIT_PILOT_EXPERT_MODE:-nominal}"
recovery_initial_lateral_gate="${AIC_MOVEIT_PILOT_RECOVERY_INITIAL_LATERAL_GATE_M:-0.005}"
nominal_initial_lateral_gate="${AIC_MOVEIT_PILOT_NOMINAL_INITIAL_LATERAL_GATE_M:-}"
guarded_lateral_servo="${AIC_MOVEIT_PILOT_GUARDED_LATERAL_SERVO:-false}"
sc_preinsert_servo_max_bias="${AIC_MOVEIT_PILOT_SC_PREINSERT_SERVO_MAX_BIAS_M:-0.012}"
precontact_port_align_sec="${AIC_MOVEIT_PILOT_PRECONTACT_PORT_ALIGN_SEC:-0.4}"
sc_tracking_gate_speed="${AIC_MOVEIT_PILOT_SC_TRACKING_GATE_SPEED_MPS:-0.006}"
precontact_fatal_force="${AIC_MOVEIT_PILOT_PRECONTACT_FATAL_FORCE_N:-1.0}"
precontact_soft_force="${AIC_MOVEIT_PILOT_PRECONTACT_SOFT_FORCE_N:-0.6}"
sc_z_gate_max_hold="${AIC_MOVEIT_PILOT_SC_Z_GATE_MAX_HOLD_SEC:-0.45}"
low_force_insert_max_z="${AIC_MOVEIT_PILOT_LOW_FORCE_INSERT_MAX_Z_OFFSET_M:-0.0}"
sc_low_force_insert_limit="${AIC_MOVEIT_PILOT_SC_LOW_FORCE_INSERT_LIMIT_N:-6.0}"
sc_recovery_max_backoff="${AIC_MOVEIT_PILOT_SC_RECOVERY_MAX_BACKOFF_M:-0.020}"
sc_recovery_backoff_sec="${AIC_MOVEIT_PILOT_SC_RECOVERY_BACKOFF_SEC:-0.80}"
sc_recovery_required_measured="${AIC_MOVEIT_PILOT_SC_RECOVERY_REQUIRED_MEASURED_M:-0.008}"
oracle_contacts="${AIC_MOVEIT_PILOT_ORACLE_CONTACTS:-false}"
visibility_labels="${AIC_MOVEIT_PILOT_VISIBILITY_LABELS:-false}"
local_actor_checkpoint="${AIC_MOVEIT_PILOT_LOCAL_ACTOR_CHECKPOINT:-}"
name="aic_moveit_sc_pilot_$(basename "$audit_dir")"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/1008/docker.sock}"

test ! -e "$audit_dir/results/scoring.yaml" || { echo "Already scored: $audit_dir" >&2; exit 1; }
"$repo/.pixi/envs/default/bin/python" - "$plan_dir" "$scene_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
d=Path(sys.argv[1]); scene=Path(sys.argv[2]); p=json.loads((d/'plan_summary.json').read_text()); t=json.loads((d/'moveit_joint_trajectory.json').read_text())
assert p['config_sha256']==hashlib.sha256((scene/'eval_config.yaml').read_bytes()).hexdigest()
assert p['result']['success'] and p['result']['metadata']['backend']=='moveit_py'
assert p['result']['metadata']['collision_objects_applied'] >= 100
assert p['result']['metadata']['recorded_start_used']
assert len(t['waypoints']) >= 2 and all(w.get('joint_positions') for w in t['waypoints'])
assert all(w['diagnostics']['planner']=='moveit_py' for w in t['waypoints'])
PY
mkdir -p "$audit_dir"
cp "$0" "$audit_dir/launcher_snapshot.sh"
mkdir -p "$audit_dir/teacher_sources"
cp "$repo/aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py" \
   "$repo/aic_teacher_official/aic_teacher_official/WideApproachTeacherReplay.py" \
   "$repo/aic_teacher_official/aic_teacher_official/wide_approach_geometry.py" \
   "$audit_dir/teacher_sources/"
sha256sum "$audit_dir"/teacher_sources/*.py > "$audit_dir/teacher_sources_sha256.txt"
"$repo/.pixi/envs/default/bin/python" - "$audit_dir" "$replay_action_mode" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],'teacher_interface_manifest.json').write_text(json.dumps(dict(
 transport_action_mode=sys.argv[2],teacher=True,actor=False,
 privileged_geometry_used_by_teacher=True,autonomous_actor_geometry_input=False,
 supervisor_reset_commands_excluded_from_BC=True,
 cartesian_stiffness=[90,90,90,50,50,50],cartesian_damping=[50,50,50,20,20,20]),indent=2)+'\n')
PY
cp "$scene_dir"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} "$audit_dir/"
if [[ -f "$scene_dir/ur_llb_start.xacro" ]]; then
  cp "$scene_dir/ur_llb_start.xacro" "$audit_dir/ur_llb_start.xacro"
fi
cp "$repo/artifacts/prod_cheatcode_audit/capture_smooth_wide.py" "$audit_dir/capture_observations.py"
cp "$plan_dir"/{plan_summary.json,moveit_joint_trajectory.json} "$audit_dir/"
if [[ "$oracle_contacts" == true ]]; then
  "$repo/.pixi/envs/default/bin/python" "$repo/scripts/prepare_force_oracle_scene.py" "$audit_dir"
fi
if [[ "$visibility_labels" == true ]]; then
  visibility_mask_args=()
  if [[ "${AIC_VISIBILITY_EXCLUDE_OWN_CAMERA:-false}" == true ]]; then
    visibility_mask_args+=(--exclude-own-camera-visual)
  fi
  if [[ "${AIC_VISIBILITY_WORLD_FOLLOWER:-false}" == true ]]; then
    visibility_mask_args+=(--world-follower)
  fi
  "$repo/.pixi/envs/default/bin/python" "$repo/scripts/prepare_visibility_sensor_scene.py" "$audit_dir" \
    --sources "$repo/outputs/experiments/2026-09-30_transport_continuation/visibility_sensor_sources" \
    --aux-near-m "${AIC_VISIBILITY_AUX_NEAR_M:-.07}" "${visibility_mask_args[@]}"
fi
sha256sum "$audit_dir"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml,moveit_joint_trajectory.json} > "$audit_dir/inputs_sha256.txt"
if docker inspect "$name" >/dev/null 2>&1; then echo "Container already exists: $name" >&2; exit 1; fi
cleanup() { docker stop "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$name" --gpus "device=$gpu" --network bridge --entrypoint /bin/bash \
  -e GZ_PARTITION="$name" \
  -e "NVIDIA_VISIBLE_DEVICES=$gpu" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e ROS_DOMAIN_ID=87 \
  -e AIC_RESULTS_DIR=/audit/results -e "AIC_AUDIT_REPO_HOST=$repo" \
  -v "$audit_dir:/audit" -v "$repo:/repo:ro" -v "$repo:$repo:ro" \
  "$image" -lc 'sleep infinity' > "$audit_dir/container.id"
docker exec -d "$name" bash -lc '
  export ROS_DOMAIN_ID=87
  description_args=()
  if test -f /audit/ur_llb_start.xacro; then
    description_args+=(description_file:=/audit/ur_llb_start.xacro)
  elif test -f /audit/ur_visibility_labels.xacro; then
    description_args+=(description_file:=/audit/ur_visibility_labels.xacro)
  fi
  /entrypoint.sh ground_truth:=true start_aic_engine:=true headless:=true \
    aic_engine_config_file:=/audit/eval_config.yaml \
    world_file:=/audit/world_audit.sdf ros_gz_bridge_config_file:=/audit/bridge_audit.yaml \
    "${description_args[@]}" \
    > /audit/engine.log 2>&1
  printf "%s\n" "$?" > /audit/engine.exit
'
for _ in $(seq 1 90); do
  if test -f "$audit_dir/engine.exit"; then break; fi
  if docker exec "$name" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=87
    timeout 2 ros2 node list 2>/dev/null | grep -q aic_engine
  '; then break; fi
  sleep 2
done
if test -f "$audit_dir/engine.exit"; then echo "Engine exited early" >&2; exit 1; fi
docker exec -d -e AIC_AUDIT_EXTRA_CAMERAS="$([[ "$oracle_contacts" == true ]] && echo port_close,port_reverse || true)" "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=87
  python3 /audit/capture_observations.py > /audit/capture.log 2>&1
'
if [[ "$oracle_contacts" == true ]]; then
  docker exec -d "$name" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=87
    python3 /repo/scripts/capture_oracle_contacts.py > /audit/oracle_capture.log 2>&1
  '
fi
if [[ "$visibility_labels" == true ]]; then
  docker exec -d "$name" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=87
    python3 /repo/scripts/capture_visibility_rgbd.py > /audit/visibility_capture.log 2>&1
  '
fi
local_actor_env=()
teacher_policy=${AIC_MOVEIT_PILOT_TEACHER_POLICY:-aic_teacher_official.OfficialTeacherReplay}
case "$teacher_policy" in
 aic_teacher_official.OfficialTeacherReplay|aic_teacher_official.WideApproachTeacherReplay) ;;
 *) echo "Unsupported teacher policy" >&2; exit 2 ;;
esac
local_actor_env+=(-e "AIC_PILOT_TEACHER_POLICY=$teacher_policy")
if [[ -n "${AIC_MOVEIT_PILOT_WIDE_APPROACH_JSON:-}" ]]; then
 local_actor_env+=(-e "AIC_OFFICIAL_TEACHER_WIDE_APPROACH_JSON=$AIC_MOVEIT_PILOT_WIDE_APPROACH_JSON")
fi
if [[ -n "$local_actor_checkpoint" ]]; then
  local_actor_env+=(
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT=$local_actor_checkpoint"
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_PREPARE=${AIC_MOVEIT_PILOT_LOCAL_ACTOR_PREPARE:-false}"
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_NEAR_GATE_PREFIX=${AIC_MOVEIT_PILOT_LOCAL_NEAR_GATE_PREFIX:-false}"
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_RECORDED_PREFIX=${AIC_MOVEIT_PILOT_LOCAL_RECORDED_PREFIX:-}"
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_RECORDED_EPISODE=${AIC_MOVEIT_PILOT_LOCAL_RECORDED_EPISODE:-llb_0001}"
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_RECORDED_STOP_TIP_AXIAL_M=${AIC_MOVEIT_PILOT_LOCAL_RECORDED_STOP_TIP_AXIAL_M:--0.03}"
    -e "AIC_LOCAL_LOCATOR_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_LOCATOR_CHECKPOINT:?}"
    -e "AIC_LOCAL_ORIENTATION_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_ORIENTATION_CHECKPOINT:?}"
    -e "AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SFP_ORIENTATION_CHECKPOINT:-}"
    -e "AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT_2=${AIC_MOVEIT_PILOT_LOCAL_SFP_ORIENTATION_CHECKPOINT_2:-}"
    -e "AIC_LOCAL_SFP_COARSE_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SFP_COARSE_CHECKPOINT:-}"
    -e "AIC_LOCAL_SFP_BROAD_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SFP_BROAD_CHECKPOINT:-}"
    -e "AIC_LOCAL_SFP_FINE_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SFP_FINE_CHECKPOINT:-}"
    -e "AIC_LOCAL_CAMERA_CALIBRATION=${AIC_MOVEIT_PILOT_LOCAL_CAMERA_CALIBRATION:?}"
    -e "AIC_LOCAL_SC_REFINER_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SC_REFINER_CHECKPOINT:-}"
    -e "AIC_LOCAL_SFP_REFINER_CHECKPOINT=${AIC_MOVEIT_PILOT_LOCAL_SFP_REFINER_CHECKPOINT:-}"
    -e "AIC_LOCAL_SC_ORIENTATION_PRIOR=${AIC_MOVEIT_PILOT_LOCAL_SC_ORIENTATION_PRIOR:-}"
    -e "AIC_LOCAL_SC_ORIENTATION_MAX_DEG=${AIC_MOVEIT_PILOT_LOCAL_SC_ORIENTATION_MAX_DEG:-5}"
    -e "AIC_LOCAL_FAST_LOCKED_SC_PREPROCESS=${AIC_MOVEIT_PILOT_LOCAL_FAST_LOCKED_SC_PREPROCESS:-false}"
    -e "AIC_LOCAL_ACTOR_DEVICE=${AIC_MOVEIT_PILOT_LOCAL_ACTOR_DEVICE:-cuda:0}"
    -e AIC_SHARED_PORT_TCP_DEVICE=cuda:0
    -e AIC_SHARED_PORT_TCP_CONTROL_CLOCK=simulation
    -e AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC=180
    -e AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC=90
    -e AIC_SHARED_PORT_TCP_TRACE=/audit/local_actor_trace.jsonl
  )
fi
docker exec -d "${local_actor_env[@]}" "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=87
  if test -n "${AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT:-}"; then
    export LD_LIBRARY_PATH="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/libjpeg.so.8:$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/libpng16.so.16"
    export PYTHONPATH="/repo/aic_model:/repo/aic_teacher_official:/repo/aic_example_policies:/repo/scripts:/repo/aic_utils/lerobot_robot_aic:$AIC_AUDIT_REPO_HOST/scripts/pythonpath_bootstrap:$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/python3.12/site-packages:${PYTHONPATH:-}"
    ros_executable="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/bin/ros2"
  else
    export PYTHONPATH=/repo/aic_model:/repo/aic_teacher_official:/repo/aic_example_policies:$PYTHONPATH
    ros_executable=ros2
  fi
  export AIC_OFFICIAL_TEACHER_TRAJECTORY=/audit/moveit_joint_trajectory.json
  export AIC_OFFICIAL_TEACHER_ACTION_MODE='"$replay_action_mode"'
  export AIC_OFFICIAL_TEACHER_ONLINE_CHEATCODE_INSERTION='"$online_insertion"'
  export AIC_OFFICIAL_TEACHER_TRACKING_GATE_FORCE_FRACTION='"$tracking_gate_force_fraction"'
  export AIC_OFFICIAL_TEACHER_FORCE_MEDIAN_THRESHOLD_N='"$force_confirm_threshold"'
  export AIC_OFFICIAL_TEACHER_FORCE_SUSTAINED_THRESHOLD_N='"$force_confirm_threshold"'
  export AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY='"$sc_contact_recovery"'
  export AIC_EXPERT_MODE='"$expert_mode"'
  export AIC_OFFICIAL_TEACHER_RECOVERY_INITIAL_TRACKING_GATE_MAX_LATERAL_ERROR_M='"$recovery_initial_lateral_gate"'
  if test -n '"$nominal_initial_lateral_gate"'; then
    export AIC_OFFICIAL_TEACHER_NOMINAL_INITIAL_TRACKING_GATE_MAX_LATERAL_ERROR_M='"$nominal_initial_lateral_gate"'
  fi
  export AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO='"$guarded_lateral_servo"'
  export AIC_OFFICIAL_TEACHER_SC_PREINSERT_SERVO_MAX_BIAS_M='"$sc_preinsert_servo_max_bias"'
  export AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC='"$precontact_port_align_sec"'
  export AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_SPEED_MPS='"$sc_tracking_gate_speed"'
  export AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_FATAL_FORCE_THRESHOLD_N='"$precontact_fatal_force"'
  export AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SOFT_FORCE_THRESHOLD_N='"$precontact_soft_force"'
  export AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_Z_GATE_MAX_HOLD_SEC='"$sc_z_gate_max_hold"'
  export AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_IGNORE_MAX_Z_OFFSET='"$low_force_insert_max_z"'
  export AIC_OFFICIAL_TEACHER_SC_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N='"$sc_low_force_insert_limit"'
  export AIC_OFFICIAL_TEACHER_SC_RECOVERY_MAX_BACKOFF_DISTANCE_M='"$sc_recovery_max_backoff"'
  export AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_SEC='"$sc_recovery_backoff_sec"'
  export AIC_OFFICIAL_TEACHER_SC_RECOVERY_REQUIRED_MEASURED_BACKOFF_M='"$sc_recovery_required_measured"'
  export AIC_OFFICIAL_TEACHER_RUNTIME_TRACE=/audit/runtime_trace.jsonl
  export AIC_OFFICIAL_TEACHER_FINAL_HOLD_SEC=2
  "$ros_executable" run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p "policy:=$AIC_PILOT_TEACHER_POLICY" > /audit/policy.log 2>&1
'
deadline=$((SECONDS + ${AIC_MOVEIT_PILOT_DEADLINE_SEC:-900}))
while ! test -f "$audit_dir/engine.exit" && (( SECONDS < deadline )); do
  if test -s "$audit_dir/results/scoring.yaml" &&
     { grep -q 'All Trials Processed!' "$audit_dir/engine.log" ||
       grep -q 'process has died.*aic_engine' "$audit_dir/engine.log"; }; then
    printf 'score_written_engine_finished\n' > "$audit_dir/launcher.exit"
    break
  fi
  sleep 5
done
if ! test -f "$audit_dir/engine.exit" && ! test -f "$audit_dir/launcher.exit"; then
  printf 'timeout\n' > "$audit_dir/launcher.exit"
fi
if test -f "$audit_dir/engine.exit"; then cat "$audit_dir/engine.exit"; else cat "$audit_dir/launcher.exit"; fi
