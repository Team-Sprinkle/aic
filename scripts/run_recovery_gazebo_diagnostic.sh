#!/usr/bin/env bash
set -euo pipefail

audit_dir="$(realpath "${1:?diagnostic directory with eval_config.yaml}")"
variant="${2:?stock or across_cards or outside_left or outside_left_behind_first or outside_left_far_return or outside_left_clear_return or outside_left_low_cross_return}"
case "$variant" in stock|stock_then_local|across_cards|outside_left|outside_left_behind_first|outside_left_far_return|outside_left_clear_return|outside_left_low_cross_return) ;; *) exit 2 ;; esac
repo="$(realpath "$(dirname "$0")/..")"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
name="aic_recovery_diagnostic_${variant}_20260924"
gpu="${AIC_AUDIT_GPU:-1}"
visibility_labels="${AIC_AUDIT_VISIBILITY_LABELS:-false}"
test -f "$audit_dir/eval_config.yaml"
if [[ "${AIC_AUDIT_SMOOTH_CAPTURE:-0}" == 1 ]]; then
  cp "$repo/artifacts/prod_cheatcode_audit/capture_smooth_wide.py" "$audit_dir/capture_observations.py"
else
  cp "$repo/artifacts/prod_cheatcode_audit/official_qualification/capture_observations.py" "$audit_dir/capture_observations.py"
fi
if [[ "$variant" == stock ]]; then
  policy=aic_example_policies.ros.CheatCode
elif [[ "$variant" == stock_then_local ]]; then
  policy=aic_example_policies.ros.CheatCodeThenLocal
else
  cp "$repo/artifacts/prod_cheatcode_audit/CableRouteDiagnostic.py" "$audit_dir/CableRouteDiagnostic.py"
  policy=CableRouteDiagnostic
fi
if [[ "$visibility_labels" == true ]]; then
  visibility_extra=()
  if [[ "${AIC_VISIBILITY_WORLD_FOLLOWER:-false}" == true ]]; then
    visibility_extra+=(--world-follower)
  fi
  if [[ "${AIC_VISIBILITY_PRIMARY_RGBD:-false}" == true ]]; then
    visibility_extra+=(--primary-rgbd)
  fi
  if [[ "${AIC_VISIBILITY_EXCLUDE_OWN_CAMERA:-false}" == true ]]; then
    visibility_extra+=(--exclude-own-camera-visual)
  fi
  "$repo/.pixi/envs/default/bin/python" "$repo/scripts/prepare_visibility_sensor_scene.py" "$audit_dir" \
    --sources "$repo/outputs/experiments/2026-09-30_transport_continuation/visibility_sensor_sources" \
    "${visibility_extra[@]}"
  cp "$repo/scripts/capture_visibility_rgbd.py" "$audit_dir/capture_visibility_rgbd.py"
fi
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Audit container already exists: $name" >&2
  exit 1
fi
cleanup() { docker stop "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$name" --gpus "device=$gpu" --net=host \
  --entrypoint /bin/bash \
  -e "NVIDIA_VISIBLE_DEVICES=$gpu" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e ROS_DOMAIN_ID=86 \
  -e AIC_RESULTS_DIR=/audit/results -e "AIC_AUDIT_REPO_HOST=$repo" \
  -v "$audit_dir:/audit" -v "$repo:/repo:ro" -v "$repo:$repo:ro" \
  "$image" -lc 'sleep infinity' > "$audit_dir/container.id"
docker exec -d "$name" bash -lc '
  export ROS_DOMAIN_ID=86
  world_args=()
  if test -f /audit/world_audit.sdf && test -f /audit/bridge_audit.yaml; then
    world_args+=(world_file:=/audit/world_audit.sdf)
    world_args+=(ros_gz_bridge_config_file:=/audit/bridge_audit.yaml)
  fi
  if test -f /audit/ur_visibility_labels.xacro; then
    world_args+=(description_file:=/audit/ur_visibility_labels.xacro)
  fi
  /entrypoint.sh ground_truth:=true start_aic_engine:=true headless:=true \
    aic_engine_config_file:=/audit/eval_config.yaml "${world_args[@]}" > /audit/engine.log 2>&1
  printf "%s\n" "$?" > /audit/engine.exit
'
for _ in $(seq 1 90); do
  if test -f "$audit_dir/engine.exit"; then break; fi
  if docker exec "$name" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
    timeout 2 ros2 node list 2>/dev/null | grep -q aic_engine
  '; then break; fi
  sleep 2
done
if test -f "$audit_dir/engine.exit"; then
  echo "Engine exited before policy startup" >&2
  exit 1
fi
docker exec -d -e "AIC_AUDIT_CAPTURE_HZ=${AIC_AUDIT_CAPTURE_HZ:-0}" "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  python3 /audit/capture_observations.py > /audit/capture.log 2>&1
'
if [[ "$visibility_labels" == true ]]; then
  docker exec -d -e "AIC_VISIBILITY_PRIMARY_RGBD=${AIC_VISIBILITY_PRIMARY_RGBD:-false}" "$name" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
    if test "${AIC_VISIBILITY_PRIMARY_RGBD:-false}" = true; then
      export AIC_VISIBILITY_SOURCE_ROOT=primary
    fi
    python3 /audit/capture_visibility_rgbd.py > /audit/visibility_capture.log 2>&1
  '
fi
actor_env=()
if [[ "$variant" == stock_then_local ]]; then
  actor_env+=(
    -e "AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT=${AIC_AUDIT_LOCAL_ACTOR_CHECKPOINT:?}"
    -e "AIC_LOCAL_LOCATOR_CHECKPOINT=${AIC_AUDIT_LOCAL_LOCATOR_CHECKPOINT:?}"
    -e "AIC_LOCAL_ORIENTATION_CHECKPOINT=${AIC_AUDIT_LOCAL_ORIENTATION_CHECKPOINT:?}"
    -e "AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT=${AIC_AUDIT_LOCAL_SFP_ORIENTATION_CHECKPOINT:-}"
    -e "AIC_LOCAL_CAMERA_CALIBRATION=${AIC_AUDIT_LOCAL_CAMERA_CALIBRATION:?}"
    -e "AIC_LOCAL_SFP_REFINER_CHECKPOINT=${AIC_AUDIT_LOCAL_SFP_REFINER_CHECKPOINT:-}"
    -e "AIC_LOCAL_ACTOR_DEVICE=${AIC_AUDIT_LOCAL_ACTOR_DEVICE:-cpu}"
    -e "AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M=${AIC_AUDIT_SFP_HANDOFF_Z_OFFSET_M:-.03}"
    -e "AIC_SHARED_PORT_TCP_DEVICE=cuda:0"
    -e "AIC_SHARED_PORT_TCP_CONTROL_CLOCK=simulation"
    -e "AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC=180"
    -e "AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC=90"
    -e "AIC_SHARED_PORT_TCP_TRACE=/audit/local_actor_trace.jsonl"
  )
fi
docker exec "${actor_env[@]}" -e "AIC_CABLE_ROUTE_VARIANT=$variant" -e "AIC_DIAGNOSTIC_POLICY=$policy" \
  -e "AIC_CABLE_ROUTE_LOW_OFFSET_M=${AIC_CABLE_ROUTE_LOW_OFFSET_M:-0.105}" \
  -e "AIC_CABLE_ROUTE_BEHIND_EXTRA_M=${AIC_CABLE_ROUTE_BEHIND_EXTRA_M:-0}" \
  -e "AIC_CABLE_ROUTE_LANE_EXTRA_M=${AIC_CABLE_ROUTE_LANE_EXTRA_M:-0}" \
  -e "AIC_CABLE_ROUTE_FORWARD_EXTRA_M=${AIC_CABLE_ROUTE_FORWARD_EXTRA_M:-0}" -d "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  export PYTHONPATH=/audit:$PYTHONPATH
  ros_executable=ros2
  if test "$AIC_CABLE_ROUTE_VARIANT" = stock_then_local; then
    export LD_LIBRARY_PATH="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/libjpeg.so.8:$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/libpng16.so.16"
    export PYTHONPATH="/repo/aic_model:/repo/aic_example_policies:/repo/scripts:/repo/aic_utils/lerobot_robot_aic:$AIC_AUDIT_REPO_HOST/scripts/pythonpath_bootstrap:$AIC_AUDIT_REPO_HOST/.pixi/envs/default/lib/python3.12/site-packages:$PYTHONPATH"
    ros_executable="$AIC_AUDIT_REPO_HOST/.pixi/envs/default/bin/ros2"
  fi
  "$ros_executable" run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:="$AIC_DIAGNOSTIC_POLICY" > /audit/policy.log 2>&1
'
deadline=$((SECONDS + ${AIC_AUDIT_DEADLINE_SEC:-360}))
while ! test -f "$audit_dir/engine.exit" && (( SECONDS < deadline )); do
  if test -s "$audit_dir/results/scoring.yaml" && \
     grep -q 'All Trials Processed!' "$audit_dir/engine.log"; then
    printf 'all_trials_processed\n' > "$audit_dir/launcher.exit"
    break
  fi
  sleep 5
done
if ! test -f "$audit_dir/engine.exit" && ! test -f "$audit_dir/launcher.exit"; then
  printf 'timeout\n' > "$audit_dir/launcher.exit"
fi
if test -f "$audit_dir/engine.exit"; then cat "$audit_dir/engine.exit"; else cat "$audit_dir/launcher.exit"; fi
