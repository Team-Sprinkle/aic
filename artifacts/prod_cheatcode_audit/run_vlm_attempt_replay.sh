#!/usr/bin/env bash
# Replay one archived agent/VLM joint trajectory against its exact saved scene.
set -euo pipefail

audit_dir="$(realpath "${1:?output directory with eval_config.yaml and smooth_trajectory.json}")"
repo="$(realpath "$(dirname "$0")/../..")"
name="${AIC_AUDIT_CONTAINER_NAME:-aic_vlm_attempt_replay_20260923}"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
test -f "$audit_dir/eval_config.yaml"
test -f "$audit_dir/smooth_trajectory.json"
cp "$repo/artifacts/prod_cheatcode_audit/official_qualification/capture_observations.py" \
  "$audit_dir/capture_observations.py"
printf '%q ' "$0" "$@" > "$audit_dir/host_command.txt"
printf '\n' >> "$audit_dir/host_command.txt"

if docker inspect "$name" >/dev/null 2>&1; then
  echo "Audit container already exists: $name" >&2
  exit 1
fi

cleanup() {
  docker stop "$name" >/dev/null 2>&1 || true
  docker rm "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$name" --gpus 'device=1' --net=host \
  --entrypoint /bin/bash \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true \
  -e ROS_DOMAIN_ID=86 \
  -e AIC_RESULTS_DIR=/audit/results \
  -v "$audit_dir:/audit" \
  -v "$repo/aic_teacher_official:/teacher:ro" \
  -v "$repo/aic_model:/model:ro" \
  "$image" -lc 'sleep infinity' > "$audit_dir/container.id"

docker exec -d "$name" bash -lc '
  export ROS_DOMAIN_ID=86
  /entrypoint.sh ground_truth:=true start_aic_engine:=true headless:=true \
    aic_engine_config_file:=/audit/eval_config.yaml > /audit/engine.log 2>&1
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

docker exec -d "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  python3 /audit/capture_observations.py > /audit/capture.log 2>&1
'
docker exec -d "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export PYTHONPATH=/teacher:/model:$PYTHONPATH
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  export AIC_OFFICIAL_TEACHER_TRAJECTORY=/audit/smooth_trajectory.json
  export AIC_OFFICIAL_TEACHER_ACTION_MODE=joint_position_then_cheatcode
  export AIC_EXPERT_MODE=nominal
  export AIC_OFFICIAL_TEACHER_FT_THRESHOLD_N=15
  export AIC_OFFICIAL_TEACHER_RUNTIME_TRACE=/audit/runtime_trace.jsonl
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p policy:=aic_teacher_official.OfficialTeacherReplay > /audit/teacher.log 2>&1
  printf "%s\n" "$?" > /audit/teacher.exit
'

deadline=$((SECONDS + 1200))
while ! test -f "$audit_dir/engine.exit" && (( SECONDS < deadline )); do
  if test -s "$audit_dir/results/scoring.yaml" && \
     grep -q 'All Trials Processed!' "$audit_dir/engine.log"; then
    printf 'all_trials_processed\n' > "$audit_dir/engine.exit"
    break
  fi
  if test -f "$audit_dir/teacher.exit" && ! test -s "$audit_dir/results/scoring.yaml"; then
    printf 'teacher_exited_before_score\n' > "$audit_dir/engine.exit"
    break
  fi
  sleep 5
done
if ! test -f "$audit_dir/engine.exit"; then
  printf 'timeout\n' > "$audit_dir/engine.exit"
fi
cat "$audit_dir/engine.exit"
