#!/usr/bin/env bash
set -euo pipefail

audit_dir="$(realpath "${1:?diagnostic directory with eval_config.yaml}")"
variant="${2:?across_cards or outside_left}"
case "$variant" in across_cards|outside_left) ;; *) exit 2 ;; esac
repo="$(realpath "$(dirname "$0")/../..")"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
name="aic_cable_route_${variant}_20260923"
test -f "$audit_dir/eval_config.yaml"
cp "$repo/artifacts/prod_cheatcode_audit/official_qualification/capture_observations.py" "$audit_dir/capture_observations.py"
cp "$repo/artifacts/prod_cheatcode_audit/CableRouteDiagnostic.py" "$audit_dir/CableRouteDiagnostic.py"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Audit container already exists: $name" >&2
  exit 1
fi
cleanup() { docker stop "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$name" --gpus 'device=1' --net=host \
  --entrypoint /bin/bash \
  -e NVIDIA_VISIBLE_DEVICES=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e ROS_DOMAIN_ID=86 \
  -e AIC_RESULTS_DIR=/audit/results -v "$audit_dir:/audit" \
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
docker exec -e "AIC_CABLE_ROUTE_VARIANT=$variant" \
  -e "AIC_CABLE_ROUTE_LOW_OFFSET_M=${AIC_CABLE_ROUTE_LOW_OFFSET_M:-0.105}" -d "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  export PYTHONPATH=/audit:$PYTHONPATH
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=CableRouteDiagnostic > /audit/policy.log 2>&1
'
deadline=$((SECONDS + 360))
while ! test -f "$audit_dir/engine.exit" && (( SECONDS < deadline )); do
  if test -s "$audit_dir/results/scoring.yaml" && \
     grep -q 'All Trials Processed!' "$audit_dir/engine.log"; then
    printf 'all_trials_processed\n' > "$audit_dir/engine.exit"
    break
  fi
  sleep 5
done
if ! test -f "$audit_dir/engine.exit"; then printf 'timeout\n' > "$audit_dir/engine.exit"; fi
cat "$audit_dir/engine.exit"
