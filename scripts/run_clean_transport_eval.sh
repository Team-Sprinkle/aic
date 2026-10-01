#!/usr/bin/env bash
# Bounded observation-only development rollout. No teacher, scene geometry or guide in actor.
set -euo pipefail
repo=${AIC_TRANSPORT_EVAL_REPO:-$(realpath "$(dirname "$0")/..")}
if [[ "${AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT:-0}" != 1 ]]; then
 export AIC_TRANSPORT_EVAL_REPO="$repo" AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT=1
 mkdir -p "$repo/outputs/experiments/2026-09-30_transport_continuation/launcher_snapshots"
 snapshot=$(mktemp "$repo/outputs/experiments/2026-09-30_transport_continuation/launcher_snapshots/eval_XXXXXX.sh")
 cp "$0" "$snapshot"
 exec bash "$snapshot" "$@"
fi
case_name=${1:?development case}
checkpoint=$(realpath "${2:?checkpoint}")
audit=$(realpath -m "${3:?new output}")
gpu=${AIC_TRANSPORT_EVAL_GPU:-7}
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "GPU index must be 0..7" >&2; exit 2; }
: "${AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA:?Set the componentwise action cap explicitly, e.g. .02 or .06}"
scene="$repo/outputs/experiments/2026-09-30_transport_continuation/scenes/$case_name"
export DOCKER_HOST=${DOCKER_HOST:-unix:///run/user/1008/docker.sock}
name="aic_clean_transport_eval_$(basename "$audit")"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
test ! -e "$audit" || { echo "Preserve existing rollout: $audit" >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then echo "Existing container: $name" >&2; exit 1; fi
mkdir -p "$audit"
export AIC_TRANSPORT_EVAL_AUDIT="$audit" AIC_TRANSPORT_EVAL_CHECKPOINT="$checkpoint" AIC_TRANSPORT_EVAL_CASE="$case_name"
export AIC_TRANSPORT_EVAL_GPU="$gpu"
"$repo/.pixi/envs/default/bin/python" - <<'PY'
import json,os
from pathlib import Path
d=dict(schema='aic_clean_transport_eval_manifest/v1',case=os.environ['AIC_TRANSPORT_EVAL_CASE'],
 checkpoint=os.environ['AIC_TRANSPORT_EVAL_CHECKPOINT'],
 policy=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','aic_example_policies.ros.RunSharedPortTCP'),
 stiffness=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS','[90,90,90,50,50,50]')),
 damping=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING','[50,50,50,20,20,20]')),
 rootless=True,gpu=int(os.environ['AIC_TRANSPORT_EVAL_GPU']),control_clock='simulation',simulation_budget_s=70,wall_policy_budget_s=180,
 diagnostic_teacher_or_geometry=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','').endswith('RecordedTeacherControlDiagnostic'),
 teacher_replay_source=os.environ.get('AIC_TEACHER_REPLAY_SOURCE'),teacher_replay_cap=os.environ.get('AIC_TEACHER_REPLAY_CAP'),reserved_final=False)
d['max_translation_delta_m']=float(os.environ['AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA'])
Path(os.environ['AIC_TRANSPORT_EVAL_AUDIT'],'eval_manifest.json').write_text(json.dumps(d,indent=2)+'\n')
PY
cp "$scene"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} "$audit/"
cp "$repo/artifacts/prod_cheatcode_audit/capture_smooth_wide.py" "$audit/capture_observations.py"
cp "$0" "$audit/launcher_snapshot.sh"
sha256sum "$checkpoint" "$audit"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} \
 "$repo/aic_example_policies/aic_example_policies/ros/RunSharedPortTCP.py" > "$audit/inputs_sha256.txt"
cleanup() { docker stop "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$name" --gpus "device=$gpu" --shm-size=4g --net=host --entrypoint /bin/bash \
 -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=all \
 -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e ROS_DOMAIN_ID=86 \
 -e AIC_RESULTS_DIR=/audit/results -e AIC_SHARED_PORT_TCP_CHECKPOINT="$checkpoint" \
 -e AIC_TRANSPORT_EVAL_POLICY="${AIC_TRANSPORT_EVAL_POLICY:-aic_example_policies.ros.RunSharedPortTCP}" \
 -e AIC_TEACHER_REPLAY_SOURCE="${AIC_TEACHER_REPLAY_SOURCE:-}" \
 -e AIC_TEACHER_REPLAY_CAP="${AIC_TEACHER_REPLAY_CAP:-}" \
 -e AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA="$AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA" \
 -e AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS="${AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS:-[90,90,90,50,50,50]}" \
 -e AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING="${AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING:-[50,50,50,20,20,20]}" \
 -v "$audit:/audit" -v "$repo:$repo:ro" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act:ro \
 -w "$repo" "$image" -lc 'sleep infinity' > "$audit/container.id"
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export ROS_DOMAIN_ID=86 RMW_IMPLEMENTATION=rmw_zenoh_cpp
 ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id aic_world \
   --ros-args -p use_sim_time:=true > /audit/world_frame.log 2>&1 &
 /entrypoint.sh ground_truth:=false start_aic_engine:=true headless:=true \
  aic_engine_config_file:=/audit/eval_config.yaml world_file:=/audit/world_audit.sdf \
  ros_gz_bridge_config_file:=/audit/bridge_audit.yaml > /audit/engine.log 2>&1
 printf "%s\n" "$?" > /audit/engine.exit
'
for _ in $(seq 1 90); do
 if test -f "$audit/engine.exit"; then break; fi
 if docker exec "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
  timeout 2 ros2 node list 2>/dev/null | grep -q aic_engine
 '; then break; fi
 sleep 2
done
test ! -f "$audit/engine.exit" || { echo 'Engine exited early' >&2; exit 1; }
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
 python3 /audit/capture_observations.py > /audit/capture.log 2>&1
'
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=86
 export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
 export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
 export PYTHONPATH="$PWD/.pixi/envs/default/lib/python3.12/site-packages:$PWD/aic_model:$PWD/aic_example_policies:$PWD/scripts/pythonpath_bootstrap:${PYTHONPATH:-}"
 export AIC_CHECKOUT_PYTHONPATH="$PWD/aic_model:$PWD/aic_example_policies"
 export PYTHONPATH="$PWD/aic_utils/lerobot_robot_aic:$PYTHONPATH"
 export AIC_SHARED_PORT_TCP_DEVICE=cuda:0 AIC_SHARED_PORT_TCP_CONTROL_CLOCK=simulation
 export AIC_SHARED_PORT_TCP_CONTROL_HZ=20 AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC=70
 export AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC=180 AIC_SHARED_PORT_TCP_TRACE=/audit/runtime_trace.jsonl
 export AIC_SHARED_PORT_TCP_MAX_TRANSLATION_DELTA="$AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA" AIC_SHARED_PORT_TCP_MAX_ROTATION_DELTA=.2
 .pixi/envs/default/bin/ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
   -p policy:="$AIC_TRANSPORT_EVAL_POLICY" > /audit/policy.log 2>&1
 printf "%s\n" "$?" > /audit/policy.exit
'
deadline=$((SECONDS+450))
while ! test -f "$audit/engine.exit" && (( SECONDS < deadline )); do
 if test -s "$audit/results/scoring.yaml" && grep -q 'All Trials Processed!' "$audit/engine.log"; then break; fi
 sleep 5
done
if (( SECONDS >= deadline )); then echo timeout > "$audit/launcher.exit"; exit 1; fi
echo finished > "$audit/launcher.exit"
