#!/usr/bin/env bash
# Bounded observation-only development rollout. No teacher, scene geometry or guide in actor.
set -euo pipefail
# Serialize project render/policy workers on each actual GPU. Locks precede
# domain locks and use numeric order, so independent sequential collectors
# can share a GPU without overlapping simulators or deadlocking two GPUs.
render_lock_gpu=${AIC_TRANSPORT_EVAL_GPU:-7}
policy_lock_gpu=${AIC_TRANSPORT_EVAL_POLICY_GPU:-$render_lock_gpu}
[[ "$render_lock_gpu" =~ ^[0-7]$ && "$policy_lock_gpu" =~ ^[0-7]$ ]] || exit 2
lock_gpus=("$render_lock_gpu")
if [[ "$policy_lock_gpu" != "$render_lock_gpu" ]]; then
 if (( policy_lock_gpu < render_lock_gpu )); then lock_gpus=("$policy_lock_gpu" "$render_lock_gpu"); else lock_gpus+=("$policy_lock_gpu"); fi
fi
for lock_gpu in "${lock_gpus[@]}"; do
 case ",${AIC_PROJECT_GPU_LOCKS_HELD:-}," in
  *",$lock_gpu,"*) ;;
  *) export AIC_PROJECT_GPU_LOCKS_HELD="${AIC_PROJECT_GPU_LOCKS_HELD:+$AIC_PROJECT_GPU_LOCKS_HELD,}$lock_gpu"
     exec flock -w 3600 "/var/tmp/chmin_aic_gpu${lock_gpu}.lock" bash "$0" "$@" ;;
 esac
done
# Each worker has its own ROS domain, network namespace and Gazebo partition.
eval_domain=${AIC_TRANSPORT_EVAL_ROS_DOMAIN:-86}
[[ "$eval_domain" =~ ^[0-9]+$ ]] && (( eval_domain <= 232 )) || { echo 'Invalid ROS domain' >&2; exit 2; }
if [[ "$eval_domain" == 86 && "${AIC_DOMAIN86_LOCK_HELD:-0}" != 1 ]]; then
 export AIC_DOMAIN86_LOCK_HELD=1
 exec flock -w 600 /var/tmp/chmin_aic_gazebo_domain86.lock bash "$0" "$@"
fi
if [[ "$eval_domain" != 86 && "${AIC_CUSTOM_DOMAIN_LOCK_HELD:-}" != "$eval_domain" ]]; then
 export AIC_CUSTOM_DOMAIN_LOCK_HELD="$eval_domain"
 exec flock -w 600 "/var/tmp/chmin_aic_gazebo_domain${eval_domain}.lock" bash "$0" "$@"
fi
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
policy_gpu=${AIC_TRANSPORT_EVAL_POLICY_GPU:-$gpu}
[[ "$policy_gpu" =~ ^[0-7]$ ]] || { echo "Policy GPU index must be 0..7" >&2; exit 2; }
if [[ "$policy_gpu" == "$gpu" ]]; then
 docker_gpu_devices="device=$gpu"
 docker_visible_devices="$gpu"
 policy_cuda_device=cuda:0
 engine_cuda_visible_device=0
else
 docker_gpu_devices="\"device=$gpu,$policy_gpu\""
 docker_visible_devices="$gpu,$policy_gpu"
 # NVIDIA's container runtime enumerates selected host GPUs in ascending
 # index order, regardless of NVIDIA_VISIBLE_DEVICES list order.
 if (( policy_gpu < gpu )); then
  policy_cuda_device=cuda:0
  engine_cuda_visible_device=1
 else
  policy_cuda_device=cuda:1
  engine_cuda_visible_device=0
 fi
fi
: "${AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA:?Set the componentwise action cap explicitly, e.g. .02 or .06}"
scene=${AIC_TRANSPORT_EVAL_SCENE_DIR:-"$repo/outputs/experiments/2026-09-30_transport_continuation/scenes/$case_name"}
scene=$(realpath "$scene")
export DOCKER_HOST=${DOCKER_HOST:-unix:///run/user/1008/docker.sock}
name="aic_clean_transport_eval_$(basename "$audit")"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
test ! -e "$audit" || { echo "Preserve existing rollout: $audit" >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then echo "Existing container: $name" >&2; exit 1; fi
mkdir -p "$audit"
if [[ -n "${AIC_SFP_MOVEIT_PLAN_SOURCE:-}" ]]; then
 cp "$AIC_SFP_MOVEIT_PLAN_SOURCE" "$audit/moveit_plan.json"
fi
export AIC_TRANSPORT_EVAL_AUDIT="$audit" AIC_TRANSPORT_EVAL_CHECKPOINT="$checkpoint" AIC_TRANSPORT_EVAL_CASE="$case_name"
export AIC_TRANSPORT_EVAL_GPU="$gpu"
export AIC_TRANSPORT_EVAL_POLICY_GPU="$policy_gpu"
export AIC_TRANSPORT_EVAL_POLICY_CUDA_DEVICE="$policy_cuda_device"
export AIC_TRANSPORT_EVAL_ENGINE_CUDA_VISIBLE_DEVICE="$engine_cuda_visible_device"
"$repo/.pixi/envs/default/bin/python" - <<'PY'
import json,os
from pathlib import Path
d=dict(schema='aic_clean_transport_eval_manifest/v1',case=os.environ['AIC_TRANSPORT_EVAL_CASE'],
 checkpoint=os.environ['AIC_TRANSPORT_EVAL_CHECKPOINT'],
 policy=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','aic_example_policies.ros.RunSharedPortTCP'),
 stiffness=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS','[90,90,90,50,50,50]')),
 damping=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING','[50,50,50,20,20,20]')),
 rootless=True,gpu=int(os.environ['AIC_TRANSPORT_EVAL_GPU']),control_clock='simulation',
 simulation_budget_s=float(os.environ.get('AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC','70')),
 wall_policy_budget_s=180,
 policy_gpu=int(os.environ['AIC_TRANSPORT_EVAL_POLICY_GPU']),
 policy_cuda_device=os.environ['AIC_TRANSPORT_EVAL_POLICY_CUDA_DEVICE'],
 engine_cuda_visible_device=os.environ['AIC_TRANSPORT_EVAL_ENGINE_CUDA_VISIBLE_DEVICE'],
 oracle_contact_capture=os.environ.get('AIC_TRANSPORT_EVAL_ORACLE_CONTACTS','false')=='true',
 diagnostic_teacher_or_geometry=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','').endswith(('RecordedTeacherControlDiagnostic','RecordedTeacherChunkExecutorDiagnostic','SfpWideApproachTeacher')),
 sfp_wide_approach_JSON=os.environ.get('AIC_SFP_WIDE_APPROACH_JSON') or None,
 rule_transport_template=os.environ.get('AIC_RULE_TRANSPORT_TEMPLATE') or None,
 temporal_stop_threshold=os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD') or None,
 route_diffusion_checkpoint=os.environ.get('AIC_ROUTE_DIFFUSION_CHECKPOINT') or None,
 sfp_native_jpeg_quality=os.environ.get('AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY') or None,
 route_diffusion_zero_previous=os.environ.get('AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS','0')=='1',
 route_diffusion_prewarm=os.environ.get('AIC_ROUTE_DIFFUSION_PREWARM','0')=='1',
 route_diffusion_sample_steps=(int(os.environ['AIC_ROUTE_DIFFUSION_SAMPLE_STEPS'])
                               if os.environ.get('AIC_ROUTE_DIFFUSION_SAMPLE_STEPS') else None),
 execute_complete_chunk=os.environ.get('AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK','0')=='1',
 handoff_prior_json=os.environ.get('AIC_TEMPORAL_HANDOFF_PRIOR_JSON') or None,
 sfp_diagnostic_stop=os.environ.get('AIC_SFP_DIAGNOSTIC_STOP','0')=='1',
 temporal_stop_streak=int(os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_STREAK','3')),
 temporal_stop_min_sim_s=float(os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S','0')),
 sc_goal_probe_checkpoint=os.environ.get('AIC_SC_GOAL_PROBE_CHECKPOINT') or None,
 sc_goal_servo_trigger_m=float(os.environ.get('AIC_SC_GOAL_SERVO_TRIGGER_M','.035')),
 sc_goal_servo_step_m=float(os.environ.get('AIC_SC_GOAL_SERVO_STEP_M','.008')),
 sc_goal_servo_stop_m=float(os.environ.get('AIC_SC_GOAL_SERVO_STOP_M','.006')),
 sc_goal_stop_only=os.environ.get('AIC_SC_GOAL_STOP_ONLY','0')=='1',
 sc_goal_stop_only_threshold_m=float(os.environ.get('AIC_SC_GOAL_STOP_ONLY_THRESHOLD_M','.030')),
 sc_goal_stop_only_streak=int(os.environ.get('AIC_SC_GOAL_STOP_ONLY_STREAK','3')),
 sc_visual_standoff_checkpoint=os.environ.get('AIC_SC_VISUAL_STANDOFF_CHECKPOINT') or None,
 sc_visual_stop_mode=os.environ.get('AIC_SC_VISUAL_STOP_MODE','visual'),
 sc_visual_standoff_calibration=os.environ.get('AIC_SC_VISUAL_STANDOFF_CALIBRATION') or None,
 sc_visual_standoff_trigger_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_TRIGGER_M','.060')),
 sc_visual_standoff_step_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STEP_M','.008')),
 sc_visual_standoff_stop_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STOP_M','.008')),
 sc_visual_standoff_stiffness=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STIFFNESS','0')),
 sc_visual_standoff_damping=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_DAMPING','80')),
 sc_visual_standoff_orientation=os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION','0')=='1',
 sc_visual_standoff_orientation_probe=os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT') or None,
 sc_visual_standoff_orientation_yaw_blend=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND','1')),
 sc_visual_resize='opencv_area_288x256' if os.environ.get('AIC_SC_VISUAL_STANDOFF_CHECKPOINT') else None,
 sc_visual_start_elapsed_sim_s=30.0 if os.environ.get('AIC_SC_VISUAL_STANDOFF_CHECKPOINT') else None,
 sc_transport_z_floor_m=float(os.environ.get('AIC_SC_TRANSPORT_Z_FLOOR_M','0')),
 sc_direct_native_rgb=os.environ.get('AIC_SC_DIRECT_NATIVE_RGB','0')=='1',
 sc_shadow_handoff=os.environ.get('AIC_SC_SHADOW_HANDOFF','0')=='1',
 sc_shadow_history_audit=os.environ.get('AIC_SC_SHADOW_HISTORY_AUDIT','0')=='1',
 sc_shadow_use_history_orientation=os.environ.get('AIC_SC_SHADOW_USE_HISTORY_ORIENTATION','0')=='1',
 sc_shadow_native_snapshot_dir=os.environ.get('AIC_SC_SHADOW_NATIVE_SNAPSHOT_DIR',''),
 sc_shadow_direct_native_rgb=os.environ.get('AIC_SC_SHADOW_DIRECT_NATIVE_RGB','0')=='1',
 sc_shadow_counterfactual_request=os.environ.get('AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST','0')=='1',
 sc_shadow_identity_retry=os.environ.get('AIC_SC_SHADOW_IDENTITY_RETRY','0')=='1',
 sc_shadow_allow_stale_pause=os.environ.get('AIC_SC_SHADOW_ALLOW_STALE_PAUSE','0')=='1',
 sc_shadow_locator=os.environ.get('AIC_SC_SHADOW_LOCATOR') or None,
 sc_shadow_identity=os.environ.get('AIC_SC_SHADOW_IDENTITY') or None,
 sc_shadow_orientation_on_identity_miss=os.environ.get(
  'AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS','0')=='1',
 sc_shadow_orientation=os.environ.get('AIC_SC_SHADOW_ORIENTATION') or None,
 sc_shadow_refiner=os.environ.get('AIC_SC_SHADOW_REFINER') or None,
 joint_native_refiner=os.environ.get('AIC_JOINT_NATIVE_REFINER_CHECKPOINT') or None,
 sc_candidate_association=os.environ.get('AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT') or None,
 sc_candidate_gate_mode=os.environ.get('AIC_SC_CANDIDATE_GATE_MODE') or 'intersection',
 sc_requested_history_locator=os.environ.get('AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT') or None,
 sc_native_pose_max_reprojection_px=float(os.environ.get('AIC_SC_NATIVE_POSE_MAX_REPROJECTION_PX','2')),
 sfp_rule_priors=os.environ.get('AIC_SFP_RULE_PRIORS') or None,
 sfp_requested_locator=os.environ.get('AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT') or None,
 sfp_requested_backbone=os.environ.get('AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT') or None,
 ROS_domain=int(os.environ.get('AIC_TRANSPORT_EVAL_ROS_DOMAIN','86')),
 network_isolation='bridge namespace and unique per-container GZ_PARTITION',
 CPU_BLAS_threads=1,
 teacher_replay_source=os.environ.get('AIC_TEACHER_REPLAY_SOURCE'),teacher_replay_cap=os.environ.get('AIC_TEACHER_REPLAY_CAP'),reserved_final=False)
d['privileged_oracle_local_diagnostic']=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','').endswith(('RuleThenOracleTipInsertionDiagnostic','RuleActorThenOracleContinuationDiagnostic'))
d['oracle_mating_prior']=os.environ.get('AIC_ORACLE_MATING_PRIOR') or None
d['privileged_teacher_prefix_diagnostic']=bool(os.environ.get('AIC_TEMPORAL_TEACHER_PREFIX_FILE'))
d['temporal_teacher_prefix_file']=os.environ.get('AIC_TEMPORAL_TEACHER_PREFIX_FILE') or None
d['privileged_corrective_teacher_diagnostic']=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','').endswith(('RunTemporalThenCheatCodeDiagnostic','RuleActorThenOracleContinuationDiagnostic'))
d['privileged_moveit_replay_diagnostic']=os.environ.get('AIC_TRANSPORT_EVAL_POLICY','').endswith('RunTemporalThenMoveItDiagnostic')
d['moveit_plan_source']=os.environ.get('AIC_SFP_MOVEIT_PLAN_SOURCE') or None
d['moveit_live_replan']=os.environ.get('AIC_SFP_MOVEIT_LIVE_REPLAN','0')=='1'
d['corrective_actor_duration_s']=(float(os.environ['AIC_SFP_CORRECTIVE_ACTOR_DURATION_S'])
                                  if os.environ.get('AIC_SFP_CORRECTIVE_ACTOR_DURATION_S') else None)
d['corrective_teacher_duration_s']=(float(os.environ['AIC_SFP_CORRECTIVE_TEACHER_DURATION_S'])
                                    if os.environ.get('AIC_SFP_CORRECTIVE_TEACHER_DURATION_S') else None)
d['corrective_require_full_insert']=os.environ.get('AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT','0')=='1'
d['max_translation_delta_m']=float(os.environ['AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA'])
d['sfp_local_handoff_tip_standoff_m']=(float(os.environ['AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M'])
                                     if os.environ.get('AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M') else None)
d['sfp_local_handoff_lateral_tolerance_m']=(float(os.environ['AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M'])
                                          if os.environ.get('AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M') else None)
d['sfp_local_pair_orientation']=os.environ.get('AIC_LOCAL_SFP_PAIR_ORIENTATION','0')=='1'
d['sfp_local_lock_samples']=int(os.environ.get('AIC_LOCAL_SFP_LOCK_SAMPLES','5'))
d['sfp_post_handoff_teacher_diagnostic']=os.environ.get('AIC_SFP_POST_HANDOFF_TEACHER_DIAGNOSTIC','0')=='1'
d['sfp_stage_preserving_teacher_diagnostic']=os.environ.get('AIC_SFP_POST_HANDOFF_STAGE_PRESERVING_TEACHER','0')=='1'
d['sfp_stage_preserve_after_local']=os.environ.get('AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL','0')=='1'
d['sfp_local_decision_limit']=int(os.environ.get('AIC_SFP_LOCAL_DECISION_LIMIT','0'))
Path(os.environ['AIC_TRANSPORT_EVAL_AUDIT'],'eval_manifest.json').write_text(json.dumps(d,indent=2)+'\n')
PY
cp "$scene"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} "$audit/"
if [[ -f "$scene/ur_llb_start.xacro" ]]; then
 cp "$scene/ur_llb_start.xacro" "$audit/ur_llb_start.xacro"
fi
cp "$repo/artifacts/prod_cheatcode_audit/capture_smooth_wide.py" "$audit/capture_observations.py"
if [[ "${AIC_TRANSPORT_EVAL_ORACLE_CONTACTS:-false}" == true ]]; then
  "$repo/.pixi/envs/default/bin/python" "$repo/scripts/enable_oracle_contact_capture.py" "$audit" \
    --plugin "$repo/outputs/experiments/2026-09-30_force_oracle/libOracleContactsPlugin.so"
fi
cp "$0" "$audit/launcher_snapshot.sh"
sha256sum "$checkpoint" "$audit"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} \
 "$repo/aic_example_policies/aic_example_policies/ros/RunSharedPortTCP.py" > "$audit/inputs_sha256.txt"
# Freeze all learned inputs used after transport as well as the entry actor.
for additional_checkpoint in "${AIC_LOCKED_GRASP_CHECKPOINT:-}" "${AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT:-}"; do
 if [[ -n "$additional_checkpoint" ]]; then
  sha256sum "$additional_checkpoint" >> "$audit/inputs_sha256.txt"
 fi
done
if [[ -n "${AIC_LOCKED_GRASP_CHECKPOINT:-}" ]]; then
 sha256sum "$repo/scripts/sc_native_grasp_observation.py" \
  "$repo/aic_example_policies/aic_example_policies/ros/RunLockedPortLocal.py" >> "$audit/inputs_sha256.txt"
fi
policy_name=${AIC_TRANSPORT_EVAL_POLICY:-aic_example_policies.ros.RunSharedPortTCP}
policy_basename=${policy_name##*.}
policy_source="$repo/aic_example_policies/aic_example_policies/ros/$policy_basename.py"
if [[ -f "$policy_source" ]]; then
 cp "$policy_source" "$audit/runtime_policy_source.py"
 sha256sum "$policy_source" >> "$audit/inputs_sha256.txt"
fi
if [[ "$policy_basename" == RunNative5RequestedRuleTransport ]]; then
 sha256sum "$repo/scripts/requested_native_port_pose.py" \
  "$repo/scripts/native_port_handoff_verifier.py" \
  "$repo/scripts/native_keypoint_pose_geometry.py" \
  "$AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT" "$AIC_SFP_RULE_MOUNT_PRIORS" >> "$audit/inputs_sha256.txt"
 cp "$repo/scripts/requested_native_port_pose.py" "$audit/requested_native_port_pose_source.py"
 cp "$repo/scripts/native_port_handoff_verifier.py" "$audit/native_port_handoff_verifier_source.py"
 cp "$repo/aic_example_policies/aic_example_policies/ros/RunRuleTransport.py" "$audit/RunRuleTransport_source.py"
 sha256sum "$repo/aic_example_policies/aic_example_policies/ros/RunRuleTransport.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_RULE_TRANSPORT_TEMPLATE:-}" ]]; then
 cp "$AIC_RULE_TRANSPORT_TEMPLATE" "$audit/rule_transport_template.json"
 sha256sum "$AIC_RULE_TRANSPORT_TEMPLATE" >> "$audit/inputs_sha256.txt"
fi
if [[ -f "$audit/moveit_plan.json" ]]; then
 sha256sum "$audit/moveit_plan.json" \
  "$repo/aic_example_policies/aic_example_policies/ros/RunTemporalThenMoveItDiagnostic.py" \
  >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_TEMPORAL_TRANSPORT_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_TEMPORAL_TRANSPORT_CHECKPOINT" "$AIC_TEMPORAL_TRANSPORT_IMAGENET" \
  "$repo/aic_example_policies/aic_example_policies/ros/RunTemporalTransport.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_ROUTE_DIFFUSION_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_ROUTE_DIFFUSION_CHECKPOINT" \
  "$repo/scripts/train_route_aware_diffusion_transport.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_TEMPORAL_HANDOFF_PRIOR_JSON:-}" ]]; then
 sha256sum "$AIC_TEMPORAL_HANDOFF_PRIOR_JSON" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_TEMPORAL_TEACHER_PREFIX_FILE:-}" ]]; then
 sha256sum "$AIC_TEMPORAL_TEACHER_PREFIX_FILE" \
  "$repo/aic_example_policies/aic_example_policies/ros/RunTemporalTeacherPrefixDiagnostic.py" \
  >> "$audit/inputs_sha256.txt"
fi
if [[ "${AIC_TRANSPORT_EVAL_POLICY:-}" == *RunTemporalThenCheatCodeDiagnostic ]]; then
 sha256sum "$repo/aic_example_policies/aic_example_policies/ros/RunTemporalThenCheatCodeDiagnostic.py" \
  "$repo/aic_example_policies/aic_example_policies/ros/CheatCode.py" \
  >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SC_GOAL_PROBE_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_SC_GOAL_PROBE_CHECKPOINT" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SC_VISUAL_STANDOFF_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_SC_VISUAL_STANDOFF_CHECKPOINT" "$AIC_SC_VISUAL_STANDOFF_CALIBRATION" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_JOINT_NATIVE_REFINER_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_JOINT_NATIVE_REFINER_CHECKPOINT" "$repo/scripts/train_joint_native_opening_refiner.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT" "$repo/scripts/native_candidate_association.py" \
  "$repo/scripts/train_native_candidate_history_association.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT:-}" ]]; then
 sha256sum "$AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT" \
  "$repo/scripts/sc_requested_history_observation.py" \
  "$repo/scripts/train_sc_requested_history_locator.py" >> "$audit/inputs_sha256.txt"
fi
if [[ -n "${AIC_SFP_RULE_PRIORS:-}" ]]; then
 sha256sum "$AIC_SFP_RULE_PRIORS" "$AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT" \
  "$AIC_SFP_RULE_MOUNT_PRIORS" >> "$audit/inputs_sha256.txt"
 if [[ -n "${AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT:-}" ]]; then
  sha256sum "$AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT" "$AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT" >> "$audit/inputs_sha256.txt"
 else
  sha256sum "$AIC_SFP_RULE_BOARD_CHECKPOINT" "$AIC_SFP_RULE_CROP_CHECKPOINT" >> "$audit/inputs_sha256.txt"
 fi
fi
if [[ "${AIC_SC_SHADOW_HANDOFF:-0}" == 1 ]]; then
 sha256sum "$AIC_SC_SHADOW_LOCATOR" "$AIC_SC_SHADOW_IDENTITY" \
  "$AIC_SC_SHADOW_ORIENTATION" "$AIC_SC_SHADOW_REFINER" \
  "$AIC_SC_SHADOW_CALIBRATION" "$AIC_SC_SHADOW_BOARD_MOUNT" \
  "$AIC_SC_SHADOW_TCP_PRIOR" "$repo/scripts/sc_shadow_handoff_verifier.py" \
  "$repo/scripts/evaluate_sc_board_lines_yaw.py" >> "$audit/inputs_sha256.txt"
 if [[ -n "${AIC_SC_SHADOW_CROP_IDENTITY:-}" ]]; then
  sha256sum "$AIC_SC_SHADOW_CROP_IDENTITY" +   "$repo/scripts/train_sc_observation_crop_identity.py" >> "$audit/inputs_sha256.txt"
 fi
fi
cleanup() { docker stop "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$name" --gpus "$docker_gpu_devices" --shm-size=4g --network bridge --entrypoint /bin/bash \
 -e GZ_PARTITION="$name" \
 -e NVIDIA_VISIBLE_DEVICES="$docker_visible_devices" -e NVIDIA_DRIVER_CAPABILITIES=all \
 -e AIC_ENGINE_CUDA_VISIBLE_DEVICE="$engine_cuda_visible_device" \
 -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e ROS_DOMAIN_ID="$eval_domain" -e AIC_EVAL_ROS_DOMAIN="$eval_domain" \
 -e AIC_RESULTS_DIR=/audit/results -e AIC_SHARED_PORT_TCP_CHECKPOINT="$checkpoint" \
 -e AIC_TRANSPORT_EVAL_AUDIT=/audit \
 -e AIC_AUDIT_CAPTURE_HZ="${AIC_AUDIT_CAPTURE_HZ:-0}" \
 -e AIC_TRANSPORT_EVAL_CONTROL_HZ="${AIC_TRANSPORT_EVAL_CONTROL_HZ:-20}" \
 -e AIC_TEMPORAL_TRANSPORT_CHECKPOINT="${AIC_TEMPORAL_TRANSPORT_CHECKPOINT:-}" \
 -e AIC_TEMPORAL_CPU_THREADS="${AIC_TEMPORAL_CPU_THREADS:-8}" \
 -e AIC_DELAY_TF_LISTENER="${AIC_DELAY_TF_LISTENER:-0}" \
 -e AIC_ROUTE_DIFFUSION_CHECKPOINT="${AIC_ROUTE_DIFFUSION_CHECKPOINT:-}" \
 -e AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY="${AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY:-}" \
 -e AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS="${AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS:-0}" \
 -e AIC_ROUTE_DIFFUSION_PREWARM="${AIC_ROUTE_DIFFUSION_PREWARM:-0}" \
 -e AIC_ROUTE_DIFFUSION_SAMPLE_STEPS="${AIC_ROUTE_DIFFUSION_SAMPLE_STEPS:-}" \
 -e AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK="${AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK:-0}" \
 -e AIC_TEMPORAL_TRANSPORT_IMAGENET="${AIC_TEMPORAL_TRANSPORT_IMAGENET:-}" \
 -e AIC_TEMPORAL_TEACHER_PREFIX_FILE="${AIC_TEMPORAL_TEACHER_PREFIX_FILE:-}" \
 -e AIC_SFP_CORRECTIVE_ACTOR_DURATION_S="${AIC_SFP_CORRECTIVE_ACTOR_DURATION_S:-2}" \
 -e AIC_SFP_CORRECTIVE_TEACHER_DURATION_S="${AIC_SFP_CORRECTIVE_TEACHER_DURATION_S:-30}" \
 -e AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT="${AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT:-0}" \
 -e AIC_SFP_MOVEIT_PLAN_JSON="${AIC_SFP_MOVEIT_PLAN_SOURCE:+/audit/moveit_plan.json}" \
 -e AIC_SFP_MOVEIT_VARIANT="${AIC_SFP_MOVEIT_VARIANT:-retreat_align_approach}" \
 -e AIC_SFP_MOVEIT_LIVE_REPLAN="${AIC_SFP_MOVEIT_LIVE_REPLAN:-0}" \
 -e AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD="${AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD:-}" \
 -e AIC_TEMPORAL_HANDOFF_PRIOR_JSON="${AIC_TEMPORAL_HANDOFF_PRIOR_JSON:-}" \
 -e AIC_TEMPORAL_TRANSPORT_STOP_STREAK="${AIC_TEMPORAL_TRANSPORT_STOP_STREAK:-3}" \
 -e AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S="${AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S:-0}" \
 -e AIC_SFP_DIAGNOSTIC_STOP="${AIC_SFP_DIAGNOSTIC_STOP:-0}" \
 -e AIC_SC_GOAL_PROBE_CHECKPOINT="${AIC_SC_GOAL_PROBE_CHECKPOINT:-}" \
 -e AIC_SC_GOAL_SERVO_TRIGGER_M="${AIC_SC_GOAL_SERVO_TRIGGER_M:-.035}" \
 -e AIC_SC_GOAL_SERVO_STEP_M="${AIC_SC_GOAL_SERVO_STEP_M:-.008}" \
 -e AIC_SC_GOAL_SERVO_STOP_M="${AIC_SC_GOAL_SERVO_STOP_M:-.006}" \
 -e AIC_SC_GOAL_STOP_ONLY="${AIC_SC_GOAL_STOP_ONLY:-0}" \
 -e AIC_SC_GOAL_STOP_ONLY_THRESHOLD_M="${AIC_SC_GOAL_STOP_ONLY_THRESHOLD_M:-.030}" \
 -e AIC_SC_GOAL_STOP_ONLY_STREAK="${AIC_SC_GOAL_STOP_ONLY_STREAK:-3}" \
 -e AIC_SC_VISUAL_STANDOFF_CHECKPOINT="${AIC_SC_VISUAL_STANDOFF_CHECKPOINT:-}" \
 -e AIC_SC_VISUAL_STOP_MODE="${AIC_SC_VISUAL_STOP_MODE:-visual}" \
 -e AIC_SC_VISUAL_STANDOFF_CALIBRATION="${AIC_SC_VISUAL_STANDOFF_CALIBRATION:-}" \
 -e AIC_SC_VISUAL_STANDOFF_TRIGGER_M="${AIC_SC_VISUAL_STANDOFF_TRIGGER_M:-.060}" \
 -e AIC_SC_VISUAL_STANDOFF_STEP_M="${AIC_SC_VISUAL_STANDOFF_STEP_M:-.008}" \
 -e AIC_SC_VISUAL_STANDOFF_STOP_M="${AIC_SC_VISUAL_STANDOFF_STOP_M:-.008}" \
 -e AIC_SC_VISUAL_STANDOFF_STIFFNESS="${AIC_SC_VISUAL_STANDOFF_STIFFNESS:-0}" \
 -e AIC_SC_VISUAL_STANDOFF_DAMPING="${AIC_SC_VISUAL_STANDOFF_DAMPING:-80}" \
 -e AIC_SC_VISUAL_STANDOFF_ORIENTATION="${AIC_SC_VISUAL_STANDOFF_ORIENTATION:-0}" \
 -e AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT="${AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT:-}" \
 -e AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND="${AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND:-1}" \
 -e AIC_SC_TRANSPORT_Z_FLOOR_M="${AIC_SC_TRANSPORT_Z_FLOOR_M:-0}" \
 -e AIC_SC_DIRECT_NATIVE_RGB="${AIC_SC_DIRECT_NATIVE_RGB:-0}" \
 -e AIC_SC_SHADOW_HANDOFF="${AIC_SC_SHADOW_HANDOFF:-0}" \
 -e AIC_SC_SHADOW_HISTORY_AUDIT="${AIC_SC_SHADOW_HISTORY_AUDIT:-0}" \
 -e AIC_SC_SHADOW_USE_HISTORY_ORIENTATION="${AIC_SC_SHADOW_USE_HISTORY_ORIENTATION:-0}" \
 -e AIC_SC_SHADOW_NATIVE_SNAPSHOT_DIR="${AIC_SC_SHADOW_NATIVE_SNAPSHOT_DIR:-}" \
 -e AIC_SC_SHADOW_DIRECT_NATIVE_RGB="${AIC_SC_SHADOW_DIRECT_NATIVE_RGB:-0}" \
 -e AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST="${AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST:-0}" \
 -e AIC_SC_SHADOW_IDENTITY_RETRY="${AIC_SC_SHADOW_IDENTITY_RETRY:-0}" \
 -e AIC_SC_SHADOW_ALLOW_STALE_PAUSE="${AIC_SC_SHADOW_ALLOW_STALE_PAUSE:-0}" \
 -e AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC="${AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC:-70}" \
 -e AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC="${AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC:-180}" \
 -e AIC_SC_SHADOW_LOCATOR="${AIC_SC_SHADOW_LOCATOR:-}" \
 -e AIC_SC_SHADOW_IDENTITY="${AIC_SC_SHADOW_IDENTITY:-}" \
 -e AIC_SC_SHADOW_CROP_IDENTITY="${AIC_SC_SHADOW_CROP_IDENTITY:-}" \
 -e AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS="${AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS:-0}" \
 -e AIC_SC_SHADOW_ORIENTATION="${AIC_SC_SHADOW_ORIENTATION:-}" \
 -e AIC_SC_SHADOW_REFINER="${AIC_SC_SHADOW_REFINER:-}" \
 -e AIC_JOINT_NATIVE_REFINER_CHECKPOINT="${AIC_JOINT_NATIVE_REFINER_CHECKPOINT:-}" \
 -e AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT="${AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT:-}" \
 -e AIC_SC_CANDIDATE_GATE_MODE="${AIC_SC_CANDIDATE_GATE_MODE:-intersection}" \
 -e AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT="${AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT:-}" \
 -e AIC_SFP_RULE_PRIORS="${AIC_SFP_RULE_PRIORS:-}" \
 -e AIC_SFP_RULE_BOARD_CHECKPOINT="${AIC_SFP_RULE_BOARD_CHECKPOINT:-}" \
 -e AIC_SFP_RULE_CROP_CHECKPOINT="${AIC_SFP_RULE_CROP_CHECKPOINT:-}" \
 -e AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT="${AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT:-}" \
 -e AIC_SFP_RULE_MOUNT_PRIORS="${AIC_SFP_RULE_MOUNT_PRIORS:-}" \
 -e AIC_SFP_WIDE_APPROACH_JSON="${AIC_SFP_WIDE_APPROACH_JSON:-}" \
 -e AIC_SC_NATIVE_POSE_MAX_REPROJECTION_PX="${AIC_SC_NATIVE_POSE_MAX_REPROJECTION_PX:-2}" \
 -e AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT="${AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT:-}" \
 -e AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT="${AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT:-}" \
 -e AIC_SC_SHADOW_CALIBRATION="${AIC_SC_SHADOW_CALIBRATION:-}" \
 -e AIC_SC_SHADOW_BOARD_MOUNT="${AIC_SC_SHADOW_BOARD_MOUNT:-}" \
 -e AIC_SC_SHADOW_TCP_PRIOR="${AIC_SC_SHADOW_TCP_PRIOR:-}" \
 -e AIC_TRANSPORT_EVAL_POLICY="${AIC_TRANSPORT_EVAL_POLICY:-aic_example_policies.ros.RunSharedPortTCP}" \
 -e AIC_TRANSPORT_EVAL_TEACHER_PREFIX="${AIC_TRANSPORT_EVAL_TEACHER_PREFIX:-false}" \
 -e AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT="${AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT:-}" \
 -e AIC_ORACLE_GOAL_DEPTH_M="${AIC_ORACLE_GOAL_DEPTH_M:-.01564}" \
 -e AIC_ORACLE_CAUSAL_TF_CALIBRATION="${AIC_ORACLE_CAUSAL_TF_CALIBRATION:-0}" \
 -e AIC_ORACLE_ALIGNMENT_LATERAL_M="${AIC_ORACLE_ALIGNMENT_LATERAL_M:-.0005}" \
 -e AIC_ORACLE_INTEGRAL_LIMIT_M="${AIC_ORACLE_INTEGRAL_LIMIT_M:-.05}" \
 -e AIC_ORACLE_ALIGNMENT_GATE="${AIC_ORACLE_ALIGNMENT_GATE:-0}" \
 -e AIC_ORACLE_STOP_ON_PHYSICAL_GATE="${AIC_ORACLE_STOP_ON_PHYSICAL_GATE:-0}" \
 -e AIC_ORACLE_AXIAL_RATE_MPS="${AIC_ORACLE_AXIAL_RATE_MPS:-.001}" \
 -e AIC_ORACLE_LATERAL_INTEGRAL="${AIC_ORACLE_LATERAL_INTEGRAL:-0}" \
 -e AIC_ORACLE_MATING_PRIOR="${AIC_ORACLE_MATING_PRIOR:-}" \
 -e AIC_LOCKED_GRASP_CHECKPOINT="${AIC_LOCKED_GRASP_CHECKPOINT:-}" \
 -e AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M="${AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M:-.03}" \
 -e AIC_SFP_LOCAL_HANDOFF_HOLD_SEC="${AIC_SFP_LOCAL_HANDOFF_HOLD_SEC:-.5}" \
 -e AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M="${AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M:-}" \
 -e AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M="${AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M:-}" \
 -e AIC_SFP_POST_HANDOFF_TEACHER_DIAGNOSTIC="${AIC_SFP_POST_HANDOFF_TEACHER_DIAGNOSTIC:-0}" \
 -e AIC_SFP_POST_HANDOFF_STAGE_PRESERVING_TEACHER="${AIC_SFP_POST_HANDOFF_STAGE_PRESERVING_TEACHER:-0}" \
 -e AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL="${AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL:-0}" \
 -e AIC_SFP_LOCAL_DECISION_LIMIT="${AIC_SFP_LOCAL_DECISION_LIMIT:-0}" \
 -e AIC_NEAR_PERTURB_SIGN="${AIC_NEAR_PERTURB_SIGN:-1}" \
 -e AIC_NEAR_PERTURB_LATERAL_M="${AIC_NEAR_PERTURB_LATERAL_M:-.003}" \
 -e AIC_NEAR_PERTURB_YAW_DEG="${AIC_NEAR_PERTURB_YAW_DEG:-4}" \
 -e AIC_NEAR_PERTURB_HOLD_S="${AIC_NEAR_PERTURB_HOLD_S:-.75}" \
 -e AIC_LOCAL_LOCATOR_CHECKPOINT="${AIC_LOCAL_LOCATOR_CHECKPOINT:-}" \
 -e AIC_LOCAL_ORIENTATION_CHECKPOINT="${AIC_LOCAL_ORIENTATION_CHECKPOINT:-}" \
 -e AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT="${AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT:-}" \
 -e AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT_2="${AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT_2:-}" \
 -e AIC_LOCAL_SFP_PAIR_ORIENTATION="${AIC_LOCAL_SFP_PAIR_ORIENTATION:-0}" \
 -e AIC_LOCAL_SFP_LOCK_SAMPLES="${AIC_LOCAL_SFP_LOCK_SAMPLES:-5}" \
 -e AIC_LOCAL_SFP_COARSE_CHECKPOINT="${AIC_LOCAL_SFP_COARSE_CHECKPOINT:-}" \
 -e AIC_LOCAL_SFP_BROAD_CHECKPOINT="${AIC_LOCAL_SFP_BROAD_CHECKPOINT:-}" \
 -e AIC_LOCAL_SFP_FINE_CHECKPOINT="${AIC_LOCAL_SFP_FINE_CHECKPOINT:-}" \
 -e AIC_LOCAL_CAMERA_CALIBRATION="${AIC_LOCAL_CAMERA_CALIBRATION:-}" \
 -e AIC_LOCAL_ACTOR_DEVICE="${AIC_LOCAL_ACTOR_DEVICE:-cpu}" \
 -e AIC_POLICY_CUDA_DEVICE="$policy_cuda_device" \
 -e AIC_TEACHER_REPLAY_SOURCE="${AIC_TEACHER_REPLAY_SOURCE:-}" \
 -e AIC_TEACHER_REPLAY_CAP="${AIC_TEACHER_REPLAY_CAP:-}" \
 -e AIC_RULE_TRANSPORT_TEMPLATE="${AIC_RULE_TRANSPORT_TEMPLATE:-}" \
 -e AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA="$AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA" \
 -e AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS="${AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS:-[90,90,90,50,50,50]}" \
 -e AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING="${AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING:-[50,50,50,20,20,20]}" \
 -v "$audit:/audit" -v "$repo:$repo:ro" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act:ro \
 -v /var/tmp/aic_sc_fix_20260924:/var/tmp/aic_sc_fix_20260924:ro \
 -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives:ro \
 -w "$repo" "$image" -lc 'sleep infinity' > "$audit/container.id"
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export CUDA_VISIBLE_DEVICES="$AIC_ENGINE_CUDA_VISIBLE_DEVICE"
 export ROS_DOMAIN_ID="$AIC_EVAL_ROS_DOMAIN" RMW_IMPLEMENTATION=rmw_zenoh_cpp
 ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id aic_world \
   --ros-args -p use_sim_time:=true > /audit/world_frame.log 2>&1 &
 description_args=()
 if test -f /audit/ur_llb_start.xacro; then
  description_args+=(description_file:=/audit/ur_llb_start.xacro)
 fi
 /entrypoint.sh ground_truth:=${AIC_TRANSPORT_EVAL_TEACHER_PREFIX:-false} start_aic_engine:=true headless:=true \
  aic_engine_config_file:=/audit/eval_config.yaml world_file:=/audit/world_audit.sdf \
  ros_gz_bridge_config_file:=/audit/bridge_audit.yaml "${description_args[@]}" > /audit/engine.log 2>&1
 printf "%s\n" "$?" > /audit/engine.exit
'
for _ in $(seq 1 90); do
 if test -f "$audit/engine.exit"; then break; fi
 if docker exec "$name" bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID="$AIC_EVAL_ROS_DOMAIN"
  timeout 2 ros2 node list 2>/dev/null | grep -q aic_engine
 '; then break; fi
 sleep 2
done
test ! -f "$audit/engine.exit" || { echo 'Engine exited early' >&2; exit 1; }
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID="$AIC_EVAL_ROS_DOMAIN"
 python3 /audit/capture_observations.py > /audit/capture.log 2>&1
'
if [[ "${AIC_TRANSPORT_EVAL_ORACLE_CONTACTS:-false}" == true ]]; then
  docker exec -d "$name" bash -lc '
   source /ws_aic/install/setup.bash
   export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID="$AIC_EVAL_ROS_DOMAIN"
   python3 "$PWD/scripts/capture_oracle_contacts.py" > /audit/oracle_capture.log 2>&1
  '
fi
docker exec -d "$name" bash -lc '
 source /ws_aic/install/setup.bash
 export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID="$AIC_EVAL_ROS_DOMAIN"
 export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
 export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
 export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
 export PYTHONPATH="$PWD/.pixi/envs/default/lib/python3.12/site-packages:$PWD/aic_model:$PWD/aic_example_policies:$PWD/scripts/pythonpath_bootstrap:${PYTHONPATH:-}"
 export AIC_CHECKOUT_PYTHONPATH="$PWD/aic_model:$PWD/aic_example_policies"
 export PYTHONPATH="$PWD/aic_utils/lerobot_robot_aic:$PYTHONPATH"
 export AIC_SHARED_PORT_TCP_DEVICE="$AIC_POLICY_CUDA_DEVICE" AIC_SHARED_PORT_TCP_CONTROL_CLOCK=simulation
 export AIC_SHARED_PORT_TCP_CONTROL_HZ="${AIC_TRANSPORT_EVAL_CONTROL_HZ:-20}" AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC="${AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC:-70}"
 export AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC="$AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC" AIC_SHARED_PORT_TCP_TRACE=/audit/runtime_trace.jsonl
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
