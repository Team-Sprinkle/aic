#!/usr/bin/env bash
set -euo pipefail

# Run inside the isaac-lab-base container from /workspace/isaaclab.
# Actor-only online RL with the stateful insertion reward. No expert guide,
# insertion guard, or hard-coded rollout policy override is enabled.

cd /workspace/isaaclab

CONFIG_PATH="${AIC_STATEFUL_CONFIG:-aic/configs/stateful_insertion_curriculum.yaml}"
RUN_ROOT="${AIC_STATEFUL_RUN_ROOT:-/tmp/aic_stateful_promotion_gated_3x0_to_40x10_2000eps_$(date -u +%Y%m%d_%H%M%S)}"
ACT_TS="${AIC_STATEFUL_ACT_TS:-aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt}"
BASE_CKPT="${AIC_STATEFUL_BASE_CKPT:-/tmp/aic_v1367_clean_v1077_v1254_true40x10_4h_segments_clip1mm/2026-06-05_03-42-24_2026-06-04_train_v1367_clean_v1077_v1254_clip1mm_constant40x10_seg08/checkpoint_latest.pt}"
if [[ ! -f "$BASE_CKPT" ]]; then
  BASE_CKPT="aic/outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-06-02_train_v1077_gentle_offline_true40x10/2026-06-02_23-44-40_isaac_online_serl/checkpoint_latest.pt"
fi

SEGMENT_SECONDS="${AIC_STATEFUL_SEGMENT_SECONDS:-1800}"
TRAIN_STEPS_LIMIT="${AIC_STATEFUL_TRAIN_STEPS_LIMIT:-100000000}"
EVAL_STEPS="${AIC_STATEFUL_EVAL_STEPS:-900}"
TRAIN_EPISODE_VARIANTS="${AIC_STATEFUL_TRAIN_EPISODE_VARIANTS:-64}"
EVAL_EPISODE_VARIANTS="${AIC_STATEFUL_EVAL_EPISODE_VARIANTS:-16}"
NUM_ENVS="${AIC_STATEFUL_NUM_ENVS:-4}"
EVAL_NUM_ENVS="${AIC_STATEFUL_EVAL_NUM_ENVS:-1}"
SUCCESS_AXIAL_THRESHOLD="${AIC_STATEFUL_SUCCESS_AXIAL_THRESHOLD:-0.0005}"
SUCCESS_LATERAL_THRESHOLD="${AIC_STATEFUL_SUCCESS_LATERAL_THRESHOLD:-0.0005}"
SUCCESS_ORIENTATION_THRESHOLD="${AIC_STATEFUL_SUCCESS_ORIENTATION_THRESHOLD:-0.04}"
GRADIENT_UPDATES_PER_STEP="${AIC_STATEFUL_GRADIENT_UPDATES_PER_STEP:-1}"
BATCH_SIZE="${AIC_STATEFUL_BATCH_SIZE:-32}"
WARMUP_STEPS="${AIC_STATEFUL_WARMUP_STEPS:-1}"
ACTOR_UPDATE_START_STEPS="${AIC_STATEFUL_ACTOR_UPDATE_START_STEPS:-1}"
TRAIN_SEED_BASE="${AIC_STATEFUL_TRAIN_SEED_BASE:-21600}"
EVAL_SEED_BASE="${AIC_STATEFUL_EVAL_SEED_BASE:-22600}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${AIC_STATEFUL_CUDA_VISIBLE_DEVICES:-0}"
ISAAC_DEVICE="${AIC_STATEFUL_DEVICE:-cuda:0}"
NO_PROGRESS_ASSESS_SECONDS="${AIC_STATEFUL_NO_PROGRESS_ASSESS_SECONDS:-7200}"
DRY_RUN="${AIC_STATEFUL_DRY_RUN:-0}"
START_CYCLE="${AIC_STATEFUL_START_CYCLE:-0}"
START_LEVEL="${AIC_STATEFUL_START_LEVEL:-0}"
START_EPISODES_USED="${AIC_STATEFUL_START_EPISODES_USED:-0}"
FORCE_DELTA_PENALTY_WEIGHT="${AIC_STATEFUL_FORCE_DELTA_PENALTY_WEIGHT:-0.0}"
NEAR_GATE_RESET_MAX_ITERATIONS="${AIC_STATEFUL_NEAR_GATE_RESET_MAX_ITERATIONS:-8}"
RESET_SETTLE_STEPS="${AIC_STATEFUL_RESET_SETTLE_STEPS:-0}"
STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_WEIGHT="${AIC_STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_WEIGHT:-24.0}"
STATEFUL_AXIAL_LATERAL_ACTION_SCALE="${AIC_STATEFUL_AXIAL_LATERAL_ACTION_SCALE:-0.00010}"
STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_MAX="${AIC_STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_MAX:-50.0}"
STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_WEIGHT="${AIC_STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_WEIGHT:-12.0}"
STATEFUL_AXIAL_ALIGNMENT_LOSS_LATERAL_SCALE="${AIC_STATEFUL_AXIAL_ALIGNMENT_LOSS_LATERAL_SCALE:-0.0005}"
STATEFUL_AXIAL_ALIGNMENT_LOSS_ORIENTATION_SCALE="${AIC_STATEFUL_AXIAL_ALIGNMENT_LOSS_ORIENTATION_SCALE:-0.020}"
STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_MAX="${AIC_STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_MAX:-16.0}"
STATEFUL_AXIAL_PURE_ACTION_WEIGHT="${AIC_STATEFUL_AXIAL_PURE_ACTION_WEIGHT:-16.0}"
STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_WEIGHT="${AIC_STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_WEIGHT:-32.0}"
STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_MAX="${AIC_STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_MAX:-20.0}"
STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_WEIGHT="${AIC_STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_WEIGHT:-32.0}"
STATEFUL_AXIAL_ROTATION_ACTION_SCALE="${AIC_STATEFUL_AXIAL_ROTATION_ACTION_SCALE:-0.00010}"
STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_MAX="${AIC_STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_MAX:-50.0}"
STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_WEIGHT="${AIC_STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_WEIGHT:-16.0}"
STATEFUL_AXIAL_FORWARD_ACTION_SCALE="${AIC_STATEFUL_AXIAL_FORWARD_ACTION_SCALE:-0.00005}"
STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_MAX="${AIC_STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_MAX:-8.0}"
STATEFUL_LATERAL_PROGRESS_WEIGHT="${AIC_STATEFUL_LATERAL_PROGRESS_WEIGHT:-8.0}"
STATEFUL_ORIENTATION_PROGRESS_WEIGHT="${AIC_STATEFUL_ORIENTATION_PROGRESS_WEIGHT:-8.0}"
STATEFUL_NEAR_MISALIGNED_WEIGHT="${AIC_STATEFUL_NEAR_MISALIGNED_WEIGHT:-0.5}"
STATEFUL_LATERAL_ALIGNMENT_ACTION_WEIGHT="${AIC_STATEFUL_LATERAL_ALIGNMENT_ACTION_WEIGHT:-6.0}"
STATEFUL_OFF_AXIS_AXIAL_ACTION_PENALTY_WEIGHT="${AIC_STATEFUL_OFF_AXIS_AXIAL_ACTION_PENALTY_WEIGHT:-16.0}"
STATEFUL_RETREAT_WEIGHT="${AIC_STATEFUL_RETREAT_WEIGHT:-0.25}"
STATEFUL_ACTION_FORWARD_SCALE="${AIC_STATEFUL_ACTION_FORWARD_SCALE:-0.00005}"
STATEFUL_ACTION_MIN_FORWARD="${AIC_STATEFUL_ACTION_MIN_FORWARD:-0.0}"
STATEFUL_ACTION_LATERAL_SIGMA="${AIC_STATEFUL_ACTION_LATERAL_SIGMA:-0.00005}"
STATEFUL_ACTION_LATERAL_SIGMA_FAR="${AIC_STATEFUL_ACTION_LATERAL_SIGMA_FAR:-0.00030}"
STATEFUL_ACTION_RADIUS_SCHEDULE_FAR_DEPTH="${AIC_STATEFUL_ACTION_RADIUS_SCHEDULE_FAR_DEPTH:--0.020}"
STATEFUL_ACTION_RADIUS_SCHEDULE_NEAR_DEPTH="${AIC_STATEFUL_ACTION_RADIUS_SCHEDULE_NEAR_DEPTH:-0.0}"
STATEFUL_POLICY_TCP_DELTA_SIGN="${AIC_STATEFUL_POLICY_TCP_DELTA_SIGN:-1.0}"
STATEFUL_ORIENTATION_ENTER_THRESHOLD="${AIC_STATEFUL_ORIENTATION_ENTER_THRESHOLD:-0.040}"
STATEFUL_SIGMA_THETA_INSERT="${AIC_STATEFUL_SIGMA_THETA_INSERT:-0.060}"
STATEFUL_SIGMA_THETA_INSERT_FAR="${AIC_STATEFUL_SIGMA_THETA_INSERT_FAR:-0.100}"
ADAPTER_LR="${AIC_STATEFUL_ADAPTER_LR:-1e-5}"
CRITIC_LR="${AIC_STATEFUL_CRITIC_LR:-1e-5}"
ACTOR_Q_WEIGHT="${AIC_STATEFUL_ACTOR_Q_WEIGHT:-3e-4}"
ACTOR_AXIAL_PURITY_WEIGHT="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_WEIGHT:-0.0}"
ACTOR_AXIAL_PURITY_LATERAL_WEIGHT="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_LATERAL_WEIGHT:-1.0}"
ACTOR_AXIAL_PURITY_ROTATION_WEIGHT="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_ROTATION_WEIGHT:-1.0}"
ACTOR_AXIAL_PURITY_FORWARD_WEIGHT="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_FORWARD_WEIGHT:-1.0}"
ACTOR_AXIAL_PURITY_BACKWARD_WEIGHT="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_BACKWARD_WEIGHT:-2.0}"
ACTOR_AXIAL_PURITY_LATERAL_SCALE="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_LATERAL_SCALE:-0.00002}"
ACTOR_AXIAL_PURITY_ROTATION_SCALE="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_ROTATION_SCALE:-0.00001}"
ACTOR_AXIAL_PURITY_FORWARD_SCALE="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_FORWARD_SCALE:-0.000005}"
ACTOR_AXIAL_PURITY_LATERAL_GATE="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_LATERAL_GATE:-0.0007}"
ACTOR_AXIAL_PURITY_ORIENTATION_GATE="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_ORIENTATION_GATE:-0.030}"
ACTOR_AXIAL_PURITY_MIN_S="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_MIN_S:--0.010}"
ACTOR_AXIAL_PURITY_MAX_S="${AIC_STATEFUL_ACTOR_AXIAL_PURITY_MAX_S:-0.100}"
ACTOR_AXIAL_DIRECTION_SIGN="${AIC_STATEFUL_ACTOR_AXIAL_DIRECTION_SIGN:-1.0}"
ACTOR_ORIENTATION_DIRECTION_SIGN="${AIC_STATEFUL_ACTOR_ORIENTATION_DIRECTION_SIGN:-1.0}"
ACTOR_INITIAL_AXIAL_BIAS_M="${AIC_STATEFUL_ACTOR_INITIAL_AXIAL_BIAS_M:-0.0}"
ADAPTER_PENALTY_WEIGHT="${AIC_STATEFUL_ADAPTER_PENALTY_WEIGHT:-1e-5}"
ACT_PRESERVATION_WEIGHT="${AIC_STATEFUL_ACT_PRESERVATION_WEIGHT:-1e-5}"
ACTOR_MODE="${AIC_STATEFUL_ACTOR_MODE:-act_adapter}"
RESET_ACTOR_HEAD="${AIC_STATEFUL_RESET_ACTOR_HEAD:-0}"
ADAPTER_DELTA_CLIP="${AIC_STATEFUL_ADAPTER_DELTA_CLIP:-0.05}"
ACTOR_EXPLORATION_NOISE_STD="${AIC_STATEFUL_ACTOR_EXPLORATION_NOISE_STD:-0.0}"
ACTOR_EXPLORATION_NOISE_STEPS="${AIC_STATEFUL_ACTOR_EXPLORATION_NOISE_STEPS:-0}"
ACTOR_EXPLORATION_NOISE_MODE="${AIC_STATEFUL_ACTOR_EXPLORATION_NOISE_MODE:-isotropic}"
ACTOR_EXPLORATION_PHASE_LATERAL_MULTIPLIER="${AIC_STATEFUL_ACTOR_EXPLORATION_PHASE_LATERAL_MULTIPLIER:-1.0}"
ACTOR_EXPLORATION_PHASE_ORIENTATION_STD="${AIC_STATEFUL_ACTOR_EXPLORATION_PHASE_ORIENTATION_STD:-0.0}"
TCP_TRANSLATION_ACTION_CLIP="${AIC_STATEFUL_TCP_TRANSLATION_ACTION_CLIP:-0.0}"
TCP_ROTATION_ACTION_CLIP="${AIC_STATEFUL_TCP_ROTATION_ACTION_CLIP:-0.0}"
ABSOLUTE_IK_TARGET_POSE="${AIC_STATEFUL_ABSOLUTE_IK_TARGET_POSE:-0}"
SFP_SHRUNK_BOX_MARGIN_X="${AIC_STATEFUL_SFP_SHRUNK_BOX_MARGIN_X:-0.00030}"
SFP_SHRUNK_BOX_MARGIN_Y="${AIC_STATEFUL_SFP_SHRUNK_BOX_MARGIN_Y:-0.0}"
SFP_SHRUNK_BOX_MARGIN_Z="${AIC_STATEFUL_SFP_SHRUNK_BOX_MARGIN_Z:-0.00030}"
EXIT_AFTER_FIRST_TRAIN="${AIC_STATEFUL_EXIT_AFTER_FIRST_TRAIN:-0}"
EXTRA_TRAIN_FLAGS=()
if [[ -n "${AIC_STATEFUL_EXTRA_TRAIN_FLAGS:-}" ]]; then
  # Intended for simple diagnostic flag passthroughs without spaces inside an
  # individual argument, e.g. "--debug_audit_steps 20 --debug_diagnostics".
  read -r -a EXTRA_TRAIN_FLAGS <<<"${AIC_STATEFUL_EXTRA_TRAIN_FLAGS}"
fi
EXTRA_EVAL_FLAGS=()
if [[ -n "${AIC_STATEFUL_EXTRA_EVAL_FLAGS:-}" ]]; then
  read -r -a EXTRA_EVAL_FLAGS <<<"${AIC_STATEFUL_EXTRA_EVAL_FLAGS}"
fi

mkdir -p "$RUN_ROOT"/{summaries,levels}
cp "$CONFIG_PATH" "$RUN_ROOT/high_level_config.yaml"

read -r TOTAL_EPISODES EVAL_EVERY LEVEL_COUNT POLICY <<<"$(
  /workspace/isaaclab/_isaac_sim/python.sh - "$CONFIG_PATH" <<'PY'
import math, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
episodes = int(cfg.get("episodes", 2000))
progression = cfg.get("progression") or {}
eval_every = int(progression.get("eval_every_episodes", 10))
policy = str(progression.get("policy", "promotion_gated"))
level_count = int(progression.get("level_count", max(1, math.ceil(episodes / max(1, eval_every)))))
print(episodes, eval_every, max(1, level_count), policy)
PY
)"
if [[ "$POLICY" != "promotion_gated" ]]; then
  echo "[AIC stateful] config progression.policy must be promotion_gated, got: $POLICY" >&2
  exit 2
fi

CONFIG_ACTION_MIN_FORWARD="$(
  /workspace/isaaclab/_isaac_sim/python.sh - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
value = (cfg.get("reward") or {}).get("action_min_forward_m")
print("" if value is None else float(value))
PY
)"
if [[ -z "${AIC_STATEFUL_ACTION_MIN_FORWARD+x}" && -n "$CONFIG_ACTION_MIN_FORWARD" ]]; then
  STATEFUL_ACTION_MIN_FORWARD="$CONFIG_ACTION_MIN_FORWARD"
fi

COMMON_FLAGS=(
  --task AIC-Task-v0
  --headless
  --rendering_mode performance
  --act_torchscript "$ACT_TS"
  --act_only_actor_mode "$ACTOR_MODE"
  --policy_hz 20
  --enable_contact_sensor
  --disable_ppo_resnet_observation_terms
  --fix_isaac_ik_xy_sign
  --no-treat_time_limit_truncation_as_terminal
  --gripper_joint_position 0.0035405
  --critic_image_encoder_override small_conv
  --target_success_orientation_threshold "$SUCCESS_ORIENTATION_THRESHOLD"
  --target_success_axial_threshold "$SUCCESS_AXIAL_THRESHOLD"
  --target_success_lateral_threshold "$SUCCESS_LATERAL_THRESHOLD"
  --target_reward_orientation_error_mode axis
  --target_reward_orientation_axis_local 0 0 1
  --reward_preset stateful_insertion_v1
  --target_reward_exp_gated_action_axis_gate
  --target_reward_exp_gated_action_axis_source policy_tcp_delta
  --target_reward_exp_gated_phase_gate_insertion_credit
  --target_reward_exp_gated_lateral_alignment_require_axial_quiet
  --target_reward_stateful_axial_lateral_action_penalty_weight "$STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_WEIGHT"
  --target_reward_stateful_axial_lateral_action_scale "$STATEFUL_AXIAL_LATERAL_ACTION_SCALE"
  --target_reward_stateful_axial_lateral_action_penalty_max "$STATEFUL_AXIAL_LATERAL_ACTION_PENALTY_MAX"
  --target_reward_stateful_axial_alignment_loss_penalty_weight "$STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_WEIGHT"
  --target_reward_stateful_axial_alignment_loss_lateral_scale "$STATEFUL_AXIAL_ALIGNMENT_LOSS_LATERAL_SCALE"
  --target_reward_stateful_axial_alignment_loss_orientation_scale "$STATEFUL_AXIAL_ALIGNMENT_LOSS_ORIENTATION_SCALE"
  --target_reward_stateful_axial_alignment_loss_penalty_max "$STATEFUL_AXIAL_ALIGNMENT_LOSS_PENALTY_MAX"
  --target_reward_stateful_axial_pure_action_weight "$STATEFUL_AXIAL_PURE_ACTION_WEIGHT"
  --target_reward_stateful_axial_impure_action_penalty_weight "$STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_WEIGHT"
  --target_reward_stateful_axial_impure_action_penalty_max "$STATEFUL_AXIAL_IMPURE_ACTION_PENALTY_MAX"
  --target_reward_stateful_axial_rotation_action_penalty_weight "$STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_WEIGHT"
  --target_reward_stateful_axial_rotation_action_scale "$STATEFUL_AXIAL_ROTATION_ACTION_SCALE"
  --target_reward_stateful_axial_rotation_action_penalty_max "$STATEFUL_AXIAL_ROTATION_ACTION_PENALTY_MAX"
  --target_reward_stateful_axial_forward_action_penalty_weight "$STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_WEIGHT"
  --target_reward_stateful_axial_forward_action_scale "$STATEFUL_AXIAL_FORWARD_ACTION_SCALE"
  --target_reward_stateful_axial_forward_action_penalty_max "$STATEFUL_AXIAL_FORWARD_ACTION_PENALTY_MAX"
  --target_reward_cheatcode_lateral_progress_weight "$STATEFUL_LATERAL_PROGRESS_WEIGHT"
  --target_reward_cheatcode_orientation_progress_weight "$STATEFUL_ORIENTATION_PROGRESS_WEIGHT"
  --target_reward_cheatcode_near_misaligned_weight "$STATEFUL_NEAR_MISALIGNED_WEIGHT"
  --target_reward_cheatcode_lateral_alignment_action_weight "$STATEFUL_LATERAL_ALIGNMENT_ACTION_WEIGHT"
  --target_reward_cheatcode_off_axis_axial_action_penalty_weight "$STATEFUL_OFF_AXIS_AXIAL_ACTION_PENALTY_WEIGHT"
  --target_reward_cheatcode_retreat_weight "$STATEFUL_RETREAT_WEIGHT"
  --target_reward_cheatcode_action_forward_scale "$STATEFUL_ACTION_FORWARD_SCALE"
  --target_reward_cheatcode_action_min_forward "$STATEFUL_ACTION_MIN_FORWARD"
  --target_reward_cheatcode_action_lateral_sigma "$STATEFUL_ACTION_LATERAL_SIGMA"
  --target_reward_cheatcode_action_lateral_sigma_far "$STATEFUL_ACTION_LATERAL_SIGMA_FAR"
  --target_reward_cheatcode_action_radius_schedule_far_depth "$STATEFUL_ACTION_RADIUS_SCHEDULE_FAR_DEPTH"
  --target_reward_cheatcode_action_radius_schedule_near_depth "$STATEFUL_ACTION_RADIUS_SCHEDULE_NEAR_DEPTH"
  --target_reward_policy_tcp_delta_sign "$STATEFUL_POLICY_TCP_DELTA_SIGN"
  --target_reward_stateful_orientation_enter_threshold "$STATEFUL_ORIENTATION_ENTER_THRESHOLD"
  --target_reward_exp_gated_sigma_theta_insert "$STATEFUL_SIGMA_THETA_INSERT"
  --target_reward_exp_gated_sigma_theta_insert_far "$STATEFUL_SIGMA_THETA_INSERT_FAR"
  --target_reward_consistency_body none
  --collision_contact_tune_prim_regex runtime_sdf_
  --collision_contact_tune_prim_regex cage_p0
  --replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes
  --near_gate_reset_max_iterations "$NEAR_GATE_RESET_MAX_ITERATIONS"
  --reset_settle_steps "$RESET_SETTLE_STEPS"
  --near_gate_reset_position_tolerance 0.00005
  --near_gate_reset_orientation_tolerance 0.0005
  --sfp_shrunk_box_margin_m "$SFP_SHRUNK_BOX_MARGIN_X" "$SFP_SHRUNK_BOX_MARGIN_Y" "$SFP_SHRUNK_BOX_MARGIN_Z"
  --collision_contact_offset_m 0.00002
  --collision_rest_offset_m 0.0
  --replace_nic_cage_p0_with_aligned_cubes
  --episode_length_s 45.0
  --device "$ISAAC_DEVICE"
  --force_delta_penalty_weight "$FORCE_DELTA_PENALTY_WEIGHT"
  --target_action_guide_weight 0.0
  --target_action_guide_collect_steps 0
  --target_action_guide_collect_blend 0.0
  --no-target_action_guide_use_episode_constant_action
  --no-insertion_action_guard
)
if [[ "$ABSOLUTE_IK_TARGET_POSE" == "1" || "$ABSOLUTE_IK_TARGET_POSE" == "true" || "$ABSOLUTE_IK_TARGET_POSE" == "TRUE" ]]; then
  COMMON_FLAGS+=(--absolute_ik_target_pose)
fi
write_level_config() {
  local level="$1"
  local count="$2"
  local output_root="$3"
  local out_config="$4"
  /workspace/isaaclab/_isaac_sim/python.sh - "$CONFIG_PATH" "$level" "$LEVEL_COUNT" "$count" "$output_root" "$out_config" <<'PY'
import copy, sys, yaml
src, level_raw, levels_raw, count_raw, output_root, out_config = sys.argv[1:7]
level = int(level_raw)
levels = max(1, int(levels_raw))
count = int(count_raw)
cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
t = 0.0 if levels <= 1 else level / float(levels - 1)
cfg = copy.deepcopy(cfg)
cfg["episodes"] = count
cfg["output_root"] = output_root
sg = cfg["start_near_gate"]
schedule = cfg.get("progression", {}).get("reintroduction_schedule") or []
if isinstance(schedule, list) and level < len(schedule):
    item = schedule[level]
    if not isinstance(item, dict):
        raise ValueError(f"progression.reintroduction_schedule[{level}] must be a mapping")
    sg["signed_axial_distance"] = True
    sg["axial_distance_m"] = {"initial": float(item["signed_depth_m"]), "terminal": float(item["signed_depth_m"])}
    sg["lateral_distance_m"] = {"initial": float(item.get("lateral_m", 0.0)), "terminal": float(item.get("lateral_m", 0.0))}
    sg["orientation_error_rad"] = {
        "initial": float(item.get("theta_rad", 0.0)),
        "terminal": float(item.get("theta_rad", 0.0)),
    }
    cfg["progression"]["schedule_name"] = str(item.get("name", f"level_{level:03d}"))
else:
    for name in ("axial_distance_m", "lateral_distance_m", "orientation_error_rad"):
        raw = sg[name]
        start = float(raw.get("initial", raw.get("start")))
        end = float(raw.get("terminal", raw.get("end")))
        value = start + t * (end - start)
        sg[name] = {"initial": value, "terminal": value}
cfg.setdefault("progression", {})["level_index"] = level
cfg["progression"]["level_count"] = levels
cfg["progression"]["interpolation_t"] = t
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"{sg['axial_distance_m']['initial']:.9f} {sg['lateral_distance_m']['initial']:.9f} {sg['orientation_error_rad']['initial']:.9f} {t:.9f}")
PY
}

materialize_level() {
  local level="$1"
  local kind="$2"
  local count="$3"
  local seed_offset="$4"
  local root
  root="$(readlink -m "$RUN_ROOT/levels/level_$(printf '%03d' "$level")/$kind")"
  local cfg_path="$root/config.yaml"
  rm -rf "$root"
  mkdir -p "$root"
  local values
  values="$(write_level_config "$level" "$count" "$root/generated" "$cfg_path")"
  /workspace/isaaclab/_isaac_sim/python.sh - "$cfg_path" "$seed_offset" <<'PY'
import sys, yaml
path, seed_offset = sys.argv[1], int(sys.argv[2])
cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
cfg["seed"] = int(cfg.get("seed", 20260613)) + seed_offset
with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  (cd aic && /workspace/isaaclab/_isaac_sim/python.sh aic_utils/aic_isaac/scripts/build_stateful_insertion_curriculum.py --config "$cfg_path" --overwrite >/dev/null)
  if ! compgen -G "$root/generated/episodes/episode_*.yaml" >/dev/null; then
    echo "[AIC stateful] failed to generate $kind episodes for level $level under $root/generated/episodes" >&2
    exit 1
  fi
  printf '%s %s\n' "$root/generated/episodes" "$values"
}

summarize_metrics() {
  local metrics="$1"
  local out_json="$2"
  /workspace/isaaclab/_isaac_sim/python.sh - "$metrics" "$out_json" <<'PY'
import json, math, sys
metrics_path, out_path = sys.argv[1], sys.argv[2]
rows = []
with open(metrics_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
summary = {
    "rows": len(rows),
    "last_step": rows[-1].get("step") if rows else None,
    "guide_blend_max": 0.0,
    "guide_action_any": False,
    "executed_minus_actor_max": 0.0,
    "guard_max": 0.0,
    "strict_success": False,
    "strict_success_count": 0,
    "best_depth_m": None,
    "best_lateral_m": None,
    "best_orientation_rad": None,
    "best_partial_candidate": None,
}
best_depth = -math.inf
best_lateral = math.inf
best_orientation = math.inf
best_score = math.inf
for row in rows:
    for key, dst in (
        ("target_action_guide_collect_blend_effective", "guide_blend_max"),
        ("executed_minus_actor_l1_mean", "executed_minus_actor_max"),
        ("insertion_action_guard_applied_fraction", "guard_max"),
    ):
        value = row.get(key)
        if isinstance(value, (int, float, bool)):
            summary[dst] = max(float(summary[dst]), float(value))
    summary["guide_action_any"] = summary["guide_action_any"] or row.get("guide_action_norm_mean") is not None
    terminated = row.get("terminated_by_env") or []
    for source in ("pre_step_insertion_geometry", "post_step_insertion_geometry"):
        geom = row.get(source) or {}
        strict = geom.get("strict_success_by_env") or []
        depths = geom.get("signed_depth_m_by_env") or []
        laterals = geom.get("lateral_error_m_by_env") or []
        orientations = geom.get("orientation_error_rad_by_env") or []
        n_env = max(len(depths), len(laterals), len(orientations), len(strict))
        for env_id in range(n_env):
            terminal_env = env_id < len(terminated) and bool(terminated[env_id])
            if terminal_env and source == "post_step_insertion_geometry":
                continue
            if env_id < len(strict) and bool(strict[env_id]):
                summary["strict_success_count"] += 1
                summary["strict_success"] = True
            d = float(depths[env_id]) if env_id < len(depths) and depths[env_id] is not None else None
            lat = float(laterals[env_id]) if env_id < len(laterals) and laterals[env_id] is not None else None
            ori = float(orientations[env_id]) if env_id < len(orientations) and orientations[env_id] is not None else None
            if d is not None and d > best_depth:
                best_depth = d
                summary["best_depth_m"] = {"step": row.get("step"), "source": source, "env": env_id, "depth_m": d, "lateral_m": lat, "orientation_rad": ori}
            if lat is not None and lat < best_lateral:
                best_lateral = lat
                summary["best_lateral_m"] = {"step": row.get("step"), "source": source, "env": env_id, "depth_m": d, "lateral_m": lat, "orientation_rad": ori}
            if ori is not None and ori < best_orientation:
                best_orientation = ori
                summary["best_orientation_rad"] = {"step": row.get("step"), "source": source, "env": env_id, "depth_m": d, "lateral_m": lat, "orientation_rad": ori}
            if d is not None and lat is not None and ori is not None:
                score = max(0.0, 0.005 - d) / 0.005 + max(0.0, lat - 0.002) / 0.002 + max(0.0, ori - 0.06) / 0.06
                if score < best_score:
                    best_score = score
                    summary["best_partial_candidate"] = {"step": row.get("step"), "source": source, "env": env_id, "score": score, "depth_m": d, "lateral_m": lat, "orientation_rad": ori}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, sort_keys=True))
PY
}

run_segment() {
  local cycle="$1"
  local level="$2"
  local checkpoint="$3"
  local train_episodes="$4"
  if [[ -n "${AIC_STATEFUL_TRAIN_EPISODES_OVERRIDE:-}" ]]; then
    train_episodes="${AIC_STATEFUL_TRAIN_EPISODES_OVERRIDE}"
  fi
  local out_dir="$RUN_ROOT/train_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")"
  local log_file="$RUN_ROOT/train_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level").log"
  local run_name="stateful_train_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")"
  local cleanup_pattern="serl/train.py .*--output_dir ${out_dir} .*--run_name ${run_name}"
  local reset_actor_flags=()
  if [[ "$RESET_ACTOR_HEAD" =~ ^(1|true|TRUE|yes|YES)$ && "$cycle" -eq $((START_CYCLE + 1)) ]]; then
    reset_actor_flags+=(--reset_actor_head)
  fi
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE" timeout --foreground --kill-after=300s "${SEGMENT_SECONDS}s" \
    /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
    "${COMMON_FLAGS[@]}" \
    --steps "$TRAIN_STEPS_LIMIT" \
    --updates 1000000 \
    --warmup_steps "$WARMUP_STEPS" \
    --actor_update_start_steps "$ACTOR_UPDATE_START_STEPS" \
    --actor_update_end_steps 0 \
    --update_every_steps 1 \
    --gradient_updates_per_step "$GRADIENT_UPDATES_PER_STEP" \
    --batch_size "$BATCH_SIZE" \
    --replay_capacity 12000 \
    --adapter_delta_clip "$ADAPTER_DELTA_CLIP" \
    --tcp_translation_action_clip "$TCP_TRANSLATION_ACTION_CLIP" \
    --tcp_rotation_action_clip "$TCP_ROTATION_ACTION_CLIP" \
    --actor_exploration_noise_std "$ACTOR_EXPLORATION_NOISE_STD" \
    --actor_exploration_noise_steps "$ACTOR_EXPLORATION_NOISE_STEPS" \
    --actor_exploration_noise_mode "$ACTOR_EXPLORATION_NOISE_MODE" \
    --actor_exploration_phase_lateral_multiplier "$ACTOR_EXPLORATION_PHASE_LATERAL_MULTIPLIER" \
    --actor_exploration_phase_orientation_std "$ACTOR_EXPLORATION_PHASE_ORIENTATION_STD" \
    --adapter_lr "$ADAPTER_LR" \
    --critic_lr "$CRITIC_LR" \
    --actor_q_weight "$ACTOR_Q_WEIGHT" \
    --actor_axial_purity_weight "$ACTOR_AXIAL_PURITY_WEIGHT" \
    --actor_axial_purity_lateral_weight "$ACTOR_AXIAL_PURITY_LATERAL_WEIGHT" \
    --actor_axial_purity_rotation_weight "$ACTOR_AXIAL_PURITY_ROTATION_WEIGHT" \
    --actor_axial_purity_forward_weight "$ACTOR_AXIAL_PURITY_FORWARD_WEIGHT" \
    --actor_axial_purity_backward_weight "$ACTOR_AXIAL_PURITY_BACKWARD_WEIGHT" \
    --actor_axial_purity_lateral_scale "$ACTOR_AXIAL_PURITY_LATERAL_SCALE" \
    --actor_axial_purity_rotation_scale "$ACTOR_AXIAL_PURITY_ROTATION_SCALE" \
    --actor_axial_purity_forward_scale "$ACTOR_AXIAL_PURITY_FORWARD_SCALE" \
    --actor_axial_purity_lateral_gate_m "$ACTOR_AXIAL_PURITY_LATERAL_GATE" \
    --actor_axial_purity_orientation_gate_rad "$ACTOR_AXIAL_PURITY_ORIENTATION_GATE" \
    --actor_axial_purity_min_s_m "$ACTOR_AXIAL_PURITY_MIN_S" \
    --actor_axial_purity_max_s_m "$ACTOR_AXIAL_PURITY_MAX_S" \
    --actor_axial_direction_sign "$ACTOR_AXIAL_DIRECTION_SIGN" \
    --actor_orientation_direction_sign "$ACTOR_ORIENTATION_DIRECTION_SIGN" \
    --actor_initial_axial_bias_m "$ACTOR_INITIAL_AXIAL_BIAS_M" \
    --adapter_penalty_weight "$ADAPTER_PENALTY_WEIGHT" \
    --act_preservation_weight "$ACT_PRESERVATION_WEIGHT" \
    --diagnostics_every 20 \
    --log_every 50 \
    --max_logged_image_steps 0 \
    --save_every_steps 0 \
    --save_latest_every_steps 200 \
    --save_final_checkpoint \
    --no-save_step_images \
    --no-save_videos \
    --no-save_replay_at_end \
    --episode_config_dir "$train_episodes" \
    --output_dir "$out_dir" \
    --run_name "$run_name" \
    "${reset_actor_flags[@]}" \
    --checkpoint "$checkpoint" \
    --num_envs "$NUM_ENVS" \
    --seed $((TRAIN_SEED_BASE + cycle)) \
    "${EXTRA_TRAIN_FLAGS[@]}" \
    > "$log_file" 2>&1
  local status="$?"
  set -e
  # Isaac's Python child can survive timeout's signal handling if train.py is
  # inside its graceful checkpoint path. Always reap any leftover process for
  # this exact output directory/run name before starting eval or the next cycle.
  pkill -TERM -f "$cleanup_pattern" || true
  sleep 5
  pkill -KILL -f "$cleanup_pattern" || true
  local new_checkpoint
  new_checkpoint="$(find "$out_dir" -maxdepth 2 -name checkpoint_latest.pt | head -n 1)"
  if [[ ! -f "$new_checkpoint" ]]; then
    echo "[AIC stateful] missing checkpoint after cycle $cycle level $level, train exit status $status" >&2
    exit 1
  fi
  printf '%s\n' "$new_checkpoint"
}

run_eval() {
  local cycle="$1"
  local level="$2"
  local checkpoint="$3"
  local eval_episodes="$4"
  local eval_root="$RUN_ROOT/eval_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")"
  local summary_json="$RUN_ROOT/summaries/eval_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")_summary.json"
  local log_file="$RUN_ROOT/eval_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level").log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE" /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
    "${COMMON_FLAGS[@]}" \
    --updates 1000000 \
    --warmup_steps 1000000 \
    --actor_update_start_steps 1000000 \
    --actor_update_end_steps 0 \
    --update_every_steps 1000000 \
    --batch_size 16 \
    --adapter_delta_clip "$ADAPTER_DELTA_CLIP" \
    --adapter_lr 0.0 \
    --critic_lr 1e-5 \
    --actor_q_weight 0.0 \
    --diagnostics_every 20 \
    --debug_diagnostics \
    --save_every_steps 0 \
    --save_latest_every_steps 0 \
    --no-save_replay_at_end \
    --no-save_final_checkpoint \
    --max_logged_image_steps 0 \
    --no-save_step_images \
    --no-save_videos \
    --episode_config_dir "$eval_episodes" \
    --output_dir "$eval_root" \
    --run_name "stateful_eval_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")" \
    --checkpoint "$checkpoint" \
    --num_envs "$EVAL_NUM_ENVS" \
    --seed $((EVAL_SEED_BASE + cycle)) \
    --steps "$EVAL_STEPS" \
    "${EXTRA_EVAL_FLAGS[@]}" \
    > "$log_file" 2>&1
  local eval_dir
  eval_dir="$(find "$eval_root" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
  summarize_metrics "$eval_dir/metrics.jsonl" "$summary_json" >/dev/null
  cp "$eval_dir/train_config.json" "$RUN_ROOT/summaries/eval_cycle$(printf '%04d' "$cycle")_level$(printf '%03d' "$level")_config.json" || true
  printf '%s\n' "$summary_json"
}

event_log="$RUN_ROOT/events.jsonl"
checkpoint="$BASE_CKPT"
level="$START_LEVEL"
cycle="$START_CYCLE"
episodes_used="$START_EPISODES_USED"
last_promotion_epoch="$(date -u +%s)"
printf '{"time":"%s","event":"start","run_root":"%s","total_episodes":%d,"eval_every":%d,"level_count":%d,"segment_seconds":%d,"checkpoint":"%s","start_cycle":%d,"start_level":%d,"start_episodes_used":%d}\n' \
  "$(date -u +%FT%TZ)" "$RUN_ROOT" "$TOTAL_EPISODES" "$EVAL_EVERY" "$LEVEL_COUNT" "$SEGMENT_SECONDS" "$checkpoint" "$START_CYCLE" "$START_LEVEL" "$START_EPISODES_USED" >> "$event_log"

if [[ "$DRY_RUN" == "1" ]]; then
  train_line="$(materialize_level 0 train "$TRAIN_EPISODE_VARIANTS" 1001)"
  eval_line="$(materialize_level 0 eval "$EVAL_EPISODE_VARIANTS" 2001)"
  printf '{"time":"%s","event":"dry_run_done","train":"%s","eval":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$train_line" "$eval_line" >> "$event_log"
  echo "[AIC stateful] dry run OK: run_root=$RUN_ROOT"
  exit 0
fi

while (( episodes_used < TOTAL_EPISODES && level < LEVEL_COUNT )); do
  cycle=$((cycle + 1))
  train_line="$(materialize_level "$level" train "$TRAIN_EPISODE_VARIANTS" $((1000 + cycle)))"
  eval_line="$(materialize_level "$level" eval "$EVAL_EPISODE_VARIANTS" $((2000 + cycle)))"
  train_episodes="$(awk '{print $1}' <<<"$train_line")"
  eval_episodes="$(awk '{print $1}' <<<"$eval_line")"
  axial="$(awk '{print $2}' <<<"$train_line")"
  lateral="$(awk '{print $3}' <<<"$train_line")"
  theta="$(awk '{print $4}' <<<"$train_line")"
  interp_t="$(awk '{print $5}' <<<"$train_line")"
  printf '{"time":"%s","event":"train_start","cycle":%d,"level":%d,"episodes_used":%d,"axial_m":%s,"lateral_m":%s,"theta_rad":%s,"t":%s,"checkpoint":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$cycle" "$level" "$episodes_used" "$axial" "$lateral" "$theta" "$interp_t" "$checkpoint" >> "$event_log"
  checkpoint="$(run_segment "$cycle" "$level" "$checkpoint" "$train_episodes")"
  cp "$checkpoint" "$RUN_ROOT/checkpoint_latest.pt"
  episodes_used=$((episodes_used + EVAL_EVERY))
  printf '{"time":"%s","event":"train_done","cycle":%d,"level":%d,"episodes_used":%d,"checkpoint":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$cycle" "$level" "$episodes_used" "$checkpoint" >> "$event_log"
  if [[ "$EXIT_AFTER_FIRST_TRAIN" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    printf '{"time":"%s","event":"exit_after_first_train","cycle":%d,"level":%d,"checkpoint":"%s"}\n' \
      "$(date -u +%FT%TZ)" "$cycle" "$level" "$checkpoint" >> "$event_log"
    echo "[AIC stateful] exit after first train requested: run_root=$RUN_ROOT"
    exit 0
  fi

  printf '{"time":"%s","event":"eval_start","cycle":%d,"level":%d,"checkpoint":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$cycle" "$level" "$checkpoint" >> "$event_log"
  summary_path="$(run_eval "$cycle" "$level" "$checkpoint" "$eval_episodes")"
  strict_success="$(/workspace/isaaclab/_isaac_sim/python.sh - "$summary_path" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("1" if bool(summary.get("strict_success")) else "0")
PY
)"
  if [[ "$strict_success" == "1" ]]; then
    level=$((level + 1))
    last_promotion_epoch="$(date -u +%s)"
    promoted=true
  else
    promoted=false
  fi
  printf '{"time":"%s","event":"eval_done","cycle":%d,"level_after_eval":%d,"summary":"%s","strict_success":%s,"promoted":%s}\n' \
    "$(date -u +%FT%TZ)" "$cycle" "$level" "$summary_path" "$strict_success" "$promoted" >> "$event_log"

  now_epoch="$(date -u +%s)"
  if (( now_epoch - last_promotion_epoch >= NO_PROGRESS_ASSESS_SECONDS )); then
    printf '{"time":"%s","event":"assessment_due_no_promotion","cycle":%d,"level":%d,"seconds_since_promotion":%d,"summary":"%s"}\n' \
      "$(date -u +%FT%TZ)" "$cycle" "$level" "$((now_epoch - last_promotion_epoch))" "$summary_path" >> "$event_log"
    last_promotion_epoch="$now_epoch"
  fi

  if (( cycle > 3 )); then
    rm -rf "$RUN_ROOT/train_cycle$(printf '%04d' $((cycle - 3)))"_level*
  fi
done

printf '{"time":"%s","event":"done","cycle":%d,"level":%d,"episodes_used":%d,"checkpoint":"%s"}\n' \
  "$(date -u +%FT%TZ)" "$cycle" "$level" "$episodes_used" "$checkpoint" >> "$event_log"
echo "[AIC stateful] done: run_root=$RUN_ROOT level=$level episodes_used=$episodes_used checkpoint=$checkpoint"
