#!/usr/bin/env bash
set -euo pipefail

# Run inside the isaac-lab-base container from /workspace/isaaclab.
# Actor-only online RL with the multiplicative exponential-gated insertion
# reward. No expert policy, no guide/guard action override, and no offline replay
# loading are used in training or evaluation.

cd /workspace/isaaclab

ROOT="aic/outputs/agentic_reward_curriculum_20260529"
RUN_ROOT="${AIC_ACTOR_ONLY_EXP_GATED_ROOT:-/tmp/aic_actor_only_exp_gated_rl_progressive_20x0_to_60x15_additive_lateral_progress_veto}"
TRAIN_EPISODES="${AIC_ACTOR_ONLY_EXP_GATED_TRAIN_EPISODES:-${AIC_ACTOR_ONLY_EXP_GATED_EPISODES:-$ROOT/generated_episode_configs/progressive_20x0_to_60x15_randomized/episodes}}"
EVAL_EPISODES="${AIC_ACTOR_ONLY_EXP_GATED_EVAL_EPISODES:-$ROOT/generated_episode_configs/v1260_selected_tip_preserving_true40x10_initial_randomized/episodes}"
ACT_TS="aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt"
BASE_CKPT="${AIC_ACTOR_ONLY_EXP_GATED_BASE_CKPT:-/tmp/aic_v1367_clean_v1077_v1254_true40x10_4h_segments_clip1mm/2026-06-05_03-42-24_2026-06-04_train_v1367_clean_v1077_v1254_clip1mm_constant40x10_seg08/checkpoint_latest.pt}"
if [[ ! -f "$BASE_CKPT" ]]; then
  BASE_CKPT="$ROOT/policy_train_runs/2026-06-02_train_v1077_gentle_offline_true40x10/2026-06-02_23-44-40_isaac_online_serl/checkpoint_latest.pt"
fi

MAX_HOURS="${AIC_ACTOR_ONLY_EXP_GATED_MAX_HOURS:-10}"
SEGMENT_STEPS="${AIC_ACTOR_ONLY_EXP_GATED_SEGMENT_STEPS:-200}"
EVAL_INTERVAL_SECONDS="${AIC_ACTOR_ONLY_EXP_GATED_EVAL_INTERVAL_SECONDS:-1800}"
EVAL_STEPS="${AIC_ACTOR_ONLY_EXP_GATED_EVAL_STEPS:-900}"
NUM_ENVS="${AIC_ACTOR_ONLY_EXP_GATED_NUM_ENVS:-4}"
EVAL_NUM_ENVS="${AIC_ACTOR_ONLY_EXP_GATED_EVAL_NUM_ENVS:-1}"
GRADIENT_UPDATES_PER_STEP="${AIC_ACTOR_ONLY_EXP_GATED_GRADIENT_UPDATES_PER_STEP:-1}"
TRAIN_SEED_BASE="${AIC_ACTOR_ONLY_EXP_GATED_TRAIN_SEED_BASE:-11600}"
EVAL_SEED_BASE="${AIC_ACTOR_ONLY_EXP_GATED_EVAL_SEED_BASE:-12600}"
CONSECUTIVE_STRICT_SUCCESS_TARGET="${AIC_ACTOR_ONLY_EXP_GATED_CONSECUTIVE_STRICT_SUCCESS_TARGET:-5}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${AIC_ACTOR_ONLY_EXP_GATED_CUDA_VISIBLE_DEVICES:-0}"
ISAAC_DEVICE="${AIC_ACTOR_ONLY_EXP_GATED_DEVICE:-cuda:0}"

mkdir -p "$RUN_ROOT/summaries"

COMMON_FLAGS=(
  --task AIC-Task-v0
  --headless
  --rendering_mode performance
  --act_torchscript "$ACT_TS"
  --policy_hz 20
  --enable_contact_sensor
  --disable_ppo_resnet_observation_terms
  --fix_isaac_ik_xy_sign
  --absolute_ik_target_pose
  --no-treat_time_limit_truncation_as_terminal
  --gripper_joint_position 0.0035405
  --critic_image_encoder_override small_conv
  --target_success_orientation_threshold 0.03
  --target_success_axial_threshold 0.0005
  --target_success_lateral_threshold 0.0005
  --target_reward_orientation_error_mode axis
  --target_reward_orientation_axis_local 0 0 1
  --reward_preset multiplicative_exp_gated_insertion_v1
  --target_reward_exp_gated_action_axis_source policy_tcp_delta
  --target_reward_exp_gated_phase_gate_insertion_credit
  --target_reward_exp_gated_lateral_alignment_require_axial_quiet
  --target_reward_consistency_body none
  --collision_contact_tune_prim_regex runtime_sdf_
  --collision_contact_tune_prim_regex cage_p0
  --replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes
  --near_gate_reset_max_iterations 8
  --near_gate_reset_position_tolerance 0.00005
  --near_gate_reset_orientation_tolerance 0.0005
  --sfp_shrunk_box_margin_m 0.00030 0.0 0.00030
  --collision_contact_offset_m 0.00002
  --collision_rest_offset_m 0.0
  --replace_nic_cage_p0_with_aligned_cubes
  --episode_length_s 45.0
  --device "$ISAAC_DEVICE"
  --target_action_guide_weight 0.0
  --target_action_guide_collect_steps 0
  --target_action_guide_collect_blend 0.0
  --no-target_action_guide_use_episode_constant_action
  --no-insertion_action_guard
)

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

def nums(key):
    return [r.get(key) for r in rows if isinstance(r.get(key), (int, float, bool))]

summary = {
    "rows": len(rows),
    "last_step": rows[-1].get("step") if rows else None,
    "guide_blend_max": max(nums("target_action_guide_collect_blend_effective") or [0.0]),
    "guide_action_any": any(r.get("guide_action_norm_mean") is not None for r in rows),
    "actor_to_guide_any": any(r.get("actor_to_guide_l1_mean") is not None for r in rows),
    "executed_minus_actor_max": max(nums("executed_minus_actor_l1_mean") or [0.0]),
    "guard_max": max(nums("insertion_action_guard_applied_fraction") or [0.0]),
    "strict_success": False,
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
    for source in ("pre_step_insertion_geometry", "post_step_insertion_geometry"):
        geom = row.get(source) or {}
        strict = geom.get("strict_success_by_env")
        if strict and any(strict):
            summary["strict_success"] = True
        depths = geom.get("signed_depth_m_by_env") or []
        laterals = geom.get("lateral_error_m_by_env") or []
        orientations = geom.get("orientation_error_rad_by_env") or []
        for env_id in range(max(len(depths), len(laterals), len(orientations))):
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

preflight_audit() {
  local audit_root="$RUN_ROOT/preflight_audit"
  mkdir -p "$audit_root"
  /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py \
    --config-dir "$(dirname "$TRAIN_EPISODES")" \
    --name actor_only_exp_gated_progressive_20x0_to_60x15_train \
    --output-dir "$audit_root"
  /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py \
    --config-dir "$(dirname "$EVAL_EPISODES")" \
    --name actor_only_exp_gated_randomized_true40x10_eval \
    --output-dir "$audit_root"
  /workspace/isaaclab/_isaac_sim/python.sh - "$audit_root/actor_only_exp_gated_progressive_20x0_to_60x15_train.json" "$audit_root/actor_only_exp_gated_randomized_true40x10_eval.json" <<'PY'
import json, sys
train_report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
eval_report = json.load(open(sys.argv[2], "r", encoding="utf-8"))
stats = train_report["episode_stats"]
ax = stats["requested_axial_distance_m"]
lat = stats["requested_lateral_distance_m"]
eps = 1e-6
train_ok = (
    train_report["episode_count"] > 0
    and abs(ax["min"] + 0.060) < eps
    and abs(ax["max"] + 0.020) < eps
    and abs(lat["min"] - 0.0) < eps
    and abs(lat["max"] - 0.015) < eps
)
if not train_ok:
    raise SystemExit(f"progressive train episode audit failed: axial={ax}, lateral={lat}")
eval_stats = eval_report["episode_stats"]
eval_ax = eval_stats["requested_axial_distance_m"]
eval_lat = eval_stats["requested_lateral_distance_m"]
eval_ok = (
    eval_report["episode_count"] > 0
    and abs(eval_ax["min"] + 0.040) < eps
    and abs(eval_ax["max"] + 0.040) < eps
    and abs(eval_lat["min"] - 0.010) < eps
    and abs(eval_lat["max"] - 0.010) < eps
)
if not eval_ok:
    raise SystemExit(f"true40x10 eval episode audit failed: axial={eval_ax}, lateral={eval_lat}")
print(
    "episode audit OK: "
    f"train_episodes={train_report['episode_count']} train_axial={ax['min']:.6f}..{ax['max']:.6f} "
    f"train_lateral={lat['min']:.6f}..{lat['max']:.6f} "
    f"eval_episodes={eval_report['episode_count']} eval_axial={eval_ax['mean']:.6f} eval_lateral={eval_lat['mean']:.6f}"
)
PY
}

run_segment() {
  local cycle="$1"
  local checkpoint="$2"
  local out_dir="$RUN_ROOT/train_cycle$(printf '%03d' "$cycle")"
  local log_file="$RUN_ROOT/train_cycle$(printf '%03d' "$cycle").log"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE" /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
    "${COMMON_FLAGS[@]}" \
    --steps "$SEGMENT_STEPS" \
    --updates 1000000 \
    --warmup_steps 1 \
    --actor_update_start_steps 1 \
    --actor_update_end_steps 0 \
    --update_every_steps 1 \
    --gradient_updates_per_step "$GRADIENT_UPDATES_PER_STEP" \
    --batch_size 32 \
    --replay_capacity 12000 \
    --adapter_lr 1e-5 \
    --critic_lr 1e-5 \
    --actor_q_weight 3e-4 \
    --adapter_penalty_weight 1e-5 \
    --act_preservation_weight 1e-5 \
    --diagnostics_every 20 \
    --log_every 50 \
    --max_logged_image_steps 0 \
    --save_every_steps 0 \
    --save_latest_every_steps "$SEGMENT_STEPS" \
    --no-save_step_images \
    --no-save_videos \
    --no-save_replay_at_end \
    --save_final_checkpoint \
    --episode_config_dir "$TRAIN_EPISODES" \
    --output_dir "$out_dir" \
    --run_name "actor_only_exp_gated_train_cycle$(printf '%03d' "$cycle")" \
    --checkpoint "$checkpoint" \
    --num_envs "$NUM_ENVS" \
    --seed $((TRAIN_SEED_BASE + cycle)) \
    > "$log_file" 2>&1
  find "$out_dir" -maxdepth 2 -name checkpoint_latest.pt | head -n 1
}

run_eval() {
  local cycle="$1"
  local checkpoint="$2"
  local eval_root="$RUN_ROOT/eval_cycle$(printf '%03d' "$cycle")"
  local summary_json="$RUN_ROOT/summaries/eval_cycle$(printf '%03d' "$cycle")_summary.json"
  local log_file="$RUN_ROOT/eval_cycle$(printf '%03d' "$cycle").log"
  # Keep eval metrics for action/reward diagnostics on failed lateral-learning runs.
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE" /workspace/isaaclab/_isaac_sim/python.sh \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
    "${COMMON_FLAGS[@]}" \
    --updates 1000000 \
    --warmup_steps 1000000 \
    --actor_update_start_steps 1000000 \
    --actor_update_end_steps 0 \
    --update_every_steps 1000000 \
    --batch_size 16 \
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
    --episode_config_dir "$EVAL_EPISODES" \
    --output_dir "$eval_root" \
    --run_name "actor_only_exp_gated_eval_cycle$(printf '%03d' "$cycle")" \
    --checkpoint "$checkpoint" \
    --num_envs "$EVAL_NUM_ENVS" \
    --seed $((EVAL_SEED_BASE + cycle)) \
    --steps "$EVAL_STEPS" \
    > "$log_file" 2>&1
  local eval_dir
  eval_dir="$(find "$eval_root" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
  summarize_metrics "$eval_dir/metrics.jsonl" "$summary_json"
  cp "$eval_dir/train_config.json" "$RUN_ROOT/summaries/eval_cycle$(printf '%03d' "$cycle")_config.json" || true
  # Preserve eval metrics so failed lateral-learning runs can be inspected at
  # the action/reward/component level.
}

preflight_audit

checkpoint="$BASE_CKPT"
cycle=0
start_epoch="$(date -u +%s)"
deadline_epoch="$((start_epoch + MAX_HOURS * 3600))"
last_eval_epoch=0
consecutive_strict_success=0
event_log="$RUN_ROOT/events.jsonl"

while (( "$(date -u +%s)" < deadline_epoch )); do
  cycle=$((cycle + 1))
  printf '{"time":"%s","event":"train_start","cycle":%d,"checkpoint":"%s"}\n' "$(date -u +%FT%TZ)" "$cycle" "$checkpoint" >> "$event_log"
  new_checkpoint="$(run_segment "$cycle" "$checkpoint")"
  if [[ ! -f "$new_checkpoint" ]]; then
    echo "[AIC actor-only exp-gated] missing checkpoint after cycle $cycle" >&2
    exit 1
  fi
  checkpoint="$new_checkpoint"
  cp "$checkpoint" "$RUN_ROOT/checkpoint_latest.pt"
  printf '{"time":"%s","event":"train_done","cycle":%d,"checkpoint":"%s"}\n' "$(date -u +%FT%TZ)" "$cycle" "$checkpoint" >> "$event_log"

  now_epoch="$(date -u +%s)"
  if (( last_eval_epoch == 0 || now_epoch - last_eval_epoch >= EVAL_INTERVAL_SECONDS )); then
    printf '{"time":"%s","event":"eval_start","cycle":%d,"checkpoint":"%s"}\n' "$(date -u +%FT%TZ)" "$cycle" "$checkpoint" >> "$event_log"
    run_eval "$cycle" "$checkpoint"
    summary_path="$RUN_ROOT/summaries/eval_cycle$(printf '%03d' "$cycle")_summary.json"
    strict_success="$(/workspace/isaaclab/_isaac_sim/python.sh - "$summary_path" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("1" if bool(summary.get("strict_success")) else "0")
PY
)"
    if [[ "$strict_success" == "1" ]]; then
      consecutive_strict_success=$((consecutive_strict_success + 1))
    else
      consecutive_strict_success=0
    fi
    last_eval_epoch="$(date -u +%s)"
    printf '{"time":"%s","event":"eval_done","cycle":%d,"summary":"%s","strict_success":%s,"consecutive_strict_success":%d,"target":%d}\n' \
      "$(date -u +%FT%TZ)" "$cycle" "$summary_path" "$strict_success" "$consecutive_strict_success" "$CONSECUTIVE_STRICT_SUCCESS_TARGET" >> "$event_log"
    if (( consecutive_strict_success >= CONSECUTIVE_STRICT_SUCCESS_TARGET )); then
      printf '{"time":"%s","event":"stop_consecutive_strict_success","cycle":%d,"consecutive_strict_success":%d}\n' \
        "$(date -u +%FT%TZ)" "$cycle" "$consecutive_strict_success" >> "$event_log"
      echo "[AIC actor-only exp-gated] stopping after $consecutive_strict_success consecutive strict-success evals"
      exit 0
    fi
  fi

  old_cycle="$(printf '%03d' $((cycle - 3)))"
  if (( cycle > 3 )); then
    rm -rf "$RUN_ROOT/train_cycle$old_cycle"
  fi
done

echo "[AIC actor-only exp-gated] done"
