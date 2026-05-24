# Agentic Reward/Curriculum Loop Design 2026-05-23

The reusable loop is implemented in `aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py`.

## Harness Contract

Inputs:
- YAML/JSON config with `episode_config_dir`, checkpoint, env/step budget, existing run dirs, and candidate flag overrides.
- Defaults now target `outputs/analysis/isaac_near_gate_handoff_r105/episode_configs/episodes` so the first bounded loop tests the final insertion/retention regime.

Outputs:
- `outputs/agentic_reward_curriculum_<date>/config.json`
- `git_status.txt`
- `git_diff.patch`
- per-iteration `agent_decision.json`
- per-iteration `summary.md`
- aggregate `agent_loop_results.json` and `agent_loop_results.csv`
- Isaac run folders with `train_config.json`, `metrics.jsonl`, `diagnostics_summary.json`, `cheatcode_phase_summary.json`, checkpoints, and center/left/right images when camera logging is enabled.

Selection rule:
- Strict success is never reward return.
- The parser prefers `post_step_insertion_geometry` and `post_step_all_body_insertion_geometry`.
- Strict thresholds in the harness are depth fraction `>=0.90`, `r<=0.0005 m`, `theta<=0.030 rad`, module consistency gate `>=0.80`, and max force `<=35 N`.
- Candidates are promoted only if strict success occurs or if the score improves without violating strict metrics. Reward-only improvements are insufficient.

Failure labels:
- `strict_success`
- `near_success_orientation_blocked`
- `near_success_module_consistency_blocked`
- `tip_depth_false_positive`
- `lateral_bypass`
- `rotation_induced_lateral_sweep`
- `no_axial_progress`
- `timeout_or_episode_too_short`
- `reset_regression`
- `contact_spike`
- `controller_realization_mismatch`
- `unstable_learning_or_actor_drift`
- `orientation_plateau_env_or_card_dependent`
- `insufficient_logs`

## Guard Changes

`train.py` now exposes:
- `--insertion_action_guard_reject_predicted_r_increase`
- `--insertion_action_guard_predicted_r_increase_margin_m`
- `--insertion_action_guard_predicted_r_reject_backoff_m`
- `--insertion_action_guard_retention_require_orientation_gate`
- `--target_action_guide_adaptive_orientation_sign`
- `--target_action_guide_orientation_probe_basis`
- `--debug_audit_rotation_axes`
- `--insertion_action_guard_module_recovery`
- `--insertion_action_guard_module_recovery_zero_rotation`

The predicted-r rejection path computes the semantic next-step lateral error from the guarded world delta and rejects commands that increase `r` beyond the configured margin. The orientation-gated retention path keeps lateral correction and final-orientation hold available, but blocks retention-driven positive axial motion while the semantic tip orientation is outside the gate.

The controller-aware orientation path adds two bounded diagnostics:
- Adaptive orientation sign flips the final-orientation guide sign if realized semantic theta worsens after the previous correction.
- Basis probing evaluates +/- root/action-frame rotation bases by predicted semantic tip-axis improvement and predicted lateral sweep. A separate debug audit can execute pure +/- rotation bases with guide/guard disabled.

The module recovery path detects near-seated false positives from tip depth, lateral error, semantic theta, and module consistency. When active it commands shallow backoff toward a configured depth and can zero rotation to keep rotational tip sweep from defeating the backoff.

This keeps the old behavior available by default in `train.py`; the harness opts into the stricter behavior for new agentic candidates.

## Tuning Policy

Current bounded sequence:
1. Re-evaluate existing best/fallback runs with strict post-step metrics.
2. Run final handoff retention probe with predicted-r rejection.
3. If it reaches depth but theta/module fail, enable orientation-gated retention.
4. Compare full-quat and final-window axis-only orientation refinement.
5. Promote only candidates that improve depth, `r`, `theta`, and module consistency together.

Next policy choice from current evidence:
- The next code change should target controller-aware semantic tip orientation trimming before axial insertion. Reward/curriculum tuning alone is unlikely to fix the current `theta ~= 0.06 rad` plateau because the guide/evaluator already withholds axial progress when orientation is gated.
