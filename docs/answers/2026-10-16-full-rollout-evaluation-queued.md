# Full rollout evaluation queued

The continuation is queued to start automatically after the active transport evaluation and its analysis finish. The current jobs remain unchanged.

1. Run **186 full-pipeline episodes**: three per scenario, covering SC/SFP, card count and requested port. For SFP, requested NIC identity is also part of the scenario. This gives 12 SC groups and 50 SFP groups.
2. Use **rule transport → coarse/native pose refinement → the trained shared 150 mm insertion actor**, with hierarchical backtracking enabled. Four separate integration checks exercise both completed actor checkpoints before the main sweep. The main actor is selected from existing validation metrics before autonomous results, not from official scores.
3. Run each **complete `sample_config.yaml` and released `eval_config.yaml` three times**, preserving original trial order and task time limits. Report original engine totals and per-trial scores, plus mean/min/max across repetitions.
4. Report handoff, actual insertion, force/contact failures and latency separately. Preserve failed episodes and save smooth videos for representative scenarios and failed isolated rollouts.

This is evaluation, not additional training or RL. It uses one GPU sequentially. Estimated runtime is 12–24 hours. No tuning on released evaluation scenes; reserved internal final scenes remain sealed. Results are not yet available.

Plan: [Full evaluation record](../experiments/2026-10-16-full-pipeline-three-per-scenario.md).

Live artifacts: `/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/`. Follow `state.json`, `progress.json`, `results.md` and `configuration_scores.json` as they appear.
