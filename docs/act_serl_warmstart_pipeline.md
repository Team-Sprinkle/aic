# ACT Warm-Start Smoke Pipeline

This is the first minimal end-to-end imitation-learning path for AIC:

```text
Gazebo CheatCode expert rollouts -> native LeRobot dataset -> ACT smoke training -> offline SERL smoke -> Isaac PPO/RSL-RL
```

It deliberately uses CheatCode-based trajectories only. It does not use VLM
planner trajectories or Gazebo recovery rollouts. A minimal offline SERL-style
pretraining smoke path is documented in
[`offline_serl_pretrain.md`](offline_serl_pretrain.md), and the current Isaac
PPO/RSL-RL Stage 5 path is documented in
[`isaac_rl_stage5.md`](isaac_rl_stage5.md).
The repo has a `Team-Sprinkle/mip` dependency and `RunMIP` runtime policy
integration, but no direct in-repo MIP ACT training wrapper for this flow; this
smoke path uses LeRobot training directly.

The runtime policy/control interfaces are left untouched. In particular,
`aic_model/aic_model/policy.py` continues to support both Cartesian
`MotionUpdate` and joint-space `JointMotionUpdate` commands through
`MoveRobotCallback`. Action mode is a recorder/dataset/training concern.

For the current Stage 1-4 hybrid cleanup commands and 2026-04-30 runtime
results, see
[`hybrid_stage1_to_4_nominal_warmstart.md`](hybrid_stage1_to_4_nominal_warmstart.md).

## Generate Dataset Artifacts Without Gazebo

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_minimal_10_cheatcode.yaml \
  --target-accepted-override 2 \
  --max-attempts-override 3 \
  --dry-run \
  --skip-recording
```

This writes request, engine config, per-trial YAMLs, and a generation summary.
It does not launch Gazebo or record data.

## Generate 10 CheatCode Trajectories

Run this only in an environment where the AIC simulation/runtime is available:

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_minimal_10_cheatcode.yaml \
  --target-accepted-override 10 \
  --max-attempts-override 15
```

Expected accepted dataset root:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

## Inspect Dataset Schema

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/inspect_dataset_schema.py \
  outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

The inspector reads `meta/info.json`, prints feature keys, FPS, robot type, and
infers whether the action schema is Cartesian or joint-like from feature names.

For the canonical hybrid metadata view:

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/inspect_hybrid_schema.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --action-horizon 8 \
  --json
```

The 2026-04-30 nominal hybrid cleanup produced this accepted dataset:

```text
outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset
```

It contains 10 Gazebo CheatCode/no-contact episodes, 5399 frames, 6D
Cartesian delta-pose actions, 32D low-dimensional state, and three camera video
streams.

## Train ACT Smoke Policy

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/train_act_policy.py \
  --dataset-root outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset \
  --output-dir outputs/train/act_smoke \
  --job-name act_smoke \
  --steps 200 \
  --batch-size 4 \
  --chunk-size 16 \
  --n-action-steps 8 \
  --device cuda
```

The wrapper builds a `lerobot-train` command with `--policy.type=act`,
`--dataset.video_backend=pyav`, local `--dataset.root`, and wandb disabled by
default. It also exposes ACT action chunking through `--chunk-size`,
`--n-action-steps`, and `--n-obs-steps` and passes them to LeRobot as
`--policy.chunk_size`, `--policy.n_action_steps`, and `--policy.n_obs_steps`.
Use `--dry-run` to print the exact command first.

## Future Task Compatibility

- Add future SFP-to-NIC and SC-to-SC request YAMLs under the same request format.
- Set `generation.action_mode` to `cartesian` or `joint` according to the recorder
  action schema being collected.
- Keep task-family details in request YAMLs and dataset metadata rather than
  narrowing `policy.py`.
- Reuse the same schema inspector and training wrapper before training on larger
  mixed datasets.

## Corrected Warm-Start Loop

1. Standardize obs/action inspection around datasets without narrowing
   `policy.py`.
2. Collect Gazebo nominal expert trajectories using CheatCode/no-contact mode.
3. Train ACT / BC warm-start policy on expert trajectories.
4. Run minimal offline SERL pretraining on Gazebo expert data. Current
   limitation: lowdim path only; ACT checkpoint loading transfers an output
   action prior, not full transformer hidden layers.
5. Train Isaac Lab online RL using PPO/RSL-RL for acceleration. Current
   implementation: PPO/RSL-RL, not true off-policy SERL/SAC yet. Main additions:
   richer domain randomization and optional insertion-aware rewards.
6. Validate Isaac-trained checkpoint in instrumented Gazebo rollout mode.
7. Classify Gazebo rollout outcomes:
   A. immediate nonsense/interface failure: debug adapter, do not spend recovery
      budget.
   B. near-port contact/insertion failure: save failed policy prefix to
      `online_buffer`; oracle takes over from current state; save recovery
      suffix to `demo_buffer_recovery`.
   C. wandering/timeout: save failed rollout to `online_buffer` only.
   D. success: save rollout/checkpoint candidate.
   E. unrecoverable failure: save prefix with failure penalty; no recovery demo
      unless recovery succeeds.
8. Offline refresh: critic/value training on all data; BC only on nominal +
   oracle recovery demos.
9. Update Isaac randomization based on Gazebo failure modes.
10. Repeat coarse Isaac <-> Gazebo loop.
11. Final official Gazebo eval.

Future work: true Isaac SERL/SAC replay-buffer training, full checkpoint
transfer into architecture-compatible policies, and same-state Gazebo recovery
automation.
