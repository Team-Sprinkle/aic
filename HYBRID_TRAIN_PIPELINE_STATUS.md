# Hybrid Train Pipeline Status

Last updated: 2026-04-30 on branch `feat/hybrid-train`, after canonical
hybrid schema cleanup, nominal Gazebo CheatCode dataset generation, ACT smoke
training, and offline SERL smoke training.

Purpose: this file is the handoff state for future Codex sessions working on the
full hybrid training pipeline. It distinguishes actual artifact-producing runs
from short smoke/adapter checks.

## Overall State

The first nominal warm-start artifacts now exist: a 10-episode accepted Gazebo
CheatCode LeRobot dataset, a 200-step ACT checkpoint, and a 200-step offline
SERL checkpoint trained on the same dataset. It is not yet the full hybrid
pipeline: Isaac-to-Gazebo transfer loops, failure classification, recovery
buffers, true online SERL/SAC, and final official Gazebo evaluation are still
incomplete or only partially represented by older unrelated utilities.

## Step Status

1. Standardize obs/action interface
   - Status: Improved/partial.
   - Implemented: Isaac Lab `AIC-Task-v0` uses a 6D differential IK relative-pose action on `wrist_3_link`.
   - Implemented: camera-required Isaac policy observations are validated at runtime: `center_rgb`, `left_rgb`, and `right_rgb` must exist and load.
   - Confirmed camera policy observation shape: `(3154,)`, with three `(1000,)` ResNet18 image-feature terms plus low-dimensional terms.
   - Implemented: canonical dataset/training metadata inspection in
     `aic_utils/lerobot_robot_aic/lerobot_robot_aic/hybrid_schema.py`.
   - Implemented: CLI JSON inspector in
     `aic_utils/lerobot_robot_aic/scripts/inspect_hybrid_schema.py`.
   - The schema reports `task_family`, `simulator_source`, `action_mode`,
     `action_dim`, `action_horizon`, `obs_mode`, `obs_dim`, camera keys, and
     low-dimensional observation keys without assuming all actions are 6D or all
     tasks are SFP-to-NIC.
   - Missing: final Isaac/Gazebo runtime policy adapter that consumes this
     canonical metadata during transfer/evaluation.

2. Collect Gazebo nominal expert trajectories, no-contact VLM/oracle + CheatCode insertion
   - Status: Complete for the requested 10-episode nominal smoke dataset.
   - Added request YAML:
     `aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_hybrid_nominal_10.yaml`.
   - Accepted dataset path:
     `outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset`.
   - Accepted episodes: 10.
   - Accepted frames: 5399.
   - Raw generation result: 15 attempts, 14 saved raw episodes, 13 score-passing
     episodes before capping accepted output at the requested 10.
   - Videos/images exist: yes, three video streams under
     `videos/observation.images.{center,left,right}_camera/`.
   - Command attempted:
     `pixi run python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_hybrid_nominal_10.yaml --target-accepted-override 10 --max-attempts-override 15`.
   - Fixes needed during generation: restart the eval container per trial, raise
     per-trial timeout to 900 seconds, record per-camera video shapes correctly,
     cap accepted selection at the requested target, and allow filtering to run
     even if some attempts fail after enough raw episodes are saved.
   - Schema summary: `task_family=sfp_to_nic`, `simulator_source=gazebo`,
     `action_mode=cartesian`, `action_dim=6`, `action_horizon=8`,
     `obs_mode=image_lowdim`, `obs_dim=32`.

3. Train ACT / BC warm start
   - Status: Complete for a 200-step smoke ACT run.
   - Dataset:
     `outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset`.
   - Output path: `outputs/train/hybrid_act_nominal_n10`.
   - ACT checkpoint:
     `outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model/model.safetensors`.
   - ACT settings: `steps=200`, `batch_size=4`, `chunk_size=16`,
     `n_action_steps=8`, `n_obs_steps=1`.
   - Note: first ACT run completed optimization but failed to create the
     checkpoint because `outputs/train` was root-owned by prior generated
     artifacts; ownership was fixed and the same command was rerun successfully.

4. SERL offline pretrain on Gazebo expert data
   - Status: Complete for a 200-step low-dimensional offline SERL smoke run.
   - Existing offline SERL path is documented as a low-dimensional smoke/pretrain path.
   - Dataset:
     `outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset`.
   - Checkpoint:
     `outputs/train/hybrid_offline_serl_nominal_n10/checkpoint_latest.pt`.
   - Offline SERL settings: `steps=200`, `batch_size=32`, `action_horizon=8`,
     `hidden_dim=256`, `num_layers=3`.
   - Checkpoint metadata: `obs_dim=32`, flattened actor `action_dim=48`,
     `action_horizon=8`, dataset `action_mode=cartesian`; normalization stats
     include `obs_mean`, `obs_std`, `action_mean`, and `action_std`.
   - Implemented after the nominal smoke run: `--act-checkpoint` support in
     `train_offline_serl.py`. The bridge validates a LeRobot ACT checkpoint and
     transfers `model.action_head.bias` into the SERL actor output bias repeated
     across the configured action horizon. This is an action-prior warm start,
     not full ACT transformer-to-MLP hidden-layer transfer.

5. Isaac Lab RL with dense reward + heavy randomization
   - Status: Partial, with camera-required PPO validated.
   - Implemented: Isaac Lab PPO/RSL-RL training entry points.
   - Implemented: dense-ish reward terms for end-effector pose tracking, orientation tracking, sparse reaching bonus, smoothness penalties, and optional insertion-aware terms.
   - Implemented: randomization profile plumbing for `none`, `light`, and `heavy`.
   - Implemented: training now enables cameras by default and fails if camera observations do not exist or do not load.
   - Actual artifact-producing camera training run: one PPO iteration, 24 timesteps, produced `model_0.pt`.
   - Implemented after the nominal smoke run: `--init-policy-checkpoint` now
     accepts an offline SERL checkpoint and passes it into the RSL-RL train
     entrypoint. The current bridge initializes PPO actor output bias/std from
     the SERL first-action prior and copies exact-shape tensors if future
     architectures match.
   - Missing: long heavy-randomization training to convergence.
   - Missing: insertion reward based on semantic cable-tip/port frames; current optional insertion terms use approximate object roots.

6. Gazebo transfer validation in instrumented mode
   - Status: Implemented as a SERL checkpoint validation path; not yet run to
     final score in this update.
   - Added:
     `aic_utils/gazebo_rl/scripts/serl_transfer_validate.py`.
   - Added:
     `aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`.
   - The validator loads an offline SERL checkpoint, converts Gazebo bridge
     observations into the canonical 32D low-dimensional state, emits the first
     6D action from the chunked SERL actor, parses Gazebo scoring output, and
     writes `transfer_validation_summary.json`.
   - Missing: Isaac PPO checkpoint export into the Gazebo policy bridge. The
     implemented transfer validator is SERL -> Gazebo, not RSL-RL PPO -> Gazebo.

7. Classify failures
   - Status: Minimal transfer validator classification implemented.
   - `serl_transfer_validate.py` classifies transfer rollouts as `success`,
     `transfer_failure`, or `no_score` from scoring output.
   - Required buckets are still missing:
     - A. nonsense/interface failure -> debug adapter
     - B. near-port contact failure -> oracle takeover/recovery
     - C. wandering/timeout -> online_buffer only
     - D. success -> online_buffer / checkpoint candidate
     - E. unrecoverable failure -> failed prefix only

8. Store data
   - Status: Partially implemented for transfer validation only.
   - `serl_transfer_validate.py` can run the existing LeRobot recorder sidecar
     through the Gazebo RL bridge recording flags.
   - Missing: `online_buffer` writer for failed policy prefixes.
   - Missing: `demo_buffer_recovery` writer for oracle recovery suffixes.

9. Offline refresh
   - Status: Not implemented.
   - Missing: critic refresh on all data.
   - Missing: BC refresh restricted to nominal plus oracle-recovery data.

10. Update Isaac randomization based on Gazebo failures
   - Status: Not implemented.
   - Missing: failure-to-randomization mapping and command/config update loop.

11. Repeat coarse Isaac <-> Gazebo loop
   - Status: Not implemented.
   - Missing: orchestration around Isaac training, Gazebo transfer validation, failure classification, data storage, offline refresh, and randomization update.

12. Final official Gazebo eval
   - Status: Not run.
   - Missing: official full-evaluation command and score artifact for a candidate hybrid-trained policy.

## Implemented Files In This Branch Area

- `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py`
  - Trains RSL-RL PPO for `AIC-Task-v0`.
  - Camera images are required. `AIC_ISAAC_DISABLE_CAMERAS=1` raises `RuntimeError`.
  - Forces `--enable_cameras` internally.
  - Validates camera sensors, camera observation terms, non-empty shapes, and computes the policy observation once so camera image-load failures raise before training proceeds.

- `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/eval.py`
  - Finite evaluator for an RSL-RL checkpoint.
  - Camera images are required and validated the same way as training.
  - Prints one `AIC_EVAL_METRICS` JSON line.

- `aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh`
  - Camera-enabled PPO smoke wrapper.
  - Defaults: `AIC_ISAAC_DISABLE_CAMERAS=0`, `RUN_NAME=stage5_ppo_smoke_camera`.
  - Produces real RSL-RL model artifacts when run; despite "smoke" naming, it performs actual training for the configured iteration count.

- `aic_utils/aic_isaac/aic_isaaclab/scripts/eval_aic_isaaclab_ppo.sh`
  - Camera-enabled finite checkpoint evaluator wrapper.
  - Requires `CHECKPOINT`.

- `aic_utils/aic_isaac/scripts/train_isaac_ppo_stage5.py`
  - Host-side wrapper for Isaac PPO.
  - Adds `--enable_cameras` and sets `AIC_ISAAC_DISABLE_CAMERAS=0`.
  - Still starts PPO from scratch or resumes an RSL-RL-native checkpoint only.

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
  - Defines scene, camera sensors, observations, actions, rewards, and randomization profile behavior.
  - Has an environment variable escape hatch in the config for disabling cameras, but the train/eval scripts now reject that mode.

## Actual Commands Executed

Setup used for all Isaac commands:

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/ubuntu/IsaacLab
ln -s /home/ubuntu/ws_aic/src/aic /home/ubuntu/IsaacLab/aic
curl -L --fail -o /tmp/aic_assets_download/Intrinsic_assets.zip https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip
unzip -q -o /tmp/aic_assets_download/Intrinsic_assets.zip -d aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task
```

This setup downloaded real NVIDIA assets and used the real Isaac Lab 2.3.2 image.
The assets are ignored by git at `aic_utils/.../Intrinsic_assets/`.

Import/registration check, smoke only:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v /home/ubuntu/ws_aic/src/aic:/workspace/isaaclab/aic \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -lc 'cd /workspace/isaaclab && \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/install_aic_task.sh && \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_import_aic_task.sh'
```

Result: success. `aic_task` imported and `AIC-Task-v0` was registered.

Camera PPO training, actual artifact-producing command:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v /home/ubuntu/ws_aic/src/aic:/workspace/isaaclab/aic \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -lc 'set -euo pipefail; cd /workspace/isaaclab; \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/install_aic_task.sh >/tmp/aic_install.log; \
    TASK_ID=AIC-Task-v0 NUM_ENVS=1 MAX_ITERATIONS=1 SEED=3 \
    RUN_NAME=stage5_ppo_camera_required_strict \
    OUTPUT_DIR=/workspace/isaaclab/aic/outputs/train/isaac_stage5_camera_required_strict \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh'
```

Result: success. This was actual PPO training, but only one iteration. It
produced real RSL-RL artifacts:

```text
outputs/train/isaac_stage5_camera_required_strict/aic_task/2026-04-30_11-29-03_stage5_ppo_camera_required_strict/model_0.pt
outputs/train/isaac_stage5_camera_required_strict/aic_task/2026-04-30_11-29-03_stage5_ppo_camera_required_strict/params/env.yaml
outputs/train/isaac_stage5_camera_required_strict/aic_task/2026-04-30_11-29-03_stage5_ppo_camera_required_strict/params/agent.yaml
outputs/train/isaac_stage5_camera_required_strict/aic_task/2026-04-30_11-29-03_stage5_ppo_camera_required_strict/events.out.tfevents...
```

Training confirmation:

- Policy observation shape: `(3154,)`.
- Camera terms: `center_rgb`, `left_rgb`, `right_rgb`, each `(1000,)`.
- Actor/critic MLP input dimension: `3154`.
- Total timesteps: `24`.
- Training time after simulator setup: about `4.24s`.

Camera checkpoint rollout/evaluator, actual Isaac checkpoint-load and step command:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v /home/ubuntu/ws_aic/src/aic:/workspace/isaaclab/aic \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -lc 'set -euo pipefail; cd /workspace/isaaclab; \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/install_aic_task.sh >/tmp/aic_install.log; \
    CHECKPOINT=/workspace/isaaclab/aic/outputs/train/isaac_stage5_camera_required_strict/aic_task/2026-04-30_11-29-03_stage5_ppo_camera_required_strict/model_0.pt \
    NUM_ENVS=1 NUM_EPISODES=1 MAX_STEPS=16 SEED=3 \
    aic/aic_utils/aic_isaac/aic_isaaclab/scripts/eval_aic_isaaclab_ppo.sh'
```

Result: success. This was an actual Isaac Lab rollout of the trained checkpoint,
but not a full episode and not a Gazebo/engine rollout.

Evaluator metrics:

```json
{
  "completed_episodes": 0,
  "num_envs": 1,
  "target_episodes": 1,
  "vector_env_steps": 16,
  "reaching_step_rate": 0.0,
  "video_recorded": false
}
```

`completed_episodes` is `0` because this was a short checkpoint-load/step smoke.
The AIC default timeout is about 6000 steps, so this command does not prove full
episode performance.

Negative camera-disabled check, intentional failure:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v /home/ubuntu/ws_aic/src/aic:/workspace/isaaclab/aic \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -lc 'set -euo pipefail; cd /workspace/isaaclab; \
    export AIC_ISAAC_DISABLE_CAMERAS=1; \
    if ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
      --task AIC-Task-v0 --headless --num_envs 1 --max_iterations 1 \
      >/tmp/camera_disabled_train.log 2>&1; then exit 1; fi; \
    grep -n "Camera images are required" /tmp/camera_disabled_train.log'
```

Result: success as a negative test. Training fails fast with:

```text
RuntimeError: Camera images are required for AIC Isaac training. Unset AIC_ISAAC_DISABLE_CAMERAS or set it to 0/false.
```

Static checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/aic_pycache python3 -m py_compile \
  aic_utils/aic_isaac/scripts/train_isaac_ppo_stage5.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/eval.py

git diff --check
```

Result: success. Host `pytest` is not installed, so the pytest file was not run
through pytest in this session. A direct wrapper assertion check confirmed
`--enable_cameras` and `AIC_ISAAC_DISABLE_CAMERAS=0` are set by the Stage 5 wrapper.

## Historical Low-Dimensional Runs

Before camera training was fixed, commit `bea623c` validated low-dimensional
training only. Those runs should not be treated as satisfying the camera-required
pipeline goal.

Historical lowdim training:

```text
outputs/train/stage5_aic_lowdim_ppo/aic_task/2026-04-30_09-08-49_stage5_aic_lowdim_ppo/model_200.pt
```

Historical lowdim evaluator result:

- Completed episodes: `4`
- Vector-env steps: `6000`
- Average reward: `-180.4537`
- Reaching episode rate: `0.0`
- Cameras were disabled.

## Known Warnings / Environment Notes

- Headless camera startup on this EC2 host is slow. Isaac can take about 3 minutes before printing the environment tables.
- Isaac logs warn that several `.glb` visual references inside `aic_unified_robot_cable_sdf.usd` cannot be opened as USD-format assets. This did not block camera feature extraction or PPO smoke training.
- The image feature extractor downloads `resnet18-f37072fd.pth` inside each fresh container unless the Torch cache is persisted.
- The one-iteration camera checkpoints are proof of wiring, not useful policies.

## Next Work

The next meaningful task is not more Isaac smoke testing. The missing pieces are
the Gazebo side of the hybrid loop:

1. Define the canonical obs/action schema and adapters for Gazebo, ACT/SERL, Isaac, and final policy.
2. Produce a new nominal Gazebo expert dataset with no-contact oracle/VLM plus CheatCode insertion.
3. Train an ACT/BC checkpoint and document the artifact path.
4. Run SERL offline pretrain on that same dataset.
5. Run longer Isaac camera PPO with `--randomization-profile heavy`, preferably from an initialized policy once checkpoint bridging exists.
6. Export/adapt the policy for instrumented Gazebo transfer validation.
7. Implement failure classification and buffer writes.
8. Implement offline refresh and the repeat Isaac <-> Gazebo loop.
9. Run final official Gazebo eval and record the score/artifacts here.
