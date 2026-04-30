# Hybrid Train Pipeline Status

Last updated: 2026-04-30 on branch `feat/hybrid-train`, after canonical
hybrid schema cleanup, nominal Gazebo CheatCode dataset generation, ACT smoke
training, lowdim offline SERL smoke training, direct-ACT vision offline SERL
smoke training, ACT-adapter vision offline SERL smoke training, Isaac online
SERL/SAC runs, ACT-adapter Gazebo transfer-adapter wiring, and bounded
adapter/action execution guards.

Purpose: this file is the handoff state for future Codex sessions working on the
full hybrid training pipeline. It distinguishes actual artifact-producing runs
from short smoke/adapter checks.

## Overall State

The first nominal warm-start artifacts now exist: a 10-episode accepted Gazebo
CheatCode LeRobot dataset, a 200-step ACT checkpoint, a 200-step lowdim offline
SERL checkpoint, a 200-step direct-ACT vision SERL checkpoint, and a 200-step
ACT-adapter vision SERL checkpoint trained on the same dataset. The primary
hybrid actor architecture is now ACT plus a small trainable adapter:
`obs -> ACT -> a_ACT`, then `state + a_ACT -> adapter -> delta`, with final
action `a_ACT + scale * delta`. ACT is frozen by default so SERL refines ACT
instead of overwriting it. Isaac online SERL/SAC now runs the same ACT-adapter
actor and saves real online checkpoints. Offline SERL, online Isaac SERL, and
Gazebo transfer execution now support explicit `adapter_delta_clip` and
`action_clip` guards so an unstable adapter cannot emit unbounded corrections
during validation. It is still not the full hybrid pipeline: recovery buffers,
failure-specific offline refresh, and final official Gazebo evaluation are still
incomplete or only partially represented by current utilities.

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
   - Implemented: Isaac online SERL runtime adapter consumes the ACT TorchScript
     base plus the offline/online adapter checkpoint while keeping cameras
     enabled.
   - Implemented: Gazebo transfer validator can now load the ACT-adapter SERL
     checkpoint shape through a TorchScript ACT base.
   - Implemented: Gazebo bridge IPC can opt into live RGB image payloads via
     `AIC_GAZEBO_RL_INCLUDE_IMAGES=true` / `--include-images`, using the
     existing `aic_model_interfaces/Observation` camera fields.
   - Runtime note: local `pixi run ros2 launch` cannot find package
     `aic_bringup` because `/home/ubuntu/ws_aic/install` is not present, but
     the user-created `aic_eval` distrobox/Docker container from
     `docs/getting_started.md` is available and works through
     `--sim-distrobox aic_eval`.
   - Implemented: the Gazebo runner now prepends the source
     `aic_utils/gazebo_rl` package on `PYTHONPATH` so bridge subprocesses use
     the edited source tree rather than a stale installed package.
   - Implemented: image IPC is resized to ACT input size `(288, 256)` and
     JPEG-encoded before newline-JSON transport. The policy decodes
     `jpeg_rgb8` payloads back into ACT image tensors.

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
   - Status: Complete for 200-step low-dimensional, direct-ACT vision, and
     ACT-adapter vision offline SERL smoke runs.
   - Existing `train_offline_serl.py` path remains a low-dimensional
     smoke/pretrain path and is still supported.
   - Dataset:
     `outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset`.
   - Lowdim checkpoint:
     `outputs/train/hybrid_offline_serl_nominal_n10/checkpoint_latest.pt`.
   - Offline SERL settings: `steps=200`, `batch_size=32`, `action_horizon=8`,
     `hidden_dim=256`, `num_layers=3`.
   - Checkpoint metadata: `obs_dim=32`, flattened actor `action_dim=48`,
     `action_horizon=8`, dataset `action_mode=cartesian`; normalization stats
     include `obs_mean`, `obs_std`, `action_mean`, and `action_std`.
   - Vision offline SERL added:
     `aic_utils/lerobot_robot_aic/lerobot_robot_aic/vision_offline_serl.py` and
     `aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py`.
   - Vision checkpoint:
     `outputs/train/hybrid_vision_offline_serl_nominal_n10/checkpoint_latest.pt`.
   - Vision settings: `steps=200`, `batch_size=2`, `action_horizon=8`,
     `bc_weight=1.0`, `cql_weight=0.0`, `lr=1e-4`, `device=cuda`.
   - Vision metadata: `state_dim=32`, single-step `action_dim=6`, flattened
     actor `action_dim=48`, `action_horizon=8`, cameras
     `observation.images.center_camera`, `observation.images.left_camera`, and
     `observation.images.right_camera`.
   - Vision ACT warm-start: ACT checkpoint
     `outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model`
     is reconstructed with LeRobot `ACTPolicy` and wrapped as the SERL actor.
     Warm-start report: 153 compatible trainable ACT tensors loaded,
     234 compatible ACT state tensors loaded, 51,580,806 of 51,580,854 actor
     parameters initialized from ACT, 99.99990694221542% actor-parameter
     coverage, no skipped tensors. The remaining actor parameter is the new
     global Gaussian `log_std`.
   - Latest vision metrics at step 200: `bc_loss=0.007389282342046499`,
     `critic_loss=4.0039492887444794e-05`, `actor_loss=-0.018424563109874725`,
     `q_mean=0.013945362530648708`.
   - ACT-adapter vision SERL checkpoint:
     `outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt`.
   - ACT-adapter settings: `actor_mode=act_adapter`, `freeze_act=true`,
     `steps=200`, `batch_size=2`, `action_horizon=8`, `bc_weight=1.0`,
     `adapter_penalty_weight=1e-3`, `act_preservation_weight=1e-2`,
     `smoothness_weight=0.0`, `cql_weight=0.0`.
   - ACT-adapter architecture: frozen LeRobot ACT base actor produces the
     flattened 8-step action chunk; a zero-initialized MLP adapter with input
     `observation.state` plus `a_ACT` and 98,864 parameters predicts `delta`.
     The actor has 51,679,718 parameters total, with 98,912 trainable by
     default including adapter and `log_std`; ACT trainable parameters: 0.
   - ACT-adapter warm-start report: 153 compatible ACT trainable tensors loaded,
     234 compatible ACT state tensors loaded, 51,580,806 ACT parameters loaded,
     99.80860576677296% of total actor parameters covered by ACT, no skipped
     tensors, `initial_delta_norm=0.0`, `initial_final_minus_act_norm=0.0`.
   - Latest ACT-adapter metrics at step 200:
     `bc_loss=0.009551051072776318`,
     `critic_loss=1.258803968084976e-05`,
     `actor_loss=0.005853863433003426`,
     `adapter_delta_norm=1.6950193643569946`,
     `final_minus_act_norm=1.6950193643569946`,
     `adapter_penalty=0.05986534804105759`,
     `act_preservation_loss=0.05986534059047699`.
   - Guarded few-step ACT-adapter offline run:
     `outputs/train/hybrid_vision_offline_serl_adapter_clipped_fewstep/checkpoint_latest.pt`.
   - Guarded few-step command used `steps=3`, `batch_size=2`,
     `adapter_penalty_weight=0.1`, `act_preservation_weight=1.0`,
     `adapter_delta_clip=0.05`, and `action_clip=0.05`.
   - Guarded few-step final metrics at step 3:
     `bc_loss=0.002062457147985697`,
     `critic_loss=8.358272316399962e-05`,
     `actor_loss=0.04385317116975784`,
     `raw_adapter_delta_norm=0.03374718874692917`,
     `adapter_delta_norm=0.03374718874692917`,
     `final_minus_act_norm=1.163395643234253`.
   - Note: with `action_clip` enabled, the final action may differ from raw ACT
     even when the adapter is zero because the final ACT-plus-adapter action is
     clamped. With `action_clip` disabled, the zero-initialized adapter path
     starts exactly at ACT.
   - The older lowdim `--act-checkpoint` bridge still exists and transfers only
     `model.action_head.bias` as an action-prior warm start. The new vision
     ACT-adapter path is the primary compatible ACT/SERL warm-start path.

5. Isaac Lab RL with dense reward + heavy randomization
   - Status: PPO/RSL-RL legacy smoke path validated; online SERL/SAC primary
     path is now short-run artifact-producing.
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
   - Missing: architecture-compatible transfer from the new vision SERL actor
     into the current camera PPO/RSL-RL actor. The current PPO actor is still a
     camera-feature-conditioned RSL-RL MLP, not the ACT-backed vision SERL
     actor.
   - Added online SERL/SAC design doc:
     `aic_utils/aic_isaac/docs/isaac_online_serl_design.md`.
   - Added online SERL host launcher:
     `aic_utils/aic_isaac/scripts/train_isaac_serl_stage5.py`.
   - Added Isaac Lab online SERL trainer:
     `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
   - Added ACT TorchScript export utility:
     `aic_utils/lerobot_robot_aic/scripts/export_act_torchscript.py`.
   - ACT TorchScript artifact used by Isaac:
     `outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt`.
   - The online SERL trainer loads the ACT TorchScript base, keeps Isaac camera
     sensors enabled, disables only PPO-specific ResNet observation terms,
     reads raw Isaac camera RGB tensors, resizes them to `(3, 256, 288)`, runs
     the trained adapter actor, collects replay, updates twin critics and the
     adapter, and saves an online checkpoint.
   - Actual artifact-producing online SERL run:
     `outputs/train/isaac_stage5_online_serl_adapter/2026-04-30_20-18-32_online_serl_adapter_short/checkpoint_latest.pt`.
   - Online run result: 3 Isaac steps, 2 online updates, final metrics
     `reward_mean=-0.017923442646861076`,
     `critic_loss=0.01120422501116991`,
     `actor_loss=0.08170586079359055`,
     `adapter_delta_norm=0.1511501669883728`,
     `q_mean=-0.0775182694196701`.
   - Online sanity run with 30-minute wall-time guard:
     `outputs/train/isaac_stage5_online_serl_adapter/2026-04-30_20-55-36_online_serl_adapter_sanity_300/checkpoint_latest.pt`.
   - Sanity command requested `steps=300`, `updates=100`, `batch_size=8`,
     `max_wall_time_minutes=30`; it stopped by `target_updates`, not by wall
     time, after 107 Isaac steps, 100 online updates, and 4.087678201599981
     elapsed minutes.
   - Sanity final metrics:
     `reward_mean=-0.029265476390719414`,
     `critic_loss=0.00016353523824363947`,
     `actor_loss=-0.12103446573019028`,
     `adapter_delta_norm=5.718219757080078`,
     `adapter_penalty=0.6926227807998657`,
     `act_preservation_loss=0.6926227807998657`,
     `q_mean=0.01647045835852623`.
   - 1k guarded online SERL run:
     `outputs/train/isaac_stage5_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt`.
   - 1k command requested `steps=1000`, `updates=1000`, `batch_size=8`,
     `max_wall_time_minutes=30`, `adapter_penalty_weight=0.01`, and
     `act_preservation_weight=0.1`; it stopped by `max_steps`, not wall time,
     after 1000 Isaac steps, 993 online updates, and 5.902379688616626 elapsed
     minutes. Updates are 993 because the replay buffer needed the first 7
     transitions before `batch_size=8` sampling could begin.
   - 1k final metrics:
     `reward_mean=-0.025411013513803482`,
     `critic_loss=0.015914855524897575`,
     `actor_loss=-3.8696117401123047`,
     `adapter_delta_norm=24.33574676513672`,
     `adapter_penalty=12.340703964233398`,
     `act_preservation_loss=12.340703964233398`,
     `q_mean=3.136911392211914`.
   - Concern: adapter correction grew substantially during the 1k run. Before
     treating the checkpoint as a policy candidate, reduce adapter learning rate
     and/or increase ACT-preservation regularization. Bounded delta/action
     clipping has now been added to offline SERL, Isaac online SERL, and Gazebo
     transfer execution; this prevents unsafe actions but does not make the 1k
     checkpoint a good policy candidate.
   - Missing: long heavy-randomization training to convergence.
   - Missing: insertion reward based on semantic cable-tip/port frames; current optional insertion terms use approximate object roots.

6. Gazebo transfer validation in instrumented mode
   - Status: Lowdim SERL validation path implemented; ACT-adapter SERL loader,
     Gazebo live-image IPC, and short online Gazebo SERL training implemented;
     true vision transfer scoring still needs to be run.
   - Added:
     `aic_utils/gazebo_rl/scripts/serl_transfer_validate.py`.
   - Added:
     `aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`.
   - Added:
     `aic_utils/gazebo_rl/gazebo_rl/serl_train.py`.
   - Added:
     `aic_utils/gazebo_rl/scripts/gazebo_serl_train.py`.
   - The validator loads an offline lowdim SERL checkpoint, converts Gazebo
     bridge observations into the canonical 32D low-dimensional state, emits the
     first 6D action from the chunked SERL actor, parses Gazebo scoring output,
     and writes `transfer_validation_summary.json`.
   - Added ACT-adapter policy kind:
     `--policy-kind act_adapter_serl --act-torchscript <act_policy_ts_cuda.pt>`.
     This reconstructs the TorchScript ACT base plus adapter from an offline or
     online ACT-adapter SERL checkpoint and emits the first 6D action from the
     final chunk.
   - The ACT-adapter Gazebo policy requires live camera images by default.
     `serl_transfer_validate.py` now defaults `--include-images` to true for
     `--policy-kind act_adapter_serl` unless `--allow-zero-images` is explicitly
     requested. The bridge serializes `center_image`, `left_image`, and
     `right_image` from the ROS `Observation` message into compact base64 image
     payloads keyed as the ACT camera observations.
   - `--allow-zero-images` exists only for explicit adapter interface validation
     and must not be used as real transfer scoring.
   - Actual short ACT-adapter Gazebo transfer command:
     `pixi run python aic_utils/gazebo_rl/scripts/serl_transfer_validate.py --policy-kind act_adapter_serl --checkpoint outputs/train/isaac_stage5_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt --workspace-dir . --sim-distrobox aic_eval --device cuda --max-steps 3 --per-trial-timeout-sec 300 --ground-truth true --gazebo-gui false --launch-rviz false --output-dir outputs/gazebo_rl/serl_transfer_validation/act_adapter_3step_latest`.
   - Actual short transfer summary:
     `outputs/gazebo_rl/serl_transfer_validation/act_adapter_3step_latest/transfer_validation_summary.json`.
   - Result: completed 3 real Gazebo steps with `include_images=true` and
     `allow_zero_images=false`; elapsed 44.19805851899946 seconds,
     `total_reward=-0.02`, `action_norm_mean=17.205318417729618`,
     `action_norm_max=17.244454467997244`,
     `adapter_delta_norm_mean=54.55885442097982`.
   - Score/classification: `classification=no_score`, `total_score=None`,
     because this was a 3-step max-steps wiring run and no terminal scoring file
     was produced.
   - Added bounded execution guards:
     `--adapter-delta-clip 0.05` and `--action-clip 0.05` are now available in
     `train_vision_offline_serl.py`, `train_isaac_serl_stage5.py`, Isaac
     `aic_isaaclab/scripts/serl/train.py`, and `serl_transfer_validate.py`.
     Isaac online SERL and Gazebo transfer default to `0.05` for both guards.
   - Actual short clipped ACT-adapter Gazebo transfer command:
     `pixi run python aic_utils/gazebo_rl/scripts/serl_transfer_validate.py --policy-kind act_adapter_serl --checkpoint outputs/train/isaac_stage5_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt --workspace-dir . --sim-distrobox aic_eval --device cuda --max-steps 3 --per-trial-timeout-sec 300 --ground-truth true --gazebo-gui false --launch-rviz false --adapter-delta-clip 0.05 --action-clip 0.05 --output-dir outputs/gazebo_rl/serl_transfer_validation/act_adapter_clipped_3step_latest`.
   - Actual clipped transfer summary:
     `outputs/gazebo_rl/serl_transfer_validation/act_adapter_clipped_3step_latest/transfer_validation_summary.json`.
   - Clipped result: completed 3 real Gazebo steps with live images; elapsed
     44.14124419899963 seconds, `total_reward=-0.02`,
     `action_norm_mean=0.10949402778003287`,
     `action_norm_max=0.10949986228346104`,
     `adapter_delta_norm_mean=0.3464101552963257`,
     `raw_adapter_delta_norm_mean=54.15252685546875`.
   - Clipped score/classification: `classification=no_score`,
     `total_score=None`, because this was still only a 3-step max-steps wiring
     run. The clamp bounded execution as intended; the raw adapter correction is
     still too large for a candidate checkpoint.
   - Online Gazebo ACT-adapter SERL training dry-run command:
     `pixi run python aic_utils/gazebo_rl/scripts/gazebo_serl_train.py --checkpoint outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt --output-dir outputs/gazebo_rl/online_serl/adapter_dry_run --device cuda --dry-run`.
   - Dry-run result: loaded the real offline ACT-adapter checkpoint and ACT
     TorchScript with `state_dim=32`, `action_dim=48`,
     `single_action_dim=6`, `action_horizon=8`, `include_images=true`,
     `adapter_delta_clip=0.05`, and `action_clip=0.05`.
   - Actual short online Gazebo SERL training command:
     `pixi run python aic_utils/gazebo_rl/scripts/gazebo_serl_train.py --checkpoint outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt --output-dir outputs/gazebo_rl/online_serl/adapter_2step_latest --workspace-dir . --sim-distrobox aic_eval --device cuda --max-episodes 1 --max-steps 2 --updates 1 --batch-size 1 --max-minutes 10 --per-trial-timeout-sec 300 --ground-truth true --gazebo-gui false --launch-rviz false --adapter-delta-clip 0.05 --action-clip 0.05`.
   - Actual online Gazebo SERL training result:
     `outputs/gazebo_rl/online_serl/adapter_2step_latest/checkpoint_latest.pt`.
     It completed 1 real Gazebo step and 1 adapter/critic update in
     44.723284710998996 seconds, then stopped because the requested update count
     was reached.
   - Online Gazebo SERL final metrics:
     `reward=-0.01`,
     `critic_loss=0.0003967539523728192`,
     `actor_loss=0.06374824792146683`,
     `raw_adapter_delta_norm=1.6684852838516235`,
     `adapter_delta_norm=0.329450786113739`,
     `final_action_norm=0.3193478584289551`,
     `final_minus_act_norm=1.1606016159057617`.
   - Reload check: `gazebo_serl_train.py --dry-run` successfully loaded the
     Gazebo-trained checkpoint from
     `outputs/gazebo_rl/online_serl/adapter_2step_latest/checkpoint_latest.pt`.
   - Missing: run a longer scored ACT-adapter Gazebo transfer validation after
     training a policy candidate with intrinsically bounded adapter corrections.
   - Missing: Isaac PPO checkpoint export into the Gazebo policy bridge. PPO is
     no longer the primary compatible hybrid path, so this is a legacy/baseline
     concern rather than the main transfer path.

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
5. Implement the Isaac online SERL/SAC loop around the ACT-adapter actor.
6. Keep PPO/RSL-RL for smoke/baseline checks, not as the primary hybrid-transfer path.
6. Export/adapt the policy for instrumented Gazebo transfer validation.
7. Implement failure classification and buffer writes.
8. Implement offline refresh and the repeat Isaac <-> Gazebo loop.
9. Run final official Gazebo eval and record the score/artifacts here.
