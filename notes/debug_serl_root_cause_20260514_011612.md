# SERL Root-Cause Debug Map - 2026-05-14 01:16:12 UTC

## Official Gazebo Eval Path

- Host orchestration: `scripts/evaluate_act_checkpoints_runtime.py`.
- Container flow per checkpoint:
  1. restart `aic_eval`;
  2. start `/entrypoint.sh ground_truth:=false start_aic_engine:=false gazebo_gui:=false launch_rviz:=false`;
  3. start `ros2 run aic_model aic_model` with the selected policy and remap `/insert_cable` to an isolated action name;
  4. wait for lifecycle state `unconfigured` and the isolated action endpoint;
  5. run `ros2 run aic_engine aic_engine` with `AIC_RESULTS_DIR`, the selected engine config, `model_node_name`, and the same action remap.
- ACT-only policy module: `aic_example_policies.ros.RunACTTorchScript`, configured by `AIC_ACT_TORCHSCRIPT`, `AIC_ACT_DEVICE`, command-mode/frame, control Hz, runtime, and `AIC_ACT_N_ACTION_STEPS`.
- Offline/online SERL policy module: `aic_example_policies.ros.RunACTAdapterSERL`, configured by `AIC_SERL_CHECKPOINT`, `AIC_SERL_ACT_TORCHSCRIPT`, `AIC_SERL_DEVICE`, optional `AIC_SERL_ADAPTER_DELTA_CLIP`, optional `AIC_SERL_ACTION_CLIP`, command-mode/frame, control Hz, runtime, and `AIC_SERL_N_ACTION_STEPS`.

## ACT Runtime Inference Path

- `RunACTTorchScript.insert_cable()` receives the official `Task`, resets `AICRuntimeFeatureAssembler`, repeatedly reads `aic_model_interfaces/Observation`, calls `select_delta_action()`, clamps command components, and sends either no command, velocity, or delta-pose commands.
- State path: `RunACTTorchScript._state_vector()` -> `AICRuntimeFeatureAssembler.assemble_ros()` -> `base_state_from_ros_observation()`, with optional 40D contact/recovery features and 10D task vector.
- Image path: ROS image bytes are reshaped as RGB, resized to `(288, 256)`, converted to CHW float `[0, 1]`, and normalized with ACT safetensor stats.
- Normalization: `RunACTTorchScript._stat()` clamps std abs `<1e-8` to `1.0`; `_apply_task_vector_identity_normalization()` forces 42D/82D task-vector mean `0` and std `1`.
- ACT TorchScript output is normalized action chunks; runtime unnormalizes with `action.mean/std`, uses the first `AIC_ACT_N_ACTION_STEPS`, and executes one 6D action at a time.

## Offline SERL Runtime Inference Path

- `RunACTAdapterSERL` wraps `gazebo_rl.serl_policy.ACTAdapterSERLGazeboPolicy`.
- ROS observation is converted to a dict with controller pose/velocity/error, joint state, wrench, and RGB images.
- Task vector is inferred from official `Task` and assigned to both `policy.task_vector` and `policy.feature_assembler.task_vector`.
- `ACTAdapterSERLGazeboPolicy._obs_to_actor()` builds the same 32/72/42/82D state with `AICRuntimeFeatureAssembler` and converts images to CHW RGB `(3, 256, 288)`.
- `TorchScriptACTAdapterActor.action_components()` runs:
  `normalized obs -> TorchScript ACT -> unnormalized base_action -> adapter(state, base_action) -> clipped delta -> final_action = base_action + adapter_scale * delta`.
- `adapter_delta_clip` defaults from online adapter config, then offline `vision_offline_serl_config`, then warmstart report unless overridden by env. `action_clip` defaults similarly but is disabled for the downloaded smoke checkpoint.
- Normalization: `gazebo_rl.serl_policy.ACTRuntimeNormalizer` clamps std abs `<1e-8` to `1.0` and forces 42D/82D task-vector mean `0` and std `1`.

## Isaac Online SERL Training Path

- Host wrapper: `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`.
- Stage-C curriculum launcher: `aic_utils/aic_isaac/scripts/launch_isaac_serl_curriculum.py`; it materializes child YAMLs, shards by GPU, then invokes the host wrapper per shard.
- Isaac trainer: `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
- The wrapper forwards ACT TorchScript, offline/online checkpoint or `--act-only`, episode config dir, reward weights, randomization profile, action/frame controls, adapter clipping, diagnostics, and camera settings into the Isaac trainer.
- The trainer loads `AIC-Task-v0`, requires camera sensors, reads raw Isaac RGB tensors, optionally flips RGB channels, resizes to `(3, 256, 288)`, builds LeRobot-compatible state, runs the ACT+adapter actor, steps Isaac, appends replay tuples, updates twin critics and the adapter, and writes `metrics.jsonl`, `train_config.json`, and checkpoints.
- Isaac state path: `_isaac_lerobot_state()` builds 32D base state from `gripper_tcp`, wrist/link wrench, joints, and task vector; for 82D it inserts `ContactRecoveryFeatureComputer` features before the task vector.
- Isaac normalizer: local `ACTRuntimeNormalizer` in `serl/train.py` uses the same std clamp and 42D/82D task-vector identity normalization.

## Reward Terms

- Isaac reward implementation: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/.../mdp/rewards.py`.
- Online reward configuration: `_configure_task_geometry_rewards()` in `serl/train.py` maps CLI weights into Isaac reward table weights and multiplies by `1 / env_step_dt`.
- Main target terms:
  - `body_to_object_distance_tanh`
  - `body_to_object_distance_exp`
  - `body_to_object_distance_progress`, implemented as `clip((previous_distance - current_distance) / scale, -1, 1)`
  - `body_to_object_orientation_gated_exp`
  - `body_to_object_reaching_bonus`
  - `body_to_object_success_once_bonus`
  - `body_to_object_lateral_error`
  - `force_delta_penalty`
- Episode YAML target poses override live target-root offsets through `_episode_target_position_w()` / `_episode_target_orientation_w()` when `_aic_current_episode_by_env` is populated.

## Episode YAML Materialization And Consumption

- Materializer: `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`.
- Minimal request YAMLs are validated by `validate_request()`, sampled by `_sample_context()`, converted into child YAMLs by `_episode_config()`, and written under `episodes/`, `manifest.csv`, `task_distribution.yaml`, and `summary.json`.
- Multi-YAML curriculum: `materialize_many_episode_configs()` writes global episode YAMLs and per-GPU shard episode dirs in curriculum order.
- Isaac consumption:
  - `train_isaac_online_serl.py` sets `AIC_ISAAC_EPISODE_CONFIG_DIR`.
  - `events._load_episode_configs_from_env()` reads `episode_*.yaml`.
  - `events._episode_randomization_for_envs()` stores `_aic_current_episode_by_env`.
  - Rewards and task-vector logic read `_aic_current_episode_by_env`.

## Near-Gate Reset Logic

- Child YAML near-gate placement: `isaac_episode_configs._apply_start_near_gate()`.
- For SFP-to-NIC, the target entrance comes from `entrance_pose_world`, not the seated target point; axial/lateral offsets are applied relative to `insertion_axis_world` and a sampled perpendicular lateral direction.
- The child YAML records `start_near_gate.reset_mode=body_start_position_world`, `reset_body_name`, `body_start_position_world`, `body_start_orientation_wxyz`, achieved axial/lateral distances, target gate position, and gate axis.
- Isaac reset application: `events.reset_robot_tcp_to_episode_start()`.
- It reads current episode `scene.start_near_gate`, resolves the reset body, then runs damped IK over UR arm joints to place that body at the requested world pose and stores `_aic_tcp_reset_report_by_env` with initial/final position and orientation errors.

## RGB, State, Task, And Action Scaling

- Gazebo ACT and Gazebo SERL image paths assume RGB byte ordering. SERL policy can decode `jpeg_rgb8`, raw RGB/BGR/RGBA/BGRA dictionaries, and ROS image wrappers pass raw RGB-shaped arrays.
- Isaac image path assumes sensor output `rgb`, converts NHWC or NCHW into CHW RGB, scales to `[0, 1]`, optionally flips channels with `--swap-rgb-channels`, and resizes to `(256, 288)`.
- State order is the canonical 32D base, optional 40D contact/recovery features, then 10D task vector for 82D.
- The downloaded offline SERL checkpoint reports `state_dim=82`, `action_horizon=4`, `single_action_dim=6`, `action_dim=24`, `actor_mode=act_adapter`, `freeze_act=true`, `adapter_delta_clip=0.01`.
- Isaac executes the first 6D action from the selected ACT/SERL chunk through its IK action term. `--isaac-action-scale` defaults to `1.0`; `--fix-isaac-ik-xy-sign` defaults true and flips Isaac IK root-frame x/y translation commands to match realized TCP direction.

## 2026-05-14 Online Findings

- GPU, NVIDIA container runtime, Isaac import smoke, and camera tensor smoke are healthy on this host.
- Official Gazebo eval is partially blocked: the ACT TorchScript runtime imports and reaches policy readiness after adding Pixi `LD_LIBRARY_PATH`/`PYTHONPATH` in `scripts/evaluate_act_checkpoints_runtime.py`, but `aic_engine` timed out during lifecycle discovery with no `scoring.yaml`.
- Offline SERL checkpoint diagnostics are stable but not better than ACT on dataset imitation. Best-val has no NaNs/infs and `final_minus_act_norm ~= 0.0129`, but BC MSE worsens from ACT `8.34e-11` to SERL `6.92e-6`.
- Pure ACT Isaac chunk audits showed `isaac_action_scale=1.0` is contact-heavy; `0.5` is safer and produces positive short-horizon motion from the 6 mm near-gate start.
- The initial online near-gate experiments accidentally used the full 24-episode shard, mixing SFP near-gate, SC near-gate, and full episodes under SFP reward geometry. Later runs use `_per_request/01_sfp_near_gate_6mm` for pure SFP near-gate smoke.
- Root cause fixed in `serl/train.py`: hard adapter clamp zeroed gradients once raw adapter outputs crossed the clip; the actor then saturated and could not recover. The fix uses a straight-through clamp for training and penalizes raw adapter norm. Runtime forward actions remain clipped.
- Regularizer scaling fixed in `serl/train.py`: adapter/ACT preservation penalties were per-element MSE over 48D, making a fully clipped 1 mm adapter only `~1.25e-7`. They now use action-vector norms.
- Added `--update-every-steps` to the Isaac trainer/wrapper to reduce update-to-data ratio on tiny online replay buffers.
- Best short online setting so far:
  - ACT warm start, pure SFP 6 mm near-gate, `policy_hz=20`, `act_only_action_horizon=8`, `n_action_steps=4`, `isaac_action_scale=0.5`, `adapter_delta_clip=0.001`, `adapter_lr=1e-4`, `adapter_penalty_weight=1.0`, `act_preservation_weight=1.0`, `update_every_steps=4`.
  - Run: `outputs/debug_serl_root_cause/20260514_011612/online_isaac_runs/act_only_sfp_near_gate_scale05_stclip_lr1e4_reg1_update4_12m/2026-05-14_03-25-58_act_only_sfp_near_gate_scale05_stclip_lr1e4_reg1_update4_12m`.
  - Result: last-100 reward positive `0.0231`, last step reward `0.0957`, last-100 progress positive `0.0381`, no adapter clipping, but last-500 reward remains slightly negative `-0.00625` and force spikes remain in the final window.
- Decision: do not start the 6 hour mixed full/near-gate SFP+SC training yet. The online setting is improved but not confirmed enough; run a longer single-task near-gate confirmation first, then add SC with correct `task_family=sc_to_sc` and target body/geometry.
