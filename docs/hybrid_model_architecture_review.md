# Hybrid Model Architecture Review

Date: 2026-05-01
Branch: `feat/hybrid-train`

This review reflects the current branch code, not a proposed architecture from memory.

## ACT Path

`aic_utils/lerobot_robot_aic/scripts/train_act_policy.py` is a thin LeRobot CLI wrapper. It does not define a custom ACT module. It validates the local LeRobot dataset schema, optionally materializes a derived task-conditioned dataset, then launches `lerobot-train --policy.type=act`.

ACT receives whatever features are present in the LeRobot dataset. For the AIC recorder this means low-dimensional `observation.state`, image observations, and `action`. `chunk_size`, `n_action_steps`, `n_obs_steps`, batch size, optimizer LR, and device are passed through as LeRobot CLI overrides. Action dimensionality is inferred by LeRobot from the dataset action feature.

Task information is not hardcoded into the ACT model config. The supported task-conditioning path is `--task-conditioning append-state`, which creates a derived local LeRobot dataset with the canonical 10D task vector appended to `observation.state`. Native `raw_dataset/` and `accepted_dataset/` remain unchanged. With no task-conditioning flag, ACT sees only the original dataset features and therefore does not receive `task_family`, `target_port_index`, `target_card_index`, or `target_card_valid`.

## Offline SERL Path

`aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py` uses custom low-dimensional SERL-style code from `lerobot_robot_aic.offline_serl`, not LeRobot ACT internals.

The actor is `GaussianActor(obs_dim -> flattened action_dim)`. The critic is a twin-Q setup: each critic consumes concatenated observation and flattened action and outputs a scalar Q value. `action_horizon` is represented by flattening future action chunks, so `effective_action_dim = single_step_action_dim * action_horizon`.

`OfflineRLTransitionDataset` normalizes observations and actions with dataset mean/std. When `--include-task-vector` is set, the 10D task vector is joined by `episode_index` from `manifests/accepted.csv` or an explicit `--task-metadata` path, then appended to `observation.state` before normalization. Checkpoints and run summaries record `include_task_vector`, `task_vector_dim`, `task_encoding_schema`, `original_obs_dim`, and `effective_obs_dim`.

`--act-checkpoint` is not a full model transfer. The current supported transfer is `--act-warmstart-mode action_head_bias`, which copies the ACT action head bias into the SERL actor mean head, repeated across action horizon. This is labeled as an actor/action bias prior only. The critic is initialized from scratch by default.

## Online SERL / Gazebo Path

`aic_utils/gazebo_rl/gazebo_rl/serl_policy.py` implements the online ACT-adapter SERL policy. The ACT base is loaded as TorchScript and frozen. A trainable MLP adapter predicts a flattened delta action. The final action chunk is:

`final_action = clipped(ACT_base_action + clipped_adapter_delta)`

`aic_utils/gazebo_rl/gazebo_rl/serl_train.py` trains the adapter and twin image/state critics from replay gathered through `GazeboRLEnv`. Task context can be passed through explicit CLI fields or `--task-context-json`; the same 10D vector is appended to low-dimensional state before the actor/critic. If a checkpoint expects task-conditioned state and no task context is supplied, the policy fails clearly on state-dimension mismatch.

Critics are scratch by default. `--critic-init checkpoint` loads a SERL/RL critic checkpoint. `--critic-init act` is rejected because ACT has no value-function semantics. `--critic-only-steps` and `--actor-update-delay` support critic-only warmup and delayed actor/adapter updates so the adapter does not immediately chase untrained Q-values.

## Isaac / RSL-RL Path

`aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py` uses Isaac Lab and RSL-RL PPO. It is a separate actor-critic training stack. Policy/value architecture and observation groups are controlled by Isaac Lab/RSL-RL task and agent configs. Checkpoint initialization is handled by the RSL-RL runner and optional branch warmstart utilities, not by the LeRobot ACT wrapper.

In this branch, the Isaac path does not consume the canonical AIC task vector by default. Policy and value share the environment observation design selected by the Isaac task config. Any ACT/SERL checkpoint transfer into this path must be treated as policy-side initialization unless a valid RL critic/value checkpoint is explicitly being loaded into matching value-function modules.

## Value-Function Conclusion

ACT should be treated as an actor/action prior only. It predicts actions from observations; it does not estimate returns, Q values, or state values. Therefore:

- ACT must not initialize the SERL critic or RSL-RL value function.
- Critic/value modules initialize from scratch unless loading a valid SERL/RL critic checkpoint.
- `critic_init=act` is invalid and should fail clearly.
- For online ACT-adapter SERL, use critic-only warmup and delayed actor updates when starting from ACT so actor updates are based on trained Q estimates.

## Canonical Task Vector

The shared task-conditioning vector is fixed at 10 dimensions:

- `task_family_onehot`: `[1, 0]` for `sfp_to_nic`, `[0, 1]` for `sc_to_sc`
- `target_port_index_onehot`: two entries for port 0 or 1
- `target_card_index_onehot`: five entries for SFP-to-NIC target card 0..4
- `target_card_valid`: one entry, `1` for SFP-to-NIC and `0` for SC-to-SC

Examples:

- SFP-to-NIC, card 3, port 1: `[1,0, 0,1, 0,0,0,1,0, 1]`
- SC-to-SC, port 0: `[0,1, 1,0, 0,0,0,0,0, 0]`

The canonical implementation is `lerobot_robot_aic.task_encoding`. Dataset generation writes human-readable task fields and task vectors into sidecar manifests. Model code remains task-agnostic: it only sees a fixed-size numeric vector appended to state.

## Run Organization

New Hydra configs follow this convention:

`outputs/train/${task_family}/${dataset_tag}/${model_family}/${stage}/${now:%Y%m%d_%H%M%S}_${run_name}`

Each run should contain resolved config, git info, task encoding schema when applicable, metrics, latest checkpoint or LeRobot output, run summary, and eval summary if evaluation ran.

Task generation details are dataset metadata, not model config. Model config describes tensor architecture. Training config describes paths, checkpoints, devices, optimization, and smoke/eval settings.
