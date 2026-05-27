# Add Hybrid ACT/SERL Training Pipeline With Task-Conditioned Data Manifests

This PR adds the next hybrid-training pass for AIC, covering YAML-driven
dataset generation, task-conditioning metadata, ACT training, ACT-backed SERL
training, Hydra configs, runtime smoke utilities, and validation docs.

The most important user-facing change is the dataset generation layout and
metadata sidecars.

## Dataset Generation Changes

Dataset generation now uses YAML request files and writes outputs into a
structured hierarchy:

```text
{root_dir}/{task_family}/{policy}/{count_label}/n{target_accepted_trajectories}__{suffix}/
```

Example:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n200__first_batch/
outputs/trajectory_datasets/sc_to_sc/cheatcode/sc_ports_2/n200__first_batch/
```

Each generated run directory contains:

```text
request.yaml
engine_config.yaml
trials/*.yaml
raw_dataset/
accepted_dataset/
scores/
logs/
generation_summary.json
manifests/
```

New `manifests/` sidecars expose task metadata without changing the native
LeRobot dataset schema:

```text
manifests/attempts.csv
manifests/accepted.csv
manifests/episode_task_metadata.jsonl
```

These include the per-trial/per-episode task fields needed for training:

- `task_family`
- `target_port_index`
- `target_card_index`
- `target_card_valid`
- `plug_type`, `plug_name`
- `port_type`, `port_name`
- `target_module_name`
- `cable_type`
- one-hot task columns
- `task_vector`

The canonical task vector is fixed-size 10D:

```text
[task_family_sfp_to_nic, task_family_sc_to_sc,
 target_port_0, target_port_1,
 target_card_0, target_card_1, target_card_2, target_card_3, target_card_4,
 target_card_valid]
```

Examples:

```text
sfp_to_nic, card 3, port 1 -> [1,0, 0,1, 0,0,0,1,0, 1]
sc_to_sc, port 0           -> [0,1, 1,0, 0,0,0,0,0, 0]
```

For SC-to-SC, the target-card vector is all zeros and `target_card_valid=0`.

Also fixed/clarified:

- SFP-to-NIC target port now randomizes across `sfp_port_0` and `sfp_port_1`
  when unspecified.
- Explicit target card/port rails remain present even for small counts.
- `cable_type` is treated as derived physical consistency metadata:
  - `sfp_to_nic -> sfp_sc_cable`
  - `sc_to_sc -> sfp_sc_cable_reversed`
- LeRobot `raw_dataset/` and `accepted_dataset/` remain schema-compatible.
  Task info is joined through manifests or an optional materialized
  task-conditioned dataset.

## Training / Model Changes

Adds canonical task-conditioning support for ACT and SERL:

- `task_encoding.py` defines and validates the 10D task vector.
- `task_metadata.py` loads manifest CSV/JSONL metadata and joins task vectors by
  episode.
- ACT can train on task-conditioned data by appending the 10D vector to
  `observation.state`.
- For DDP/multi-GPU ACT training, task-conditioned datasets can be materialized
  before training.
- Offline SERL supports `include_task_vector=true`.
- ACT-backed SERL uses ACT as an actor/action prior only.
- Critics/value functions initialize from scratch unless loading a valid
  RL/SERL critic checkpoint.
- `critic_init=act` is rejected conceptually.

## Hydra / Config Changes

Adds Hydra config organization under:

```text
configs/train/
configs/model/
configs/data/
configs/hardware/
configs/experiment/
```

Includes configs for:

- ACT BC training
- offline SERL
- ACT-adapter SERL
- online Gazebo SERL
- 1-GPU and multi-GPU hardware selection
- HF converted SFP-to-NIC dataset training

Training outputs follow a structured run layout and save resolved configs, git
info, hardware selection, and task encoding schema.

## Runtime / Evaluation Changes

Adds rootless Docker support for Gazebo RL runtime smoke on the Knuth server,
where `aic_eval` is a persistent rootless Docker container rather than a
distrobox container.

New runtime runner options include:

```text
--sim-docker-container
--docker-host
--workspace-container
--host
--port
```

Also adds `docs/pr_validation_commands.md`, which lists the commands to run
before opening PRs, including both standard distrobox and Knuth rootless-Docker
forms.

## Docs Added / Updated

Major docs include:

```text
docs/hybrid_model_architecture_review.md
docs/pr_validation_commands.md
docs/gazebo_online_serl_status.md
docs/offline_serl_pretrain.md
docs/isaac_online_rl.md
HYBRID_TRAIN_PIPELINE_STATUS.md
aic_utils/lerobot_robot_aic/README.md
```

## Validation Performed

Passed:

```bash
pixi install
```

```bash
pixi run python -m compileall aic_teacher_official aic_example_policies aic_utils aic_model
```

```bash
pixi run python -m pytest aic_teacher_official/test/test_official_teacher_pipeline.py -q
```

```bash
pixi run python -m pytest aic_model/test/test_policy_delta_pose.py -q
```

```bash
pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_generate_trajectory_dataset.py -q
```

```bash
pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_hydra_configs.py -q
```

```bash
pixi run python -m pytest aic_utils/gazebo_rl/test -q
```

Real smoke training passed:

- ACT 2-step training produced a LeRobot checkpoint.
- ACT-backed SERL 2-step training produced `checkpoint_latest.pt` and confirmed
  scratch critic initialization.

Runtime Gazebo smoke got past the previous rootless Docker/distrobox mismatch,
but final runtime validation was blocked because the shared `aic_eval` container
was already occupied by an active dataset-generation job.
