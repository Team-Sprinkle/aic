# Example Policies

This package contains baseline policy implementations that demonstrate different approaches to the cable insertion task. These examples serve as reference implementations and starting points for developing your own policies.

> [!NOTE]
> **Prerequisites:** Before running these policies, ensure you have the evaluation environment running. See [Getting Started](../docs/getting_started.md) for setup instructions.
>
> **Command Format:**
> - If using the **container workflow** (recommended): Launch with `distrobox enter -r aic_eval -- /entrypoint.sh [parameters]`
> - If **built from source**: Launch with `ros2 launch aic_bringup aic_gz_bringup.launch.py [parameters]`
> - Run policies with `pixi run ros2 run` (Pixi workspace) or `ros2 run` (native ROS 2)

---

## Available Policies

### 1. WaveArm - Minimal Example

![Wave Arm Policy](../../media/wave_arm_policy.gif)

A minimal example showing how to implement the `insert_cable()` callback and issue motion commands to the arm. This policy simply moves the robot arm back and forth in a waving motion without attempting to solve the task.

**Purpose:** Demonstrates the basic Policy API structure.

**Launch the evaluation environment:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

**Source:** [`WaveArm.py`](./aic_example_policies/ros/WaveArm.py)

---

### 2. NoOp - Teleop Assist Policy

A passive policy that accepts tasks but does not command robot motion. This is
useful when you want `aic_engine` trial sequencing while controlling the arm via
keyboard teleoperation.

**Purpose:** Keep tasks alive for teleop data collection across multi-trial engine configs.

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.NoOp
```

**Source:** [`NoOp.py`](./aic_example_policies/ros/NoOp.py)

---

### 3. CheatCode - Ground Truth Policy

![Cheat Code Policy](../../media/cheat_code_policy.gif)

A "cheating" solution that uses the TF transformation tree provided by the simulation when `ground_truth:=true` is set at launch time. This policy uses the poses of the plug and port to calculate target poses to send to `aic_controller`.

**Purpose:** Useful for training and debugging. Ground truth data will not be available during official evaluation.

**Launch simulation *with ground truth*:**
```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

**Source:** [`CheatCode.py`](./aic_example_policies/ros/CheatCode.py)

---

### 4. DiversifiedCheatCode - Ground Truth Data Collection Policy

A trajectory-diversified variant of `CheatCode` that still uses ground-truth TF, but randomizes approach and insertion style each episode for richer demonstration data.

**Purpose:** Collect varied, high-success trajectories for imitation learning / behavior cloning datasets.

**Launch simulation *with ground truth*:**
```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

For randomized setup across many episodes, generate an engine config first:
```bash
cd ~/ws_aic/src/aic
python generate_random_trials_config.py \
  --output ./outputs/configs/random_trials_10.yaml \
  --num_trials 10 \
  --seed 42
```

Then launch bringup with that config:
```bash
cd ~/ws_aic/src/aic
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/jk/ws_aic/src/aic/outputs/configs/random_trials_10.yaml
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DiversifiedCheatCode
```

**Optional deterministic sampling (repeatable trajectory styles):**
```bash
AIC_DIVERSIFIED_SEED=42 pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DiversifiedCheatCode
```

**What is randomized per episode:**
- Time profile: `linear`, `smoothstep`, or `min_jerk`
- Approach timing: control loop `dt` and total approach duration
- Approach geometry: zero to two intermediate XY/Z waypoints
- Micro-jitter: low-frequency sinusoidal XYZ offsets (small amplitude)
- Insertion strategy: `constant`, `staged`, or `peck` descent
- Correction gains: integrator windup bound and XY integral gain

**Key implementation details:**
- File: [`DiversifiedCheatCode.py`](./aic_example_policies/ros/DiversifiedCheatCode.py)
- Main entrypoint: `DiversifiedCheatCode.insert_cable()`
- Parameter sampling: `_sample_trajectory_params()`
- Approach interpolation: `_run_interpolation_segment()`
- Descent styles: `_run_diversified_descent()`

**Data-collection tuning guidance:**
- For higher success rate, narrow randomization ranges (smaller waypoint and jitter amplitudes).
- For more trajectory diversity, widen approach duration/profile and waypoint ranges gradually.
- Keep randomization bounded; extreme lateral offsets or aggressive descent can reduce insertion success.
- Log episode seed + sampled parameters alongside trajectories so training can condition on style if needed.

---

### 5. RunACT - ACT Policy

![Run ACT Policy](../../media/run_act_policy.gif)

A proof-of-concept implementation of a [LeRobot ACT](https://huggingface.co/docs/lerobot/en/act) (Action Chunking with Transformers) policy available on [HuggingFace](https://huggingface.co/grkw/aic_act_policy). This policy was trained on an NVIDIA RTX A5000 machine using `lerobot-train` with default parameters, on a small dataset collected using `lerobot-record` as explained in [`lerobot_robot_aic`](../aic_utils/lerobot_robot_aic/README.md#recording-training-data).

You may need to modify `pixi.toml` in order to run `lerobot` with your hardware setup. See [Troubleshooting](../docs/troubleshooting.md#nvidia-rtx-50xx-cards-not-supported-on-pytorch-version-locked-in-pixi). 

**Purpose:** Demonstrates integration of a trained neural network policy for the cable insertion task.

**Launch the evaluation environment:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunACT
```

**Source:** [`RunACT.py`](./aic_example_policies/ros/RunACT.py)

---

### 6. RunMIP - MIP Policy (from `much-ado-about-noising`)

A diffusion-based policy integration for AIC using a MIP checkpoint trained in the separate
`much-ado-about-noising` repository.

**Purpose:** Run a trained MIP policy inside `aic_model` with AIC observation/action interfaces.

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunMIP
```

**Source:** [`RunMIP.py`](./aic_example_policies/ros/RunMIP.py)

#### Required artifacts in this repo

`RunMIP.py` loads artifacts from:
`aic_example_policies/aic_example_policies/assets/mip`

Required files:
- `model_latest.pt`
- `configs/task/aic_lerobot_image_state.yaml`
- `configs/network/_base.yaml`
- `configs/network/mlp.yaml`
- `configs/optimization/default.yaml`
- `configs/log/default.yaml`

#### How these are sourced from `much-ado-about-noising`

Copy from `much-ado-about-noising` into this repo:

```bash
cd ~/ws_aic/src/aic
mkdir -p aic_example_policies/aic_example_policies/assets/mip/configs/{task,network,optimization,log}

cp /home/jk/much-ado-about-noising/logs/models/model_latest.pt \
  aic_example_policies/aic_example_policies/assets/mip/

cp /home/jk/much-ado-about-noising/examples/configs/task/aic_lerobot_image_state.yaml \
  aic_example_policies/aic_example_policies/assets/mip/configs/task/
cp /home/jk/much-ado-about-noising/examples/configs/network/_base.yaml \
  aic_example_policies/aic_example_policies/assets/mip/configs/network/
cp /home/jk/much-ado-about-noising/examples/configs/network/mlp.yaml \
  aic_example_policies/aic_example_policies/assets/mip/configs/network/
cp /home/jk/much-ado-about-noising/examples/configs/optimization/default.yaml \
  aic_example_policies/aic_example_policies/assets/mip/configs/optimization/
cp /home/jk/much-ado-about-noising/examples/configs/log/default.yaml \
  aic_example_policies/aic_example_policies/assets/mip/configs/log/
```

#### Runtime path overrides

- `AIC_MIP_ASSETS_DIR`: Override where `RunMIP` searches for `model_latest.pt` and configs.
- `MIP_DATASET_ROOT`: Optional root used to resolve relative `dataset_path` from task config.

Example:
```bash
AIC_MIP_ASSETS_DIR=/abs/path/to/assets/mip \
MIP_DATASET_ROOT=/abs/path/to/datasets \
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunMIP
```

#### Dependency note

`RunMIP` requires the `mip` Python package in the Pixi environment (for example via git dependency in `pixi.toml`).

---

## Scoring Examples

For expected scoring results and reproducible test commands for each policy, see the [Scoring Test & Evaluation Guide](../../docs/scoring_tests.md).
