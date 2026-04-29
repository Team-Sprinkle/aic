# Gazebo RL Pipeline

This document explains how the `aic_utils/gazebo_rl` path works. It is the low-throughput, high-fidelity RL path: it drives the existing Gazebo, ROS, `aic_engine`, `aic_model.Policy`, controller, and scoring stack instead of trying to turn Gazebo into a pure synchronous simulator.

Isaac Lab remains the high-throughput training environment. Gazebo RL is for validating policies against the challenge stack and for sim-to-sim adaptation where fidelity matters more than rollout volume.

## High-Level Flow

The core boundary is the policy loaded by `aic_model`. The trainer does not talk directly to Gazebo. Instead, it starts a local IPC server, launches the existing simulation/evaluation stack, and loads a bridge policy into `aic_model`.

```text
trainer / GazeboRLEnv
  |
  | localhost newline-delimited JSON IPC
  v
gazebo_rl.bridge_policy.GazeboRLBridgePolicy
  |
  | get_observation(), move_robot(), send_feedback()
  v
existing aic_model.Policy API
  |
  v
existing Gazebo + ROS + aic_engine + aic_controller + scoring stack
```

The bridge policy is loaded like any other `aic_model` policy:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=gazebo_rl.bridge_policy.GazeboRLBridgePolicy
```

The environment side sends six-dimensional relative TCP delta actions:

```python
action = [dx, dy, dz, droll, dpitch, dyaw]
```

Actions are clipped conservatively before they reach the controller:

```python
from gazebo_rl.action import delta_tcp_action_from_array

delta = delta_tcp_action_from_array(action)
print(delta.delta_position_xyz)
print(delta.delta_quaternion_xyzw)
```

## Main Commands

Smoke rollout with random actions:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_smoke.py \
  --sim-distrobox <your_eval_container> \
  --max-steps 5 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

Short training proof:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py \
  --sim-distrobox <your_eval_container> \
  --max-iterations 5 \
  --max-minutes 5 \
  --max-steps 25 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

Roll out a saved checkpoint:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_rollout.py \
  --checkpoint outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt \
  --sim-distrobox <your_eval_container> \
  --max-steps 25 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

Roll out a saved checkpoint and record LeRobot trajectory/video outputs using the existing recorder:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_rollout.py \
  --checkpoint outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt \
  --sim-distrobox <your_eval_container> \
  --max-steps 25 \
  --record-lerobot \
  --record-root outputs/gazebo_rl/rollouts/latest/lerobot_dataset \
  --record-video \
  --record-fps 30 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

The recorder is a sidecar ROS process. It passively subscribes to ROS topics and writes the LeRobot dataset outside the bridge policy control loop, so file and video writes are not inserted into the action path.

## Runtime Sequence

1. `GazeboRLEnv` creates an `IPCServer` on localhost.
2. `GazeboRLRunner` starts the simulation/evaluation stack.
3. If `--record-lerobot` is set, `GazeboRLRunner` starts `aic-policy-recorder`.
4. `GazeboRLRunner` starts `aic_model` with `GazeboRLBridgePolicy`.
5. `GazeboRLBridgePolicy.insert_cable()` connects back to the trainer IPC server.
6. The bridge repeatedly:
   - calls `get_observation()`
   - converts the ROS observation/task/TF state into a JSON-friendly dict
   - sends an `observation` IPC message
   - waits for an `action` IPC message
   - clips/converts the action into a relative TCP command
   - calls `move_robot()`
   - sleeps using sim-time-aware policy sleep
7. The environment returns `(obs, reward, terminated, truncated, info)` to training code.
8. The trainer updates the tiny policy and saves a checkpoint.
9. Scoring output is parsed from `scoring.yaml` when available.

## File-by-File Guide

### `gazebo_rl/bridge_policy/GazeboRLBridgePolicy.py`

This is the bridge inside the existing `aic_model.Policy` boundary. It is the only part that directly implements the policy loaded by `aic_model`.

Key idea:

```python
class GazeboRLBridgePolicy(Policy):
    def insert_cable(self, task, get_observation, move_robot, send_feedback) -> bool:
        connection = self._connect(task)
        while self._step_count < max_steps:
            obs_msg = get_observation()
            obs = observation_to_dict(obs_msg, task=task, tf_buffer=..., ground_truth=...)
            connection.send("observation", {"observation": obs})
            message = connection.recv(timeout_sec=action_timeout_sec)
            self._send_delta_action(move_robot, message.payload.get("action"))
            self.sleep_for(command_dt_sec)
```

It keeps ROS/Gazebo ownership in the existing stack. The bridge does not train, score, or record videos. It only translates between `aic_model.Policy` callbacks and trainer IPC messages.

### `gazebo_rl/gym_env.py`

This is the Gym-like environment used by smoke, training, and checkpoint rollout code.

Key idea:

```python
from gazebo_rl.gym_env import GazeboRLEnv

env = GazeboRLEnv(
    workspace_dir=".",
    sim_distrobox="aic_eval",
    max_steps=25,
    ground_truth=True,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 0])
env.close()
```

`reset()` starts processes and waits for the first real observation from `GazeboRLBridgePolicy`. `step()` sends one action to the bridge and waits for the next observation or `done`.

### `gazebo_rl/runner.py`

This manages subprocesses and launch commands:

- simulation/evaluation launch
- `aic_model` launch
- optional `aic-policy-recorder` launch
- environment variables for the bridge
- cleanup

Key idea:

```python
runner = GazeboRLRunner(
    GazeboRLRunnerConfig(
        workspace_dir=Path("."),
        sim_distrobox="my_box",
        max_steps=25,
        record_lerobot=True,
    )
)

runner.start()
runner.close()
```

If `sim_distrobox` is provided, the runner uses that exact user-created distrobox name:

```bash
distrobox enter -r --no-tty <container> -- /entrypoint.sh ...
```

If `sim_distrobox` is omitted, it uses the local pixi ROS launch path.

### `gazebo_rl/ipc.py`

This is the small localhost IPC layer between trainer/env and bridge policy. It uses Python stdlib sockets and newline-delimited JSON.

Key idea:

```python
from gazebo_rl.ipc import IPCServer, connect_with_retry

server = IPCServer(host="127.0.0.1", port=0)
client = connect_with_retry("127.0.0.1", server.port)
conn = server.accept()

conn.send("observation", {"step_count": 0})
msg = client.recv(timeout_sec=1.0)
```

Supported message types are `hello`, `observation`, `action`, `done`, and `error`.

### `gazebo_rl/observation.py`

This converts `aic_model_interfaces.msg.Observation` plus task and optional TF data into plain Python data.

Key idea:

```python
from gazebo_rl.observation import observation_to_dict

obs_dict = observation_to_dict(
    observation_msg,
    task=task,
    step_count=step_count,
    tf_buffer=tf_buffer,
    ground_truth=True,
)
```

It includes:

- step count and sim time
- joint names, positions, velocities, effort
- gripper state when available
- wrist wrench
- controller TCP pose/reference/velocity/error when available
- task fields
- optional oracle TF fields such as TCP pose, plug pose, target port pose, and relative plug-to-port vector

Missing fields are tolerated and converted to zeros or `None`.

### `gazebo_rl/action.py`

This validates, clips, and converts actions.

Key idea:

```python
from gazebo_rl.action import delta_tcp_action_from_array

delta = delta_tcp_action_from_array([0.01, 0, 0, 0, 0, 0])
assert delta.clipped_action[0] == 0.003
```

Limits:

- translation: `0.003 m` per step per axis
- rotation-vector components: `0.03 rad` per step per axis

It also converts the rotation-vector action into an `xyzw` quaternion.

### `gazebo_rl/score_parser.py`

This parses scoring output produced by the existing evaluation stack. It does not recreate scoring.

Key idea:

```python
from gazebo_rl.score_parser import score_from_scoring_yaml, dense_training_reward

score = score_from_scoring_yaml("outputs/gazebo_rl/results/iter_000")
reward = dense_training_reward(terminal=False)
```

For current training:

- each non-terminal step returns `-0.01`
- terminal reward returns `total_score / 100.0` if a score is available
- otherwise terminal reward is `0.0`

### `gazebo_rl/train.py`

This contains the current short training proof. It is intentionally minimal.

Key idea:

```python
from gazebo_rl.train import TinyPolicy

policy = TinyPolicy(seed=0)
action = policy.act(obs, explore=True)
loss = policy.update(transitions)
policy.save(Path("outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt"))
```

The current `TinyPolicy` is a small MLP when PyTorch is available:

```python
Linear(64, 32) -> Tanh -> Linear(32, 6) -> Tanh
```

This is not PPO, SAC, or a full policy-gradient implementation. It collects real transitions and performs a small optimizer update that makes the network reproduce the sampled actions it just took, lightly weighted by reward magnitude.

### `gazebo_rl/rollout.py`

This loads a saved checkpoint and rolls it out through the same real Gazebo environment. It can optionally start the existing LeRobot recorder sidecar.

Key idea:

```python
from gazebo_rl.train import TinyPolicy
from gazebo_rl.gym_env import GazeboRLEnv

policy = TinyPolicy(seed=0)
policy.load(Path("outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt"))

obs, _ = env.reset()
action = policy.act(obs, explore=False)
obs, reward, terminated, truncated, info = env.step(action)
```

It writes a rollout summary JSON and, when recording is enabled, a LeRobot dataset.

### `scripts/gazebo_rl_smoke.py`

This runs a short real rollout with random actions. It is the fastest end-to-end check that IPC, bridge policy, Gazebo, ROS, controller commands, and rewards are flowing.

Key idea:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_smoke.py --max-steps 5
```

### `scripts/gazebo_rl_train_short.py`

This is the CLI wrapper around `gazebo_rl.train.main()`.

Key idea:

```python
from gazebo_rl.train import main

main()
```

### `scripts/gazebo_rl_rollout.py`

This is the CLI wrapper around `gazebo_rl.rollout.main()`.

Key idea:

```python
from gazebo_rl.rollout import main

main()
```

### `test/`

The tests are unit tests that do not require Gazebo. They cover:

- action shape validation, clipping, and quaternion sanity
- observation conversion on fake/minimal messages
- scoring YAML parsing
- IPC send/receive
- runner command generation and recorder integration

Run them with:

```bash
pixi run python -m pytest aic_utils/gazebo_rl/test -q
```

## Outputs

Training writes:

```text
outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt
outputs/gazebo_rl/run_summary.json
outputs/gazebo_rl/results/...
```

Checkpoint rollout writes:

```text
outputs/gazebo_rl/rollouts/<name>/rollout_summary.json
outputs/gazebo_rl/rollouts/<name>/results/...
```

Recorded checkpoint rollout writes a LeRobot dataset:

```text
outputs/gazebo_rl/rollouts/<name>/lerobot_dataset/data/...
outputs/gazebo_rl/rollouts/<name>/lerobot_dataset/meta/...
outputs/gazebo_rl/rollouts/<name>/lerobot_dataset/videos/...
```

Example recorded videos from a successful test run:

```text
outputs/gazebo_rl/rollouts/test_record/lerobot_dataset/videos/observation.images.left_camera/chunk-000/file-000.mp4
outputs/gazebo_rl/rollouts/test_record/lerobot_dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
outputs/gazebo_rl/rollouts/test_record/lerobot_dataset/videos/observation.images.right_camera/chunk-000/file-000.mp4
```

## FAQ

### Why did the tip move toward the target if rewards are sparse?

It should be treated as coincidence or incidental behavior, not evidence of learned insertion. The proof run used sparse rewards and only one short training iteration. The policy did not receive a meaningful target-seeking learning signal.

The observation includes real state, and the MLP maps that state to actions, but after such a short run the movement is mostly from random initialization plus a small update toward actions the policy already sampled.

### What was the initial policy?

Random. If PyTorch is available, `TinyPolicy` starts as a randomly initialized MLP:

```python
Linear(64, 32) -> Tanh -> Linear(32, 6) -> Tanh
```

During training, actions include exploration noise:

```python
action = model(obs) * 0.002 + normal_noise(scale=0.001)
```

During checkpoint rollout, exploration is off unless `--explore` is passed.

### What algorithm is used?

The current algorithm is a proof-of-plumbing optimizer loop, not a standard RL algorithm.

It:

1. starts a random tiny policy
2. collects real transitions from Gazebo
3. updates the MLP to reproduce the actions it just took
4. saves a checkpoint

The current loss is:

```python
pred = model(obs) * 0.002
loss = ((pred - actions) ** 2 * (1.0 + rewards.abs())).mean()
```

This is closest to self-behavior-cloning on sampled actions. It proves that real rollouts, rewards, gradients, and checkpointing work. It is not expected to solve insertion.

### Does it update gradients after the full episode or along the way?

After the rollout. The training loop collects transitions for the episode and then calls `policy.update(...)` once for that batch.

```python
for _ in range(args.max_steps):
    action = policy.act(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    transitions.append((obs, action, reward))

loss = policy.update(transitions[-max(1, real_steps):])
```

### Why was the video short?

The recorded validation rollout used `--max-steps 5`. That was intentional to verify recording quickly. For longer videos, increase `--max-steps`.

### How long does each step take?

Measured on this machine with a real no-record Gazebo rollout:

```text
reset() wall time:        23.69 s
25 env.step() calls:       2.19 s total
average step wall time:    0.0877 s / step
full episode wall time:   36.32 s
```

The bridge command period is configured as `0.05` sim-time seconds by default, but wall-clock step time was about `0.06-0.14 s` in the measured run.

### How long would 5 episodes take?

With the current implementation, each episode starts and stops the evaluation stack. A 25-step episode measured around `36-39 s` end-to-end. Five episodes would therefore be roughly:

```text
5 * 36-39 s = about 3.0-3.5 minutes
```

Recording video adds extra finalization and encoding time.

### Why did the 25-step episode stop?

It stopped because `--max-steps 25` was configured. That value is passed to the environment and to the bridge policy through `AIC_GAZEBO_RL_MAX_STEPS`.

The bridge loop exits when the step counter reaches the cap:

```python
while self._step_count < max_steps:
    ...
else:
    self._connection.send("done", {"reason": "max_steps", "step_count": self._step_count})
```

It did not stop because insertion succeeded or because training converged.

### Does recording affect trajectory quality?

The intended design avoids putting recording writes in the policy action path. The rollout recorder is the existing `aic-policy-recorder` process. It subscribes to ROS topics and writes a LeRobot dataset as a sidecar.

That means the bridge policy still only does observation IPC, action IPC, controller command, and sim-time sleep. Heavy dataset/video work is not performed inside `GazeboRLBridgePolicy.insert_cable()`.

### Where are trajectory details and videos?

When `--record-lerobot` is used, trajectory rows and metadata are under:

```text
<record-root>/data/
<record-root>/meta/
```

Videos are under:

```text
<record-root>/videos/
```

For example:

```text
outputs/gazebo_rl/rollouts/test_record/lerobot_dataset/data/chunk-000/file-000.parquet
outputs/gazebo_rl/rollouts/test_record/lerobot_dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
```

### Is `aic_eval` required?

No. `aic_eval` is not a toolkit resource. It is only a local user-created distrobox container name. Pass whichever container name you created:

```bash
--sim-distrobox <your_eval_container>
```

Or omit `--sim-distrobox` to use the local pixi launch path.

## Practical Next Steps

The current pipeline proves the real Gazebo/ROS/evaluation path. To turn this into a useful learner, likely next steps are:

1. keep the same bridge and runner
2. add longer rollouts and reusable simulation sessions to reduce reset overhead
3. use teleop/teacher LeRobot datasets for imitation learning warm start
4. replace the proof optimizer with PPO, SAC, or another real on-policy/off-policy method
5. add denser task progress signals only if they are derived from existing observations/scoring and do not reimplement evaluation scoring
