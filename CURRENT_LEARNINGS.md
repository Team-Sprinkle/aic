# Current Learnings

This file summarizes what was learned so far, what was fixed, what remains blocked, and what the most likely explanations are.

## High-level status

What is clearly working:

- the live Gazebo joint-target plugin loads
- the service path is correct for the actual world
- a real joint-target request changes real robot state
- a reduced live env one-step slice can now complete and return:
  - joint positions
  - TCP pose
  - derived geometry
  - reward
  - terminated / truncated

What is not yet complete:

- full parity against the current toolkit path
- final reward alignment with the official score calculation path
- a robust low-latency training transport
- a fair benchmark on the intended final architecture
- live end-to-end validation from a built workspace / eval-session context

## Main issues discovered

### 1. World-name / service-path mismatch

Initial issue:

- validation assumptions expected `/world/default/joint_target` or `/world/aic_world/joint_target` inconsistently
- live runs showed that the actual world path must be derived from the running Gazebo world

Fix:

- `aic_gazebo/src/JointTargetPlugin.cc` was updated to derive the world name from the actual Gazebo `World` entity first

Result:

- validated live service path: `/world/aic_world/joint_target`

### 2. Real bridge command path works

Important result:

- sending a real request to `/world/aic_world/joint_target` with `model_name=ur5e` moved the real joints and changed real end-link pose

Meaning:

- the bridge itself is real and useful
- this is not a fake / teleport-only path

### 3. CLI topic reads are unreliable for live idle/change-driven topics

This became the central blocker.

Observed behavior:

- repeated `gz topic -e -n 1` on live topics like:
  - `/world/aic_world/state`
  - `/world/aic_world/pose/info`
  - `/world/aic_world/state_async`
  - `/world/aic_world/dynamic_pose/info`
  - `/world/aic_world/scene/info`
  - `/world/aic_world/scene/graph`
  can block for long periods or time out

Interpretation:

- these topics are not reliable as one-shot pull interfaces when sampled by spawning a fresh CLI process for each observation
- some are effectively change-driven from the perspective of the ad hoc subscriber

Most important practical finding:

- if a subscriber is attached first and a world step is forced afterward, `/world/aic_world/state` can produce a sample reliably

This strongly suggests:

- repeated one-shot CLI topic invocations are the wrong transport for RL training
- a persistent subscriber or direct transport API will likely be required

### 4. `pose/info` is especially problematic

Observed:

- `pose/info` often hung even when `state` could be made to return

Response:

- a fallback was added to derive moving-link poses from `/world/aic_world/state` instead of depending on `pose/info`

### 5. State topic contains useful live data

Important discovery:

- `/world/aic_world/state` contains enough structured component data to recover:
  - joint positions
  - pose-like RPY + XYZ values for tracked links

Observed ids:

- `ati/tool_link` = `79`
- `wrist_3_link` = `50`
- `tabletop` from earlier `pose/info` observation = `32`

Observed moving-link pose component type:

- `10918813941671183356`

Observed joint position component type:

- `8319580315957903596`

Meaning:

- a state-only live observation path is feasible, at least for a narrow tracked-entity slice

### 6. Pre-step observation dependency was a bug for the live env path

Original env issue:

- `joint_position_delta` translation tried to read the current observation before sending the first command
- that immediately hit the unreliable topic-read path

Response:

- the live client was changed to seed the first delta step from known home joint positions
- cached / remembered joint positions are now used afterward

Meaning:

- first-step live control no longer depends on a fragile idle observation read

### 7. Extra `/world/.../control` call after joint action was harmful in live mode

Observed:

- after sending a joint-target command, the client still called `/world/aic_world/control` with `multi_step`
- that call could hang even though the world was already running in real time

Response:

- the live path was changed to skip the control RPC after joint / pose actions and instead allow the running world to advance naturally

Meaning:

- for the current live bringup, the explicit control call is not always appropriate after every action

### 8. Timing / observation alignment is still the main parity problem

Current reduced parity traces differ mostly because they are not sampling the same settled post-command instant.

This is not primarily a world-name or request-format issue anymore.

It is now mainly:

- sampling-time mismatch
- transport latency / staleness
- possible settle-time mismatch

### 9. Reduced scoring alignment path now exists in the env bridge

New result:

- the real bridge now supports an opt-in `official_tier3` reward mode

Meaning:

- the env can now compute a Tier-3-style proximity score for the stable tracked
  pair slice using the same scoring shape as the official toolkit
- this is still only a stable comparable subset
- full official score reuse still requires the complete official inputs:
  - insertion events
  - off-limit contacts
  - FT history
  - controller-state velocity history

Interpretation:

- this is a legitimate step toward reward alignment
- it does not remove the need to reuse or integrate the official full scoring
  path for final evaluation-parity claims

## Why the local Docker build took so long

Observed:

- `aic_eval_local` built successfully
- image size later showed about `9.37GB`
- build took roughly on the order of an hour

Current best explanation:

- this local path appears to be building a large workspace inside Docker, including a substantial `colcon` build
- by contrast, the main-branch README experience may have seemed fast because it was:
  - pulling a prebuilt image, or
  - reusing cached layers / previously built artifacts

So the slow build is plausibly expected for a fresh local source build, but this should still be verified directly against:

- `docker/aic_eval/Dockerfile`
- the official README / launch flow
- whether the documented path expects an image pull rather than a full local rebuild

This investigation is now explicitly included in `CONTINUE_PROMPT.md`.

## Current measured timings

These are not final benchmark numbers, but they are useful signals.

Reduced env one-step slice:

- elapsed about `28.88s`

Reduced native direct one-step slice:

- elapsed about `9.35s`

Interpretation:

- current live env path is much too slow for RL training in its present transport form
- the dominant cost is likely transport / observation machinery, not the underlying physics alone

## Current local blocker for live validation

In the current shell context:

- `/home/ubuntu/ws_aic/install` does not exist
- `ros2` is not available on `PATH`
- `gz` is not available on `PATH`

Meaning:

- unit and fake-bridge validation can proceed
- live Gazebo parity, live eval-session parity, and real latency benchmarking
  cannot be completed from this exact local runtime context without first
  building or entering a prepared environment

## Current modified files that matter most

- `aic_gazebo/src/JointTargetPlugin.cc`
- `aic_utils/aic_gazebo_env/aic_gazebo_env/gazebo_client.py`
- `aic_utils/aic_gazebo_env/scripts/live_env_one_step.py`
- `aic_utils/aic_gazebo_env/scripts/live_native_one_step.py`
- `aic_utils/aic_gazebo_env/scripts/live_native_parity.py`
- `aic_utils/aic_gazebo_env/scripts/live_joint_bridge_parity.py`
- `aic_utils/aic_gazebo_env/scripts/live_phase2_probe.py`
- `aic_utils/aic_gazebo_env/scripts/live_pose_ids.py`
- `aic_utils/aic_gazebo_env/scripts/live_state_ids.py`

## What the first prompt file already covers

`CONTINUE_PROMPT.md` already mentions the important issues below:

- official docs / launch flow must be reread
- reward must align with official scoring
- latency and direct transport alternatives must be evaluated
- parity must be built on the smallest stable slice first
- service-path fix is done
- real live joint bridge proof is done
- topic-read unreliability is a key blocker
- state-only fallback idea is already noted
- minimal env / native slice results are recorded
- benchmark is incomplete

What was missing before this update:

- explicit investigation task for why `aic_eval_local` took so long to build

That is now added to `CONTINUE_PROMPT.md`.

## Recommended next questions

1. What is the official reward / scoring implementation on the main branch, and how should RL reward reuse it?
2. What is the official eval-session launch / middleware context, and how should parity be run inside it?
3. Should the live RL path continue using Python + `gz` CLI at all, or move to a persistent direct API?
4. Can a direct C++ / native Gazebo transport layer provide:
   - multi-tick stepping
   - stable state access
   - lower latency
   - easier parity with official scoring
