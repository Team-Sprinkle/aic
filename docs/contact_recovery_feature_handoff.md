# Contact / Recovery Feature Handoff

## Context

This note is for a follow-up Codex session that will have access to the datasets and should iterate on contact-related feature logic.

The current training stack is:

- `ACT` as the base visuomotor policy
- offline SERL-style pretraining / adaptation
- online SERL in Isaac / Gazebo

The goal is to add **deployable, causal features** into `observation.state` so that the policy can react to contact and recovery situations while still using `n_obs_steps=1`.

The main design rule is:

- prefer **sensor-measurement features** that do not assume the policy behaved correctly
- treat **memory / bookkeeping features** as potentially brittle, because they depend on a contact-event detector and runtime state logic

This matters because expert trajectories can have perfect recovery bookkeeping, but the deployed learned policy may not behave exactly like the expert.

## Reference Logic: `feat/agent-teleop`

Before implementing or revising the feature logic, read the recovery logic on `feat/agent-teleop`.

Primary references:

- `aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py`
- `aic_teacher_official/aic_teacher_official/expert_generator/ft_guard.py`
- `scripts/generate_expert_trajectories.py`

Important observations from that branch:

- recovery is explicit and bounded, with staged logic like:
  - contact detect
  - backoff
  - wait for release
  - realign
  - retry
- retries are capped
- runtime events and recovery-related traces already exist
- the branch is a strong reference for how contact and recovery are intended to work in expert mode

Important data caveat:

- some debug artifacts are sampled coarsely for analysis
- for actual feature generation, use the **raw dataset rows / parquet rows**, not coarse debug summaries

## Why We Are Adding These Features

We want the model to have access to:

- direct evidence that contact changed sharply
- memory of the most recent contact event
- progress since contact, especially whether the robot has backed off

This should let the policy learn recovery-related behavior without requiring raw observation history inside ACT.

At the same time, the next session should be careful about train/test mismatch:

- `force_delta` and `torque_delta` are relatively safe
- `contact_count`, `steps_since_last_contact`, and similar features depend on event-detection logic and can be wrong if the detector is brittle

## Feature Groups

### 1. Safe Sensor-Measurement Features

These are considered reliable because they are just measurements derived from sensor history.

#### `force_delta`

Definition:

- compare force across two time points or two time windows
- keep the 3D delta vector

Conceptually:

```python
force_delta = force_curr - force_prev   # shape (3,)
```

This yields:

- `force_delta_x`
- `force_delta_y`
- `force_delta_z`

#### `force_delta_norm`

Definition:

```python
force_delta_norm = norm(force_delta)
```

This is the scalar magnitude of the force change.

#### `torque_delta`

Definition:

```python
torque_delta = torque_curr - torque_prev   # shape (3,)
```

This yields:

- `torque_delta_x`
- `torque_delta_y`
- `torque_delta_z`

#### `torque_delta_norm`

Definition:

```python
torque_delta_norm = norm(torque_delta)
```

Note: the original note said `roquet_delta_norm`; interpret that as `torque_delta_norm`.

## Time-Window Issue

This is the main open problem for the next session.

Two known extremes:

### A. Single-step delta

```python
force_delta = force[t] - force[t - 1]
```

Pros:

- simple
- causal
- sharp contact onset may show up clearly

Cons:

- can be noisy
- may only spike for one step
- may be brittle across sim/runtime differences

### B. Non-overlapping windowed delta

```python
force_curr = median(force[t-k+1:t+1])
force_prev = median(force[t-2*k+1:t-k+1])
force_delta = force_curr - force_prev
```

Pros:

- more robust to noise

Cons:

- one physical collision can be counted as multiple contact events
- the larger the window, the more repeated counting risk there is
- recent-contact information may be okay, but older contact bookkeeping becomes inaccurate

## Current Recommendation

For now, treat the following as stable feature outputs:

- `force_delta`
- `force_delta_norm`
- `torque_delta`
- `torque_delta_norm`

But treat the exact detector implementation as unresolved. The next session should use dataset access to strengthen:

- single-count semantics for one contact event
- robustness to noise
- causal logic
- transferability between expert data and deployed runtime

## 2. Memory / Bookkeeping Features

These are useful, but they are more brittle because they depend on **event logic**, not just raw sensing.

### `contact_count_clipped`

Definition:

- number of detected contact events so far
- clip to a small max value if needed

Conceptually:

```python
if new_contact_event:
    contact_count += 1

contact_count_clipped = min(contact_count, CONTACT_COUNT_MAX)
```

Risk:

- a single collision may be counted multiple times if the detector keeps firing across adjacent windows

Requirement for next session:

- make a single physical contact count only once unless there is a clear release-and-recontact

### `steps_since_last_contact_norm`

Definition:

```python
steps_since_last_contact = current_step - last_contact_step
steps_since_last_contact_norm = clip(steps_since_last_contact / STEPS_SINCE_SCALE, 0.0, 1.0)
```

Risk:

- only meaningful if `last_contact_step` itself is detected reliably

### `in_contact_or_loaded`

Goal:

- indicate that the robot is still pressing / loaded / stuck after contact

Proposed logic:

1. when a contact event is detected, store a smoothed force reference
2. while running, compare current smoothed force to that reference
3. if they remain close, the robot is likely still loaded

Conceptually:

```python
if new_contact_event:
    contact_force_ref = median_force_over_250ms(current_force_history)

current_force_smooth = median_force_over_250ms(current_force_history)

in_contact_or_loaded = (
    norm(current_force_smooth - contact_force_ref) < CONTACT_FORCE_STILL_THRESHOLD
)
```

Current threshold idea:

- `CONTACT_FORCE_STILL_THRESHOLD = 0.5`

Open question for next session:

- whether `250 ms` is the right smoothing horizon
- whether this should use force only, or also torque

### `last_contact_force_drop`

Definition:

- magnitude of the sharp force change that triggered the most recent contact

Conceptually:

```python
if new_contact_event:
    last_contact_force_drop = force_delta_norm
```

This is a memory feature, not a fresh measurement.

### `backoff_distance_since_contact`

Definition:

- displacement norm in `base_link` from contact pose to current pose

Conceptually:

```python
if new_contact_event:
    contact_position_base = current_position_base

backoff_distance_since_contact = norm(current_position_base - contact_position_base)
```

Purpose:

- tell the policy how far it has moved away since the last contact

### `lateral_distance_since_contact`

Definition:

- XY displacement in the TCP frame since contact

Conceptually:

```python
delta_tcp = current_pose_in_contact_tcp_frame.translation - contact_pose_tcp_frame.translation
lateral_distance_since_contact = norm(delta_tcp[:2])
```

Purpose:

- capture side-to-side realignment since contact

### `axial_distance_since_contact`

Definition:

- Z displacement in the TCP frame since contact

Conceptually:

```python
delta_tcp = current_pose_in_contact_tcp_frame.translation - contact_pose_tcp_frame.translation
axial_distance_since_contact = abs(delta_tcp[2])
```

Purpose:

- capture how much the robot moved along the insertion axis since contact

## Important Warning About Memory Features

These features are **not guaranteed to stay correct** just because they were correct in expert mode.

In expert mode:

- the controller is deterministic
- contact / backoff / recovery are explicit
- bookkeeping can be exactly right

In deployed learned-policy mode:

- the model may react earlier or later
- the model may never back off
- the model may oscillate
- the model may contact obstacles in a different order than the expert

So memory features must be computed from **runtime logic that will also exist at test time**, not just from expert-only hidden state.

## Recommended Boundary

### Safe to rely on strongly

- `force_delta`
- `force_delta_norm`
- `torque_delta`
- `torque_delta_norm`

### Useful, but must be validated carefully

- `contact_count_clipped`
- `steps_since_last_contact_norm`
- `in_contact_or_loaded`
- `last_contact_force_drop`
- `backoff_distance_since_contact`
- `lateral_distance_since_contact`
- `axial_distance_since_contact`

## What the Next Codex Session Should Do

The next session should use dataset access to iterate on the detector and memory logic.

Primary goals:

1. Make one physical contact count only once.
2. Make contact detection robust to noise.
3. Keep the logic causal.
4. Avoid using teacher-only hidden state as final model input.
5. Validate that the same feature logic can exist at test time.

Suggested workflow:

1. Read the `feat/agent-teleop` recovery logic carefully.
2. Extract raw force / torque / pose traces from real dataset rows.
3. Compare candidate contact detectors:
   - single-step delta
   - median-window delta
   - detector with refractory / cooldown logic
   - detector with trigger / hold / clear logic
4. Visualize or tabulate when each detector fires.
5. Check whether a single collision becomes multiple counted contacts.
6. Decide which features are stable enough to become final policy input.

## Suggested Direction for Stronger Detection Logic

The next session should probably move toward an **event detector** rather than recomputing “contact” independently every step.

Conceptually:

```python
if contact_trigger and not detector_in_refractory:
    new_contact_event = True
    last_contact_step = current_step
    contact_count += 1
    detector_in_refractory = True

if release_condition:
    detector_in_refractory = False
```

This kind of logic is likely needed to prevent repeated counting from a single collision.

## Final Implementation Rule

If a feature cannot be computed consistently at test time, it should not become a required model input.

Teacher-only or expert-only signals are okay for:

- analysis
- debugging
- pseudo-label generation

But the final deployed policy input should use only signals that can be recomputed online.
