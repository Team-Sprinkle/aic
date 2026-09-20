# Scoring

Each trial is scored using a tiered scoring system that sums up to 100 points.
There are 3 trials in the Qualification phase so the maximum score for a
submission is 300 points.

## Scoring Tiers Overview

| Tier | Name | Description |
|------|------|-------------|
| Tier 1 | Model Validity | Prerequisite check that model loads and conforms to expectations |
| Tier 2 | Performance & Convergence | Quantitative metrics for motion quality |
| Tier 3 | Cable Insertion | Primary objective - successful or partial insertion verified |

## Geometry and distance terminology

Use these names when reading scores or diagnosing a rollout:

| Name | Meaning |
|------|---------|
| TCP | Robot tool-center-point frame on the gripper. Its pose is not the pose of the plug tip. |
| Plug tip | The leading end of the grasped connector (`sfp_tip_link` for SFP). In the SFP asset, this link is offset 23.65 mm along the module's local axis from `sfp_module_link`; this is not necessarily the TCP-to-tip distance. |
| Port entrance / opening | Mouth of the selected port, represented by `sfp_port_*_link_entrance`. |
| Port reference / seated target | `sfp_port_*_link`, farther inside the connector's insertion axis than the entrance. The full-insertion event is verified separately by the scorer. |
| Plug-to-port distance | The scorer's reported distance between plug and target port references. It is not TCP tracking error, lateral misalignment alone, or distance to the opening. |

For the SFP/NIC asset, the entrance is **45.8 mm** from the port reference
along the insertion axis. The collision cage is approximately **48.72 mm**
deep. A connector going from the mouth toward full seating therefore travels
roughly **4.6–4.9 cm along the port axis**, subject to the actual contact and
scoring geometry; 45.8 mm is a frame offset, not a measured insertion stroke.
The NIC card's main PCB collision box is **56 × 145 × 1.6 mm** and is mounted
upright, so its long dimension is roughly **14.5 cm**. These are asset
dimensions, not tolerances for insertion. See the
[NIC Card Mount](../aic_assets/models/NIC%20Card%20Mount/model.sdf),
[NIC Card](../aic_assets/models/NIC%20Card/model.sdf), and
[SFP Module](../aic_assets/models/SFP%20Module/model.sdf) models.

In the first nine completed September 18 paired world-model final trials, the
scorer reported an **initial plug-to-port distance of about 15–19 cm** (where
its path-efficiency message was available). This is a scene-specific starting
distance to the port reference, **not** a measured TCP-to-port distance and
not the distance from plug tip to the opening. A terminal score message of
"0.04–0.05 m from the port" can therefore put the plug near the mouth in
axial distance, since the mouth and port reference differ by 45.8 mm. The
single Euclidean distance cannot establish whether the tip is centered,
oriented correctly, or inside. One trial even received a *partial insertion*
message at approximately 0.05 m; other nearby-distance trials received *no
insertion*. Inspect tip-to-entrance axial depth, lateral error, orientation,
and the official insertion event separately. See the
[paired score records](../outputs/experiments/2026-09-18_dreamer60_pilot/artifacts/world_tcp_delta_final_worker_v1/progress.json).

## Tier 1: Model Validity (Prerequisite)

A sanity check to ensure the submission loads and runs without errors.

- The model must be able to successfully activate the submitted policy and respond to the `InsertCable` action request. The submitted policy must also send valid commands to the robot arm controller via `MotionUpdate` (target position/velocities) or `JointMotionUpdate` (target joint states).
- The policy must comply with all behavioral requirements defined in [Challenge Rules](./challenge_rules.md#aic_model)
- Submissions failing this check will not be scored

| Outcome | Score |
|---------|-------|
| Validation passed | 1 |
| Validation failed | 0 |

## Tier 2: Performance & convergence

Quantitative metrics measuring the quality of the robot's motion during task execution.

### Trajectory smoothness (0-6 points)

Measures the smoothness of the end effector trajectory. Lower jerk values
indicate smoother, more controlled motion. Jerk is only accumulated when the
arm is moving (speed > 0.01 m/s), so stationary periods do not dilute the
average. Only awarded if either the task is completed successfully, or the
final position of the plug is within close proximity to the target port
(Tier 3 score > 0).

- **Metric**: Time-weighted average of linear jerk magnitude (m/s³), computed via a Savitzky–Golay filter (local quadratic polynomial fit to velocity over a 15-sample window)
- **Scoring**: Inversely proportional to jerk
  - Jerk = 0 m/s³ → 6 points (maximum)
  - Jerk ≥ 50 m/s³ → 0 points (minimum)
  - Linear interpolation between thresholds
- **Not awarded**: 0 points if the final position of the plug is outside the max acceptable distance of the target port (Tier 3 score <= 0).

### Task duration (0-12 points)

Rewards faster task completion. Only awarded if either the task is completed
successfully, or the final position of the plug is within close proximity to
the target port (Tier 3 score > 0).

- **Metric**: Elapsed time from task start to task end
- **Scoring**: Inversely proportional to duration
  - Duration ≤ 5 seconds → 12 points (maximum)
  - Duration ≥ 60 seconds → 0 points (minimum)
  - Linear interpolation between thresholds
- **Not awarded**: 0 points if the final position of the plug is outside the max acceptable distance of the target port (Tier 3 score <= 0).

### Trajectory efficiency (0-6 points)

Measures the total distance traveled by the end effector during task execution.
Shorter, more direct paths score higher. Only awarded if either the task is
completed successfully, or the final position of the plug is within close
proximity to the target port (Tier 3 score > 0).

- **Metric**: Cumulative Euclidean distance of end-effector positions (meters)
- **Scoring**: Inversely proportional to total path length
  - Path length ≤ initial plug-port distance → 6 points (maximum)
  - Path length ≥ 1 m + initial plug-port distance → 0 points (minimum)
  - Linear interpolation between thresholds
- The minimum path length (for a perfect score) is set dynamically to the initial Euclidean distance between the plug and port at the start of the trial
- **Not awarded**: 0 points if the final position of the plug is outside the max acceptable distance of the target port (Tier 3 score <= 0).

### Insertion force penalty (0 to -12 points)

Penalizes excessive force during the insertion process to encourage gentle manipulation.
The force sensor reading is tared at startup, so the baseline is close to 0 N.

- **Force threshold**: 20 N
- **Duration threshold**: 1 second
- **Penalty**: -12 points if force exceeds threshold for longer than the duration threshold
- **No penalty**: If no excessive force is detected or excessive force is within duration threshold

### Off-Limit contact penalty (0 to -24 points)

Penalizes collisions with restricted areas of the environment (enclosure or task board).

- **Penalty**: -24 points if any contact with off-limit entities is detected
- **No penalty**: If no prohibited contacts occur

## Tier 3: Task Success

The primary objective verifying successful cable insertion. Scoring uses a two-step approach that rewards both full insertion and partial progress toward the port.

### Successful insertion (-12 to 75 points)

If the cable connector is fully inserted into the **correct** target port, verified via contact sensors:

| Outcome | Score |
|---------|-------|
| Correct port insertion | 75 |
| Wrong port insertion | -12 |

### Partial insertion and proximity (0-50 points)

When full insertion is not detected, the score is based on how close the plug is to the port at task completion:

- **Partial insertion** (38-50 points): If the plug is inside a bounding box between the port entrance and the bottom of the port (within a 5 mm x-y tolerance), the score is proportional to insertion depth. Deeper insertion scores higher.
- **Proximity** (0-25 points): If the plug is not inside the port, the score is inversely proportional to the max acceptable distance from the port. The max distance is set to half the distance between the initial position of the plug and the port.
  - At the port entrance → 25 points (maximum)
  - Outside of max distance → 0 points (minimum)
  - Linear interpolation between thresholds

## Total Score Calculation

```
Total Score = Tier 1 + Tier 2 + Tier 3
```

Where:
- **Tier 1**: 0 or 1 point
- **Tier 2**: Sum of smoothness (0-6), duration (0-12), efficiency (0-6), and penalties (force: 0 to -12, contacts: 0 to -24)
- **Tier 3**: Insertion success (up to 75), or partial insertion / proximity score (up to 50)
- **Maximum score per trial**: 100 points (1 + 6 + 12 + 6 + 75)

## Final Ranking

The final ranking is determined by cumulating scores across all trials, combining the quantitative performance metrics from Tier 2 and the task success score from Tier 3.

## See Also

For reproducible examples that exercise each scoring tier, see the [Scoring Test & Evaluation Guide](./scoring_tests.md).
