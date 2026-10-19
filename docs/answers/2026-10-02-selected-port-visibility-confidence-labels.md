# Labels for a selected-port pose, visibility, and confidence model

**Yes.** I recommend that the near-port model predict (1) the selected opening's pose when it can identify it, (2) **visibility/identifiability for each camera**, and (3) **calibrated confidence that the final pose is accurate enough to use**. These are different outputs.

## One training example

Each row is one synchronized robot observation and one requested target:

```text
Inputs:
  three RGB images
  task code: SC or SFP, selected card if SFP, selected port 0/1
  measured TCP pose (for camera geometry and a TCP-relative answer)

Labels:
  selected port's true 3D pose relative to the measured TCP
  selected opening's pixel location in each camera, where identifiable
  identifiable visibility in each camera: [yes/no, yes/no, yes/no]
  pose-label-valid mask: yes only when a visual pose can reasonably be learned
```

The simulator's port geometry supplies **training labels only**. It is never passed to the model during deployment. The image point and pose labels must refer to **the port named by the task code**, not whichever port is closest or clearest.

For example:

| Images | Requested target | Per-camera identifiable visibility | Pose training target |
| --- | --- | --- | --- |
| Both SC ports clear | SC port 1 | `[yes, yes, yes]` for **port 1** | Port 1's true TCP-relative pose. |
| Same images | SC port 0 | `[yes, yes, yes]` for **port 0** | Port 0's pose instead. |
| Port 0 clear; port 1 hidden by cable | SC port 1 | `[no, no, no]` | **No pose loss**; train it to reject the request. |
| Port 1 clear in left and center, hidden in right | SC port 1 | `[yes, yes, no]` | Pose label is valid if those two views make it identifiable; no right-camera pixel loss. |
| Requested SFP card absent or target offscreen | That card/port | `[no, no, no]` | **No pose loss**; train rejection. |

“Visible” here means **enough of the requested opening can be recognized to locate it**, not merely that its ideal 3D point projects inside the image. A half-occluded opening may still be identifiable in one view. We can optionally record a visible fraction, but the useful control label is whether each view supports a reliable selected-port estimate.

## How to obtain those labels

The existing dataset already has projected opening coordinates and projected-in-frame flags for all candidates. Those flags are **not true visibility**: a port can be inside an image yet hidden by a cable or card. We would need additional labels. In Gazebo, a practical route is to render or derive selected-port instance/depth masks and compare them with the opening projection, then audit a sample of full and partial occlusions visually. Previous auxiliary depth capture had issues, so its labels must be validated before use. For each scene, we can also create **same-image, different-task** examples: select another port, including one that is hidden or absent. That directly trains the model not to substitute a visible neighbor.

For visible examples, use the true port and measured TCP poses to form the TCP-relative pose target and project selected-port keypoints into each calibrated camera. Apply the pose and pixel losses only where the selected target is genuinely identifiable. Hidden-target rows still train visibility and rejection. Split by complete scene/reset group, not randomly by adjacent frames.

## What the confidence number means

Visibility and confidence should not be conflated:

- **Visibility:** “Can this camera identify the requested opening?” There are three answers, one per camera.
- **Pose confidence:** “Given the images we have, how likely is this fused pose to satisfy the handoff accuracy limits?” This is one answer for the final estimate. For example, define the event as *correct selected-port identity and lateral error ≤0.5 mm, axial error ≤a chosen threshold, orientation error ≤a chosen threshold*. The thresholds must be fixed before calibration and correspond to the control gate.

We should **not** label every visible example “confidence 1” and every hidden example “confidence 0.” A visible but blurry/ambiguous port can have a poor pose; a partly hidden port can still have a good pose from two cameras. Instead, train a pose uncertainty or quality head, then calibrate its probability using **out-of-sample predictions** and their actual pose errors on held-out scene groups. Check whether predictions assigned, say, 80% success actually meet the chosen accuracy limits about 80% of the time. Evaluate false acceptance separately on wrong-port and occluded-target scenes. If calibration is weak, deploy a conservative accept/reject gate rather than claim a meaningful probability.

At deployment the model should return either **a selected-port pose with a validated quality score**, or **“do not hand off.”** The insertion actor should never receive a fabricated pose because another port happens to be visible.

These are **proposed labels and outputs**. The current checkpoints have projected-in-frame scores and temporal/geometric consistency checks, but do **not** have validated true visibility or calibrated pose confidence. See [the current perception experiment](../experiments/2026-10-02-two-family-perception-local-bc.md) and [the selected-port design](2026-10-02-one-selected-port-perception-explained.md).
