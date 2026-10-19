# Visibility versus pose confidence

**Having both should not confuse the model if they are defined and trained as different outputs.** Visibility is about what each camera can identify in its *current image*. Confidence is about whether the **combined final pose** is likely accurate enough for insertion. The three cameras, viewing angle, image blur, similar neighboring ports, and temporal history can make these answers differ.

| Situation | Selected-port visibility | Final pose confidence | Why they differ |
| --- | --- | --- | --- |
| Opening is clear in all three views but port 0 and port 1 look very similar, and the model may choose the wrong one | High in each camera | **Low** | Seeing an opening is different from knowing it is the *requested* opening. |
| Opening is clear but only a few pixels wide because the robot is still relatively far away | High or identifiable | **Low** | The target can be recognized, yet a one-pixel error becomes too many millimeters for insertion. |
| Opening is visible, but two cameras have nearly the same viewing direction or their predicted rays disagree | High in those views | **Low** | The 3D estimate is poorly constrained or inconsistent even though the opening appears in the images. |
| Cable blocks the selected opening in the right camera; center and left show crisp, different-angle views | Right: **no**; center/left: **yes** | **Potentially high** | Two independent good views can still locate the selected port accurately. The confidence must be checked against actual held-out errors. |
| Only a thin edge of the opening is visible in one camera, but two other cameras show it clearly | One camera: low/uncertain; two: high | **Potentially high** | The weak view can be ignored; the other views carry the pose. |
| All three current images are occluded, but a port pose was previously verified and locked while the robot moved briefly | Current visibility: **no** in all views | **Possibly temporarily high** for a *tracked* pose | A recent, well-verified pose of a stationary port can be carried forward. Confidence must decay with time and motion uncertainty. This is a separate tracking mode, not a fresh visual detection. |
| Requested opening is hidden; only the neighboring port is visible | Selected-port visibility: **no** | **Low / invalid** | The neighboring opening must not be substituted for the target. |

The easiest definition for training is:

- **Per-camera visibility head:** “Can this image identify the **requested** opening well enough to locate it?” Three labels, one per camera. A visible fraction can be recorded separately, but it is not the confidence score.
- **Pose and quality head:** Estimate one selected-port pose from the usable cameras. Confidence means “probability that the **selected identity is correct** and this pose falls within fixed lateral, axial, and orientation error limits.” It is calibrated using the pose model's errors on held-out scene groups.

I would avoid training confidence as a second copy of visibility. A simple implementation is to train the visibility and pose heads with separate losses, mask pose loss when the selected opening cannot be identified, then **calibrate the pose-quality score after training** using out-of-sample errors. This lets a view be visible but inaccurate, or one view be hidden while the fused estimate remains accurate. If all views are hidden and there is no previously verified tracked pose, the output is **no valid pose**, regardless of what the network's raw confidence says.

The current model does not yet have these true labels. Its “in-frame” output is only a projection-related signal, and its agreement/lock checks are not calibrated pose confidence. The table describes the proposed design, not observed reliability of a trained model.
