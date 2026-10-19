# How the evaluated rollout actually works

This describes the frozen 140-rollout evaluation completed on October 15. It distinguishes the executed code from the more general pipeline we discussed. No new rollouts, training or policy changes were made to answer these questions.

## 1. Is the runtime actor → handoff probability → coarse check → perception → insertion?

**That resembles our proposed learned-transport pipeline, but it is not what both branches of this evaluation executed.** We evaluated two different pipelines:

```text
SC-to-SC:
rule-based transport
      |
      v
camera check INCLUDING port-pose estimation
      |
      v
estimate where the held plug tip is relative to the TCP
      |
      v
explicit controller approaches to 30 mm before the opening
      |
      v
existing BC insertion actor

SFP-to-NIC:
learned diffusion transport with observation history
      |
      v
more transport chunks until the test budget or force stop

No connected SFP camera-check/insertion stage in these tests.
```

### A. How the learned SFP transport actor moves

1. Read the three robot-camera images and measured robot state. Measured state includes TCP position/orientation in `base_link`, velocities, joints and wrist force. Read the public task code: family, requested port and requested NIC card. Card count is not a separate input; the images show the cards.
2. Encode the current images and assemble stored, time-indexed observation features and robot-state context. This is the transport actor's history, not the coarse model's history.
3. The diffusion model predicts **12 future TCP-relative poses, 100 ms apart**, using 12 denoising steps. Execute the first six, nominally 600 ms of motion, then obtain a new chunk. Inference/simulation scheduling can add delay; this is not a guarantee of one inference every 600 ms of wall time.
4. Each future target is relative to the TCP at the chunk's observation. It is not an absolute world coordinate and is not an increment that must be repeatedly accumulated on the preceding prediction. For example, targets 2 mm and 4 mm ahead mean those two offsets from the same anchor, not a total of 6 mm.
5. Convert each anchored target to a fixed `base_link` TCP target. Using the latest measured TCP, calculate the current body-frame movement needed to reach it, apply movement limits, and submit the corresponding base-frame pose command to the controller. The controller then drives the joints.

In symbols, with `anchor` meaning the TCP when the chunk was predicted:

\[
T^{base}_{goal,j}=T^{base}_{anchor}\,\hat T^{anchor}_{goal,j},\qquad
\Delta T^{TCP_{now}}_j=(T^{base}_{TCP_{now}})^{-1}T^{base}_{goal,j}.
\]

**The actor's handoff signal did not trigger stopping here.** The selected checkpoint's trained handoff support is SC-only, and the SFP runtime's learned-stop threshold was disabled. The SFP arm consequently never reached a coarse inspection or insertion actor. This is an integration limitation of the evaluated runtime, not the intended final deployment design.

In the optional learned-stop implementation, the probability is checked at each new action-chunk decision, not after an arbitrary number of camera frames. A trained/supported family, configured threshold, rough TCP-region veto and a consecutive-positive streak are required. That optional mechanism must not be described as active in this matrix.

### B. How the SC candidate moves and hands over

1. The rule controller follows the bypass stages described below. No learned transport actor chooses these actions, and no actor handoff probability is consulted.
2. At its last acquisition stage, it invokes the camera/perception stack at each new chunk decision—nominally every four 50 ms commands, about 200 ms of simulated command time, with inference overhead.
3. The stack locates the requested port, checks its identity, refines its image coordinates, estimates its 3D opening position and orientation, and checks the robot's distance/posture/force. **Perception is therefore already used before transport is approved.**
4. Approval is a conjunction of tests, not one final learned “stop probability.” Among the tests: requested-port identity score at least 0.5 in each of three views; refined-view reprojection agreement within 2 pixels; acceptable board-orientation evidence; TCP within 15 mm of the estimated acquisition waypoint; posture error at most 3°; measured force norm at most 40 N.
5. Require at least three consecutive valid inspections, then sustained validity for two simulated seconds, with TCP speed at most 5 mm/s and angular speed at most 3°/s. If inspection fails, visual guidance can adjust the final waypoint within bounds; otherwise the controller continues or ends on its no-progress/budget guard.
6. After rule approval, collect three additional valid camera checks and three image-based estimates of the held plug-tip position relative to the TCP. Require those tip estimates to agree within 0.5 mm. Take their median.
7. **Lock** the approved port pose in base coordinates and the estimated gripper-to-tip position for this attempt. The next controller moves to the 30 mm stand-off. Only after its readiness test passes does the existing BC actor start.

The insertion actor receives fresh three-camera images, robot state and task code on each decision, plus the measured TCP expressed in the locked estimated port frame and the inferred plug-tip position. It predicts four complete plug-tip translation/TCP-orientation targets in that port frame, 50 ms apart, and executes all four. The runtime subtracts the rotated TCP-to-tip offset to obtain TCP targets, transforms them through the estimated port frame into base coordinates, then commands the robot.

**The selected runtime does not rerun the full port estimator continuously during insertion.** It assumes the task board and held-plug offset remain fixed, while measured TCP motion updates the estimated plug position. Fresh images still enter the actor. A slipping grasp or biased initial pose can therefore leave a persistent geometry error. The actor returns after its bounded attempt or force guard; the independent scorer determines physical insertion. An acknowledgment from the actor is not a success label.

No true simulator port/plug pose, guide, RL or backtracking controls these autonomous attempts. Diagnostic overview cameras and ground-truth transforms are for recording/scoring only.

## 2. What do “6/6 reused” and “6/6 fresh visual acquisitions” mean?

- **Reused:** rerun six development reset configurations already used in earlier debugging. All six reached and passed the acquisition test used in that screen.
- **Fresh:** run six newly generated development configurations within the supported normal-board envelope. All six passed that screen too.
- **Visual acquisition:** the robot has reached a viewpoint/standoff where the visual stack approves an opening pose for the requested port. It does not mean the plug is inserted, and insertion was disabled in those screens.

Those small screens were not a systematic five-per-card-count/port evaluation. The later matrix exposed failures and wrong-port approvals, so the old 6/6 figures cannot be treated as general reliability.

### The actual rule

It uses **nominal base-frame waypoints derived from the existing training trajectories**, with separate final waypoints for SC port 0 and port 1. It does not read the randomized board/port ground truth for the current episode.

```text
move away on the starting side
      -> move sideways and descend behind the row
      -> travel beside and beyond the row
      -> move across after clearing the row
      -> tilt while still beyond the cards
      -> return toward the acquisition point
      -> adjust the final point using camera observations
```

The first segment partly uses the measured starting position and orientation. Later corridor coordinates are nominal base-frame coordinates; the requested port selects the nominal endpoint. The route conservatively goes around the row rather than planning a new path from detected card positions. Card count does not invoke a different obstacle planner.

The controller smoothly advances its commanded reference at bounded rates (nominally up to 80 mm/s translation and 0.35 rad/s rotation), emits four targets per decision and converts them through the same movement interface. It checks measured waypoint completion—within 15 mm and 3° for three successive decisions—rather than assuming the robot arrived because enough time passed. The final camera guidance uses observed opening position and orientation to adjust acquisition; it can also use a consistent two-view estimate to reposition before requiring strict final three-view approval. Lack of measured progress terminates the attempt.

Thus it is **not just “move body-frame +X by a fixed amount and hope.”** It combines nominal fixed-frame route geometry, measured robot feedback and visual correction near the end. Its limitation is equally important: it assumes a supported board location/orientation and starting envelope. It does not infer an entirely new safe corridor for arbitrary board rotations. The broader alternate-board SC test failed 0/10 acquisition.

### Is rule transport better than diffusion?

For supported SC development scenes it was a more promising acquisition baseline than the earlier learned-only transport tests. But the latest matrix did not compare both transport methods on the same 50 SC scenes, and the hybrid still had 0/50 full insertions. We cannot claim a universal or cleanly matched superiority.

For SFP, the matrix evaluated learned diffusion transport only. It performed poorly: no sampled visits to the declared broad approach region in 60 runs. **We did not evaluate a matched SFP rule controller here**, so these results cannot decide which transport method is better for SFP.

## 3. How did the orientation repair work?

This repaired the **camera-derived orientation estimate**, not the diffusion model or insertion actor.

The estimator starts with a learned rough orientation prior. It then detects straight board/rail edges in the original images. Using known camera calibration and measured TCP pose, it asks which board rotation would produce those observed line directions. Two or more agreeing camera estimates supply a more precise board yaw; the calibrated board-to-port mounting rotation converts that into port orientation. This exploits a level, rigid board; it is not unrestricted six-degree-of-freedom object reconstruction.

The bug/limitation was a search window of only **±0.07 radians, approximately ±4°**, around the prior. In one difficult case, the best answer lay outside the window. All cameras hit the same boundary, which falsely looked like strong agreement.

The repair widened the search to **±0.21 radians, approximately ±12°**, widened the image-line prefilter, and rejected edge-saturated estimates. It also requires line consistency within a view and agreement across views. The matched case's measured error improved from 1.37° to 0.017°. No images were reconstructed or altered, no ground-truth angle was inserted into inference, and no model was retrained for that repair. Ground truth measured the error afterward.

If good board edges were seen earlier in the same attempt, a guarded orientation memory can retain them when current edges disappear. Current requested-port identity and position must still agree. That is **not** a full camera-history model for recognizing port identity. A corrected estimate can guide a robot posture adjustment; the approach controller below subsequently holds that acquired posture.

## 4. What is the “explicit visual/proprioceptive controller toward 30 mm stand-off”?

It is a small feedback controller, **not a newly trained actor**. “Proprioception” means measured robot position, orientation and motion.

After acquisition, the plug can still be about 150 mm before the opening. The existing insertion actor failed when asked to cover that whole gap. This controller takes ownership of that intermediate approach:

1. Use the locked visual port pose and image-estimated TCP-to-tip offset.
2. Combine them with the current measured TCP to estimate the plug tip relative to the opening.
3. Set a goal of **zero lateral offset and 30 mm before the opening**. Thirty millimeters describes the plug tip, not the TCP or camera.
4. Interpolate toward that goal at about 15 mm/s. Keep the acquired TCP orientation fixed and accumulate a bounded lateral correction for persistent tracking error.
5. Recompute from measured TCP roughly every 50 ms, without rerunning the full port estimator.
6. Hand over only after estimated axial error is within ±1.5 mm of that stand-off, lateral error is at most 0.5 mm, and measured speed is at most 5 mm/s continuously for one second. Otherwise stop on no-progress, time or force guards.

So “visual” means its target and plug offset came from cameras; the subsequent feedback is measured robot motion relative to those locked estimates. It is not continuous visual servoing with a fresh port pose every frame.

This explains the latest result: 17 additional SC runs reached roughly the correct neighborhood but failed the strict readiness condition. Most residual errors were lateral. It does not automatically mean they could not physically approach the port, or that the insertion actor failed in those runs—the actor never started.

## 5. Is refinement the perception model predicting 3D pose?

**Refinement is one part of the perception pipeline. The selected crop refiner itself predicts a more accurate 2D opening point in each camera, not a complete 3D pose directly.**

```text
three original images (1152 x 1024)
      |
      v
coarse locator on downsized images (288 x 256)
      |
      v
requested-port identity check and candidate selection
      |
      v
160 x 160 crops cut from ORIGINAL images
      |
      v
learned refiner: precise 2D opening point in each crop
      |
      v
calibrated camera geometry + measured TCP: triangulate 3D position
      |
      +---- board-edge orientation estimate + fixed port mounting
      |
      v
one requested opening's 3D position and orientation
      |
      v
combined identity / consistency / distance / posture / force approval
```

Cutting from original images preserves detail that was lost in the coarse downsize. Triangulation uses where the three calibrated camera rays agree, much like stereo depth. Board edges supply orientation separately. The coarse identity score can still be wrong: a highly accurate refiner can faithfully refine the wrong opening. That is exactly what happened in three candidate approvals.

This older implementation has 12 internal coarse 2D slots; the requested SC code selects one for fine opening refinement. The learned orientation-prior helper also consumes rough candidate geometry. This must not be confused with the proposed target-only temporal coarse architecture or with reliable prediction of every hidden port's full pose.

## 6. What did the table's test names mean?

| Previous name | Plain-language meaning | Why it was tested |
| --- | --- | --- |
| **SC candidate** | SC-to-SC, normal-board development layout: rule bypass, visual approval, explicit 30 mm approach, then the existing actor if ready. 50 runs. | Test the complete proposed hybrid used in the recent pilot. “Candidate” does not mean a newly trained model. |
| **SC far-start control** | Same rule/visual stack, scene settings and actor, but omit the explicit 30 mm approach. Start the actor at the acquisition point, roughly 150 mm before the opening. 20 runs. | Isolate whether adding the intermediate approach improves full insertion. “Far” refers to the insertion actor's takeover point, not a different initial episode start. |
| **SC alternate board pose** | Same candidate pipeline, but a different board location/orientation from a released public example; three cards, both ports, five repeats. 10 runs. | Check whether the nominal route and perception generalize beyond the usual board pose. It did not. These are development stress scenes, not reserved final scenes. |
| **SFP transport** | SFP-to-NIC using the learned diffusion transport actor only. 50 normal-board runs plus 10 alternate-board runs. | Measure whether this actor reaches a plausible approach region. Coarse approval and insertion are not connected, so full insertion is N/A. |

The “no insertion” outcomes therefore have different meanings. Some SC episodes never passed acquisition, some got near but never handed over, and three handed over but remained shallow. SFP episodes did not test the connected insertion pipeline at all. These must not be combined into one insertion-actor success rate.

## Code and evidence

- [Rule route and acquisition gate](../../aic_example_policies/aic_example_policies/ros/RunRuleTransport.py)
- [Visual port and held-tip lock](../../aic_example_policies/aic_example_policies/ros/RunGuardedRuleThenLockedTipLocal.py)
- [30 mm approach controller](../../aic_example_policies/aic_example_policies/ros/RunVisualApproachThenLockedTipLocal.py)
- [Existing actor runtime](../../aic_example_policies/aic_example_policies/ros/RunLockedPortLocal.py)
- [Combined perception/verifier](../../scripts/sc_shadow_handoff_verifier.py)
- [Full frozen evaluation](../experiments/2026-10-14-frozen-pipeline-coverage.md)
