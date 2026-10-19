# Component summary, handoff timing and camera geometry

Saved from the side-conversation explanation on October 15, 2026, with answers to the two follow-up questions. This is a snapshot of the inspected main-thread state; it does not claim that pending main-thread experiments have since completed.

## Component-by-component summary

Your description captures the **intended direction**, but some parts are still being prepared, and a few work differently in the code. In particular:

- Rule transport is **not yet proven reliable across both families and all settings**.
- The new coarse model with history and two heads is the intended design, not the older model used in the 140-rollout test.
- The crop refiner predicts **2D opening locations**; calibrated geometry and a separate orientation estimator produce the 3D pose.
- Our latest tested insertion actor is **PoseInsert-inspired**, not the paper's complete diffusion architecture.

### Overall pipeline

```text
Task request: family, target card and target port
                         |
                         v
RULE TRANSPORT — move toward the target area
                         |
                         v
COARSE PERCEPTION — identify requested port and propose handoff
                         |
                         v
ORIGINAL-IMAGE REFINEMENT + CAMERA GEOMETRY + ORIENTATION
                         |
                         v
INSERTION ACTOR — approach, align and insert
                         |
                  blockage detected?
                         |
                         v
HIERARCHICAL BACKTRACKING — retreat, retry, escalate
                         |
             large retreat / unreliable view?
                         |
                         +----> transport reacquisition + fresh perception
```

The latest approved plan makes the insertion actor responsible for **the whole final 150 mm approach**.

### 1. Transport: following a route instead of learning every movement

#### What we tried

We trained transport actors with BC from expert trajectories, including transformer and diffusion models. Several repairs improved offline prediction, but autonomous transport/handoff remained unreliable.

We then tested a rule-based SC bypass. It showed promising acquisition on small supported-layout tests, but the broader evaluation exposed substantial failures. Therefore, we have **chosen to develop rule transport further**, rather than established that it already works reliably everywhere.

#### How it works

For SC, the route is approximately:

```text
move away
  → move sideways and descend behind the cards
  → travel beside and beyond them
  → move across
  → tilt
  → return toward the requested port
```

It uses:

- Nominal waypoints in the robot's fixed base frame.
- Measured TCP position/orientation to check progress.
- A different final waypoint for each requested port.
- Visual correction near acquisition.

It does **not** simply issue a fixed sequence of body-frame movements without checking whether they happened.

For SFP, the approved starting point is a **CheatCode-style route**, adapted into an autonomous rule controller. CheatCode itself can use privileged geometry; that geometry must not enter the deployed controller.

**Current status:** validating and adapting these routes for both families remains pending. Small waypoint adjustments alone will not handle every substantially different board pose.

### 2. Coarse perception: “Is this the requested port, and is this a useful handoff view?”

#### Intended architecture

Yes: the intended model shares its visual/history backbone between two outputs:

| Output | Meaning |
|---|---|
| Handoff eligibility | “The requested target is sufficiently observable and suitably placed for the next stage.” |
| Crop location for each camera | “Look around this image location for that requested target.” |

Inputs should include:

- Current images from all three cameras.
- Earlier images sampled by **distance traveled**, rather than elapsed time alone.
- Robot state.
- Requested task family, card and port.

The current images determine what is visible **now**; history helps distinguish repeated-looking ports.

For example:

> The images contain SC ports 0 and 1. With `requested_port=1`, the model must locate port 1. Changing the request to port 0 should change its answer. If only port 0 is visible while port 1 is requested, it should reject handoff.

#### Does it predict four rectangle corners?

**The existing implementation predicts image points, not four rectangle corners.** A fixed-size crop is constructed around each predicted center.

Also, the two heads can be computed together whenever the coarse model runs. There is no need to wait for handoff approval before computing the crop coordinates. The expensive refiner can run when there is a plausible handoff proposal.

The precise polling cadence, probability threshold and history spacing are **not yet finalized for the new design**. We should not present the earlier examples—such as 40 mm spacing and ten history points—as settled constants.

#### Important current limitation

The older evaluated stack used internal candidate slots and a separate current-image identity check. It had **no full camera-image history** and made three wrong-port approvals in the primary SC candidate tests.

The new execution is auditing its data and labels before deciding the repair. Projected “inside the image” labels are also being distinguished from genuine visibility.

### 3. Refiner: finding the opening precisely

The refiner's job is narrower than “predict the complete 3D pose from a crop.”

```text
Coarse center in each camera
          |
          v
Cut a crop from the ORIGINAL high-resolution image
          |
          v
Neural refiner predicts the precise 2D opening point
          |
          v
Combine calibrated camera rays to obtain 3D opening position
```

#### How the model predicts a point

The prepared shared refiner uses a ResNet18 image encoder and a small spatial decoder. It produces a heatmap, then calculates a weighted average of its pixel positions. This allows a continuous coordinate rather than restricting the answer to one integer pixel.

The new code conditions it on **13 numbers**:

- Task family: 2.
- Requested port: 2.
- Requested NIC card: 5.
- NIC-card-valid flag: 1.
- Camera identity: 3.

These are supplied as additional feature channels. For SC, the NIC-card portion is inactive.

#### What has actually been trained?

The previously selected high-resolution refiner was trained on **83 SC training episodes**, not a broad shared SC/SFP corpus.

The main thread has prepared the shared trainer and is auditing an expanded native-image corpus. **The broad retraining has not yet been completed according to the saved execution state.**

A useful repair underway is matching each image to the TCP pose at **that image's timestamp**, rather than using a slightly newer robot state to generate its projection label.

### 4. Orientation: how OpenCV contributes

**The OpenCV edges are not currently encoded as extra neural-network input channels.** They are used by a separate geometric calculation.

The selected SC estimator does this:

1. A learned model provides a rough orientation.
2. OpenCV **Canny** detects edges.
3. **Hough line detection** converts long edges into line segments:

   \[
   (x_1,y_1,x_2,y_2).
   \]

4. Each segment gives an observed image angle.
5. Using camera calibration and measured TCP pose, the algorithm tests which board angle would produce that line direction.
6. It combines consistent estimates across cameras and rejects unreliable ones.

In easy words:

> “If the board were rotated by this amount, its rails should look like these lines. Which rotation best matches the actual image?”

It then applies the calibrated board-to-port mounting rotation to obtain the port orientation:

\[
R_{\text{port}}=R_{\text{board}}R_{\text{board-to-port}}.
\]

The result becomes an ordinary numerical rotation—stored as a quaternion and converted into the actor's rotation representation where needed.

The previous repair widened an overly narrow angle-search range and rejected estimates stuck at its boundary. That fixed one matched case from **1.37° to 0.017°**.

**Limit:** this exploits a level board and known mounting geometry. It is not a general estimator for arbitrary board tilt, and equivalent SFP orientation handling still needs validation.

### 5. PoseInsert: what we borrow, and what our actor actually does

#### The paper's idea

PoseInsert expresses the held object relative to the target, then predicts its future trajectory in that target frame. Demonstrations supply relative-pose sequences. Its pose encoder treats translation and rotation separately; its RPDP variant also processes current/goal RGBD images and uses a learned gate to add visual features to pose features. A diffusion decoder generates future pose trajectories. At deployment, those trajectories are transformed into gripper commands using camera calibration and the grasp relationship. [Paper](https://arxiv.org/html/2505.09424v2)

The released fusion code implements:

\[
f_{\text{combined}}
=
f_{\text{pose}}
+
g(f_{\text{pose}})\odot f_{\text{image}}.
\]

That is **feature fusion**, not an action correction added to another policy. [Implementation](https://github.com/sunhan1997/PoseInsert/blob/main/policy/cnn.py)

#### Our latest tested Gazebo actor

It adopts the **port-relative action representation**, but uses a deterministic ResNet18/transformer BC actor. It does **not** reproduce the paper's complete gated RGBD/diffusion architecture.

**Training:**

1. Take recorded expert target commands.
2. Express their future targets in the estimated port frame.
3. Train the actor to predict those complete targets from images, robot state, task code and estimated geometry.

**Runtime:**

1. Obtain the camera-estimated port pose.
2. Express the current TCP/estimated plug position relative to it.
3. Predict future port-relative targets.
4. Account for the plug-to-TCP offset.
5. Convert targets into base-frame robot commands.
6. Execute a chunk and repeat.

For example, the policy reasons in terms of:

> “The plug is 100 mm before the opening and 5 mm sideways,”

rather than memorizing one particular world-coordinate destination.

The **new 150 mm insertion model is approved work, not an already validated checkpoint**.

### 6. Hierarchical backtracking

This is a separate deterministic supervisor around the learned insertion actor:

```text
blockage
  → retrace measured position/orientation
  → verify clearance
  → actor retries
  → repeated blockage: retreat to an earlier checkpoint
  → large retreat: reacquire target and refresh perception
```

The actor learns alignment and retry behavior; the supervisor owns forced retreat. Hierarchical deployment and its controlled comparison remain upcoming work.

**In short:** the intended system is rule transport → requested-target coarse perception → precise geometric perception → a port-relative insertion/recovery actor. The current main thread is still preparing and auditing that upgraded system; the fully integrated, broadly trained version is not finished yet.

---

## Follow-up 1: Does coarse run every 200 ms and hand over above a probability threshold?

**That is a reasonable description of the intended scheduling, with two important distinctions.**

### Both heads run together

At a coarse-model decision, provide the current three images, selected task code, robot state and retained history. One shared model computes:

- A proposed opening/crop center `(x, y)` for each camera.
- A handoff-eligibility probability.

It outputs coordinates, not an actual cropped image. Ordinary image-processing code cuts the requested regions from the original images afterward. A fixed-size crop needs only its center, so four predicted corners are unnecessary in the current design.

### A high probability proposes a check; it does not bypass validation

The intended sequence is:

```text
While rule transport moves:
    periodically run coarse perception
           |
    probability low -> continue transport
           |
    probability high for a sufficient streak
           |
           v
    hold briefly and run precise refinement/geometry checks
           |
    requested identity, pose consistency and starting envelope valid?
           |
       no -> reject proposal and continue/reacquire
      yes -> hand over to insertion
```

Thus, “handoff probability” means readiness to propose the next stage; it is not a predicted probability that insertion will succeed. One high score must not automatically accept the wrong port or an inconsistent pose.

**Checking every 200 ms would mean approximately five checks per second.** That is an example cadence, not a finalized or validated setting for the new model. Actual wall-clock frequency also depends on inference time and simulator scheduling. The older evaluated SC rule stack checked near its final route stage at approximately this command cadence, but it did not use the new shared two-head history model.

Two clocks must remain separate:

- **Inference cadence:** how often to inspect the current view, perhaps every 200 ms.
- **History-keyframe spacing:** when to add a distinct historical observation, based on distance traveled.

The robot can be inspected repeatedly while stationary without filling its history with many identical frames. Values such as 40 mm keyframe spacing and ten history entries still require validation.

## Follow-up 2: Original-image refinement and camera geometry, explained from zero

### A. What “original-image refinement” means

A camera image is a grid of colored pixels. Our original images are **1152 × 1024 pixels**. Running coarse perception on a smaller **288 × 256** version is cheaper, but throws away fine detail.

Suppose coarse perception finds the opening around `(150, 100)` in the small image. The corresponding original-image neighborhood is approximately `(600, 400)` because both dimensions were reduced by about four. In practice, the implementation uses its normalized-coordinate mapping consistently rather than relying on this approximate example.

We return to the stored original image and cut out, for example, a **160 × 160** patch around that neighborhood. The refiner looks at those original pixels and estimates the opening's location more precisely.

```text
Full original image -> smaller image -> rough location
         |
         +--> cut original pixels around that location -> fine location
```

It does not enlarge the small image and recover missing detail. It does not invent or reconstruct an image. It reuses pixels that the camera captured originally.

The refiner initially returns a location *within the crop*. Adding the crop's top-left offset converts that result into a location in the full original image. This is done independently for the cameras.

### B. Why a pixel location is not yet a 3D position

One camera can tell us the **direction** toward the opening, but not its distance from that camera. A nearby small object and a farther large object can occupy similar image locations.

Think of drawing a straight line from the camera lens through the detected pixel and extending it into the scene. The opening lies somewhere along that line. This line is called a **ray**.

### C. What calibration tells us

We need two kinds of known camera information:

1. **How pixels correspond to viewing directions.** This includes focal length/field of view, image center and image size. These are called camera *intrinsics*.
2. **Where each camera is and which way it points.** These are camera *extrinsics*.

The cameras are mounted on the robot. Their positions relative to the TCP are fixed and calibrated. Combining that fixed mounting relationship with the measured TCP position/orientation tells us where each camera was when the image was captured.

This uses known camera mounting and measured robot state—not the simulator's hidden target-port position. Camera geometry is therefore mostly a calibrated mathematical relationship, not geometry that a neural network must extract from RGB by itself. The neural network supplies the observed opening pixels.

### D. How multiple cameras recover distance: triangulation

Each camera predicts the location of the **same requested opening**, giving a ray from each viewpoint:

```text
left camera    --------\
                       \
                        *  opening's estimated 3D position
                       /
right camera   --------/

The center camera supplies another ray/check.
```

Where those rays meet tells us the opening's location in 3D. This is **triangulation**, similar to how two eyes provide depth information.

Real predictions are imperfect, so the rays usually do not intersect exactly. The implementation solves for a point consistent with their image measurements, then checks the disagreement. We can also project that estimated 3D point back into each camera and ask: “Does it land near the pixel the refiner predicted?” This is a **reprojection check**. The older stack rejected results with more than two pixels of disagreement.

Agreement does not prove correct identity: if all cameras consistently find the other port, geometry can reconstruct that other port accurately. That is why requested-port identification remains essential.

### E. Where orientation comes from

The triangulated opening point gives **position only**. A point cannot tell us which way the port faces.

Orientation comes separately from the learned rough estimate, visible board/rail line directions, camera calibration and the known board-to-port mounting relationship described above. The selected SC method assumes a level rigid board. Its result must not be described as general unrestricted 3D orientation recovery.

Together, position and orientation define a **port coordinate frame**:

- Its origin is the opening.
- Its axes specify sideways, vertical/in-plane and insertion directions according to our convention.

### F. How the insertion actor uses the result

The actor can now express the plug position as “150 mm before the opening, 5 mm sideways,” even when the entire task board has moved in the robot's base frame.

Its predicted future targets are expressed in that estimated port frame. The runtime transforms them back into robot-base coordinates, accounts for the offset between TCP and plug tip, and commands the robot.

So the responsibilities are:

| Part | Responsibility |
|---|---|
| Coarse model | Identify the requested target, suggest crop centers and propose readiness. |
| Native crop refiner | Precisely locate the opening in each image. |
| Calibrated camera geometry | Turn corresponding image points into a 3D position. |
| Orientation estimator | Determine the port axes within its validated assumptions. |
| Insertion actor | Choose the approach/alignment/insertion trajectory relative to those axes. |

## Relevant implementation references

- [Main-thread execution plan](../experiments/2026-10-15-rule-perception-hierarchical-execution.md)
- [Prepared shared native refiner](../../scripts/train_joint_native_opening_refiner.py)
- [Older combined visual verifier](../../scripts/sc_shadow_handoff_verifier.py)
- [Camera triangulation](../../scripts/evaluate_sc_native_triangulation.py)
- [Board-line orientation estimation](../../scripts/evaluate_sc_board_lines_yaw.py)
- [Existing port-relative insertion runtime](../../aic_example_policies/aic_example_policies/ros/RunLockedPortLocal.py)
