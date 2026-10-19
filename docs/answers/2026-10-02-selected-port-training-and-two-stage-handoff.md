# Selected-port training and the two-stage handoff (October 2, 2026)

## 1. How I would train perception to find the requested nearby port

The model should answer **“Where is the port named by this task?”**, not **“Where is the nearest port?”** The difference matters because two similar openings can appear together. The task code names one of them.

### A concrete SC example

Imagine a camera image in which SC port 0 is on the left and SC port 1 is on the right. Both are visible. Use the **same three camera images and the same measured robot pose** to make two training examples:

| Images and robot pose | Task input | Correct output |
| --- | --- | --- |
| Same scene, both ports visible | `SC, port 0` | Identify the **left** opening; report its image points and pose relative to the TCP. |
| Exactly the same scene | `SC, port 1` | Identify the **right** opening; report its image points and pose relative to the TCP. |

This is *counterfactual target training*: changing only the requested port must change which opening is returned. For SFP, the same method pairs a scene with requests such as `card 2, port 0` and `card 2, port 1`, or different visible cards. We should include neighboring, similar-looking openings as deliberate hard negatives.

Now consider an image where **port 0 is visible but requested port 1 is hidden by the cable**. The correct output is **“requested port not currently identifiable”**, even though port 0 is clear. It must not substitute port 0 or invent port 1's precise pose. The pose loss is masked for that training example; the visibility/rejection loss remains active. A separate example can show the requested port outside every camera view. This teaches abstention.

### Proposed inputs and outputs

At the near-target verification stage, one prediction would use:

```text
Inputs
  3 synchronized RGB camera images
  public task code: SFP/SC, selected card if SFP, selected port 0/1
  measured TCP pose and possibly a short recent observation window
  fixed per-camera calibration

Outputs for the SELECTED port only
  per-camera: identifiable? yes/no, and opening pixel coordinate if yes
  fused: selected-port 3D position + orientation relative to measured TCP
  quality: calibrated chance that position/orientation error is within
           chosen safe-handoff tolerances, or "no reliable estimate"
```

The **image model** can first inspect a low-resolution full view to nominate a region for the requested port, then inspect an original-resolution crop of that region to place the opening precisely. The crop center must come from the images and task code, never the simulator's target coordinates. At least two usable camera views permit 3D triangulation; the measured TCP pose and camera calibration locate those camera rays in 3D. If the requested port cannot be identified, the system returns **invalid**, not a confident pose. We may keep a short-lived locked pose after a *verified* observation through a brief occlusion, but that is a separate, time-limited tracking rule.

Training supervision can use Gazebo's known selected-port pose to project target keypoints into each camera and assess 3D error. **Actual identifiable visibility** needs its own labels: projection inside the image is insufficient when a cable or card occludes the opening. We should derive these from a checked render/segmentation or depth test and audit examples by eye, since prior auxiliary-depth recordings were unreliable. Accuracy confidence would be calibrated on held-out complete scene/reset groups: for example, among predictions reported as “90% likely to be within 0.5 mm and 3°,” about 90% should truly satisfy that event. These are **proposed outputs and tests**, not capabilities already proven by the current checkpoint.

An implementation may internally detect both nearby SC openings before selecting one. That is fine: the important point is that it only **returns and validates the requested target pose**, and it does not waste effort claiming precise 3D poses for every port in the whole scene. The current coarse locator has 12 fixed output slots and selects one afterward; the proposed near-target model can be explicitly task-conditioned and trained on same-image, different-target examples.

## 2. The two decisions are sequential, not alternatives

I was **not** asking you to choose between decision 1 and decision 2. Both are needed:

```text
Transport actor moves
    |
    v
Decision 1: "This looks like the right neighborhood; pause to inspect."
    |
    v
Hold current measured TCP pose and run precise selected-port perception
    |
    v
Decision 2: "The requested opening is verified and the waiting pose is safe."
    |
    +-- no --> resume transport / seek view / safe hold
    |
    +-- yes --> switch to insertion actor
```

**Decision 1 resembles an output we already have.** The temporal transport model has an auxiliary handoff-probability head based on images, task code, measured state, and motion history. In one diagnostic, probability ≥0.95 for two decisions after 15 seconds proposed a stop. It produced **2 valid, 2 false, and 3 missed arrivals** on seven frozen SC paths. Today this stop can end a transport runner, but it does not launch the insertion actor.

The proposed change is how we **interpret and train** that head. Its output would mean **“pause for precise inspection,” not “insertion is safe.”** We would train/check it specifically against safe pre-insertion-neighborhood examples and hard negatives: wrong port, close position but wrong orientation, unsafe route, and starts where the destination is still offscreen. It need not be accurate to 0.5 mm; it must have enough recall to invite inspection without firing all the time. The existing head and threshold have **not** passed that gate, so it may need a targeted repair.

**Decision 2 is the new strict verifier at the handoff.** It checks the requested port's identity and local pose from fresh close-up images, the measured TCP's relation to a safe waiting pose, visibility/quality, and force/contact state. It must reject a false proposal from decision 1. This is why precise perception can run **only after** transport proposes arrival, yet we still have a way to decide *when* to run it.

The existing SC visual-standoff experiment already mixes some local perception into late transport and got **5 valid, 1 false, 0 missed** stops on those seven paths. That is evidence for visual verification, but its false stop shows that current position-based checks do not yet establish safe orientation and route. The proposed two-stage controller should be evaluated on those same failures before allowing insertion. SFP still lacks a validated first-stage stop.

See [current stop evidence](../experiments/2026-10-02-transport-stop-comparison.md), [perception evidence](../experiments/2026-10-02-two-family-perception-local-bc.md), and [the earlier deployment explanation](2026-10-02-near-target-perception-and-explicit-handoff.md).
