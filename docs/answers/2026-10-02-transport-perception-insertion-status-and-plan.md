# Transport, perception, and insertion: status and plan (October 2, 2026)

## The idea in plain words

We want the robot to use two different movement policies. **Transport** gets the plug from its initial position to a safe place near the chosen port. **Perception** looks for that particular port and estimates its opening relative to the robot's measured tool pose. **Insertion** uses that estimate to align and push the plug in. The intended handoff is:

```text
task request + three camera views + measured robot state
                         |
                         v
     TRANSPORT: navigate toward the selected port
                         |
                         v
  PERCEPTION: is the selected opening reliably located?
                         |
                 safe handoff near port
                         |
                         v
     INSERTION: align, approach, and insert
```

This is the **design**, not an already reliable end-to-end result. Teacher-assisted near-port tests have succeeded, but the current learned actors have not demonstrated reliable autonomous full insertion from a normal start. We have therefore kept the final evaluation scenes sealed and have not started this branch's RL training.

## What each part sees and does

| Part | Main inputs | Job and output | Current limitation |
| --- | --- | --- | --- |
| Transport actor | Three wrist RGB views, measured robot state and motion history, prior commands, and task code | Predicts a short chunk of motions in the moving tool/TCP frame. It can navigate while the target is out of view. For SC-to-SC it learns the route **around** the cards from the new Low Lateral Bypass demonstrations. | Some starts reach the vicinity; other starts take a poor route or end at the wrong orientation. A trustworthy automatic stopping decision is still missing. |
| Perception pipeline | Three RGB views, measured TCP pose, camera calibration, and selected target identity from the task code | Finds candidate port openings, selects the requested one, refines its image position using original-resolution crops, and estimates the opening's 3D pose relative to the TCP. | A geometrically plausible estimate can still be for the wrong opening. Its current consistency score is not a certified visibility or calibrated confidence score. |
| Insertion actor | Three RGB views, measured robot state, task code, and the **predicted** TCP pose relative to the selected port | Predicts four complete target TCP poses in the predicted port frame. Runtime transforms these targets into commands for the robot. | On a frozen set of actor-generated near-port states, recent BC candidates tended to retreat rather than approach. More teacher demonstrations alone have not resolved that gap. |

The perception pipeline is a sequence of models and geometric calculations, rather than one network doing everything. In the SFP case it locates the likely opening in the full images, refines it in a 480-pixel region and then a 160-pixel crop cut from the **original 1152 × 1024 image**, and combines the camera views. SC has its own native crop and two-port locator. During autonomous use, the images and public task code choose the crop; Gazebo's true port coordinates are used for training labels and later scoring, not as an actor input.

The estimate is primarily the **port opening relative to the measured TCP**. TCP is the robot's tool reference point; it is not exactly the plug tip. That distinction matters when judging whether the plug is physically lined up.

## What the task code means

The shared code has ten entries:

```text
[is_SFP, is_SC, port_0, port_1,
 card_0, card_1, card_2, card_3, card_4, card_valid]
```

For an SFP-to-NIC request, it identifies the chosen NIC card and port 0 or 1. For an SC-to-SC request, it identifies the chosen SC port 0 or 1; the NIC-card entries are empty. Here **port 0/1 means which port to target**. The first two entries tell the model whether the task is SFP or SC. The code does not describe all obstacle cards, their positions, the cable shape, or whether the target is visible. Those must be inferred from the observations.

The transport actor uses this code to condition its route and motions. The perception pipeline uses it to pick the requested opening among candidates; individual vision submodels may detect several candidates before that selection. The insertion actor uses the same code along with the **predicted** port-relative pose. Thus, when the target is out of view, transport can keep moving from images and history; the insertion actor should only take over once the selected opening has been located reliably.

## How the training data and coordinate frames fit

For SFP transport, we have verified successful expert trajectories, including additional stock CheatCode successes spanning the ten card/port identities. For SC, the revised Low Lateral Bypass route takes the cable **around the card row** rather than over it. Collection attempted 200 varied SC settings and yielded 104 officially scored full insertions suitable for the successful set. Some unsuccessful attempts remain useful for diagnosis, but are not successful BC demonstrations.

The transport actor learns motions in the robot's moving TCP/body frame. The insertion actor learns complete TCP targets in the **selected port frame**. To make those insertion labels, preprocessing composes the recorded expert TCP-frame movement with the causally measured TCP pose, then expresses the resulting expert target relative to the port estimate. This is not an absolute base-frame coordinate predictor or a small correction added to a second actor's answer. At deployment, the insertion actor gets a vision-derived port estimate, never the true simulator port pose.

The SC route demonstrations and the near-port insertion demonstrations solve different pieces of the episode. A teacher bringing the robot to a near-port start does **not** count as autonomous success from the initial state.

## What has actually worked, and what has not

- The SFP visual cascade is often very accurate on accepted development frames, but a wrong-port visual agreement still occurred. In one 100-frame wrong-target check, two individual pose estimates were wrong by about 48 mm; the five-estimate lock did not fire. We therefore have **not** enabled SFP visual stopping as a reliable handoff.
- On seven frozen SC transport traces, visual stopping found five valid handoffs, made one false handoff, and missed none of the valid arrivals. The transport-only stop made two valid handoffs, two false ones, and missed three. The improvement is useful, but the false stop and live route variability keep this a development result.
- A recent SFP BC correction improved a held-out near-port lateral target error from 1.82 to 1.72 mm at the 95th percentile, below its preset improvement gate. On nine frozen starts generated by the actor itself, both compared BC models still retreated. Teacher-state predictions looked much better, showing that the model behaves differently after its own earlier actions.
- A previous near-port SFP trial had a good port estimate but the actor moved the plug far sideways and did not insert. A SC near-port actor reached the face of the opening but remained roughly 1.44 mm laterally off and did not insert. Precise perception by itself does not guarantee good control.

These results are development measurements on particular splits and starts. They do not establish a full-episode success rate for the intended combined system. The 300 ms complete inference limit also still needs to be checked for the integrated pipeline; some older SC local-actor timings exceeded it.

## Next steps

1. **Make the handoff trustworthy.** For SC, reject the known wrong-route visual stop using observation-only route and orientation checks, while retaining the valid arrivals on the frozen comparison. For SFP, improve selected-card identity and determine whether the requested opening is actually visible; do not treat crop projection or repeated agreement alone as proof. Require a safe pre-insertion position and orientation, acceptable contact/force, and complete inference p95 below 300 ms.
2. **Fix the insertion actor's first moves from real transport-like starts.** Diagnose why it retreats on its own near-port states. Test an explicit pose-conditioned approach phase or a bounded geometric approach control against matched BC controls. Evaluate forward progress, lateral and orientation alignment, and force on frozen development starts. All runtime pose information must come from vision.
3. **Join transport and insertion.** Start at ordinary SFP and SC episode starts, including cases where the target is initially offscreen. At the handoff, hold the measured pose while preparing the insertion actor, then run the entire episode without a teacher prefix or true port coordinates. Measure actual full insertions, route and collision failures, latency, and errors by task and target. Use new episode-grouped development starts; leave reserved final scenes sealed.
4. **Only after reliable supervised insertion, add recovery and RL.** The planned single actor should handle ordinary motion and learned recovery, with hierarchical backtracking available as a rule for detected blockage. RL would then address misalignment, force contact, and cable snags. World-model and Seer directions remain parked until this simpler perception-plus-control path is understood.

For underlying evidence and exact artifacts, see [current status](../STATUS.md), [perception and local BC record](../experiments/2026-10-02-two-family-perception-local-bc.md), and [transport stopping comparison](../experiments/2026-10-02-transport-stop-comparison.md).
