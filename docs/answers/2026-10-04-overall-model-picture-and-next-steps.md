# Overall picture: what we have and what to do next

## The task in plain words

The robot starts away from a connector. It must carry an orange cable to the **requested** socket, line up the plug, and insert it fully. There are two task families: **SFP-to-NIC**, which can have several numbered NIC cards with two similar-looking ports on each card, and **SC-to-SC**, where the route may need to go around intervening cards. The official outcome is measured in **Gazebo**. We have also used **Isaac** for faster local experiments. Simulator truth is available to teachers and training labels, but an autonomous test must use only cameras, measured robot state and force, calibration, and the public task request.

The public request is a 10-number code identifying task family, selected port and, for SFP, selected card. It tells the robot *which* socket is wanted; it does not tell it where that socket is in the current image.

## The intended system

```text
Task request + 3 wrist cameras + measured robot state
                         |
                         v
  1. TRANSPORT actor: follow a safe route toward requested card/port
                         |
             proposes "pause near the destination"
                         |
                         v
  2. CAMERA CHECK: is this the requested opening, and where is it?
             |                         |
           unsure                 verified and safe
             |                         |
        keep looking             3. INSERTION actor
                                  align and insert using the estimated port frame
                                            |
                                            v
                            force/contact recovery, eventually learned with RL
```

This is the **design**, not a complete working deployment. The current runner does not reliably connect all stages from a normal start to a full autonomous insertion.

## The models and rules we actually have

| Part | What it does in easy words | Current evidence and limit |
| --- | --- | --- |
| **Transport actor** | A shared supervised policy for SFP and SC. A frozen ImageNet ResNet-18 reads each of three camera views; a temporal head combines image features with the task code, measured robot state, recent movement and commands. Every roughly 200 ms it proposes four complete target tool poses in the moving TCP/body frame. SC training uses successful *around-the-cards* routes; SFP uses verified teacher routes. | It can imitate teacher actions on recorded observations, and longer history helped one SFP development run. But closed-loop routes still diverge early. Recent balanced SFP actors reached **0/3** safe requested-port arrival buffers; a matched SC transport comparison reached **0/16** stable handoffs/insertions. This stage is not reliable enough to feed insertion. |
| **Transport stop proposal** | A small learned head or visual rule says when transport should pause for a camera check. It is only a proposal, not permission to insert. | SC visual stopping looked better than a transport-only head on saved paths but still made a wrong-route false stop. SFP stopping is not validated. The automatic switch remains disabled. |
| **Coarse perception** | An RGB locator finds likely socket pixels in each camera. The task code selects the requested card/port; measured TCP and fixed camera calibration turn matching image points into an estimated 3D opening. Variants predict all ports, only the requested port, or card slots and port pairs. | Some accepted predictions are accurate to a few millimeters, but middle-card coverage is poor and several heads confidently choose a neighboring card or the other port. On ten fresh SFP scenes the four-head union made **46 accepted wrong predictions**. A later slot/pair model accepted only **1/136** old development views. |
| **Original-image crop refiner** | First use an observation-derived coarse location, then crop the **original high-resolution** camera images around that location and look more precisely for the opening. It never chooses a crop from the simulator's true target pose. | SC clear-view local position can be quite accurate. For SFP, the true target was inside many proposed crops, yet the model could still confuse port identity. In our latest four-scene side-view test, the crop head accepted **3/4** at baseline and still **3/4** after the useful side move; the hard card-2/port-0 request was never accepted. |
| **Camera check / verifier** | Combines requested-port identity, three-camera geometric agreement, estimated opening orientation, tool standoff and orientation, and force/contact conditions. It should return a pose or **invalid**. | This is a pipeline of heads and fixed checks, not one magic model. Its current “in-frame” output says a port projects inside the picture; it is **not a proven visibility or calibrated confidence score**. Three cameras can agree on the same wrong socket. Hard occlusion, false handoffs and full live latency still need validation. |
| **Near-port insertion actor** | A separate PoseInsert-inspired transformer receives RGB, measured state, task code and the *estimated* port pose. It predicts four **complete tool targets in that estimated port frame**. It does not add a correction to a transport action or output absolute base-link coordinates. | A corrected deterministic version achieved **14/14 local 8 mm seating events** across two Isaac development sets, but those were near-opening starts and not official full-depth, normal-start Gazebo insertions. Later Gazebo near-port experiments advanced the plug substantially but did not reliably finish insertion. A retrained actor must be tested on real transport handoff states and full-depth scoring. |
| **Recovery / RL branch** | A rule can backtrack measured motion when blocked; an experimental mixture actor can choose different full port-relative action trajectories and was trained with SAC-style updates. This is where cable snag and contact recovery would eventually fit. | Local Isaac comparisons did **not** establish a reliable RL improvement over frozen BC. One video rerun favored RL plus backtracking, but matched metric repeats disagreed. Hierarchical backtracking is a promising design idea, not an integrated, proven full-episode solution. No new RL was run in the current SFP identity work. |
| **Older ACT and world-model branches** | Earlier baselines for behavior cloning and learned future dynamics. | Useful historical comparisons, but currently parked. The world-model predictions were not accurate enough to justify using them for precise insertion control. |

The strong CheatCode videos and scores are **teacher demonstrations**. CheatCode can read simulator target geometry. The latest side-view rollouts also used it to reach a diagnostic camera position, so their later full insertions do not count as autonomous learned-policy successes.

## Why the pieces have not added up yet

There are two upstream failures. First, the transport actor can make a slightly wrong early route decision and drift far away, even when its recorded-action prediction error looks small. Second, when the robot does get a view of several similar sockets, perception can pick the wrong numbered target or decline to decide. A precise estimate of the *wrong* socket is dangerous. The recent ±15 mm camera move was physically feasible but did not rescue the hard SFP target. If either upstream stage fails, improving the insertion actor or adding RL cannot fix the full episode reliably.

We also need to distinguish **“the projected opening lies inside the camera image”** from **“the opening can really be seen and identified.”** Much of the current training uses the first kind of label. Cable, gripper or cards can hide an in-frame opening. We have not yet certified true visibility for the difficult SFP examples.

## Suggested next steps, in order

1. **Find out what the cameras actually show in the hard cases.** Audit the existing original-resolution card-2/port-0 and card-1/port-0 images, plus wrong-port examples, camera by camera. Record whether the requested face is exposed, partly occluded, or hidden, and whether the card number and port number can be inferred from image context. Use simulator segmentation/depth only to *propose labels*, then check samples against RGB. Do not collect another routine batch or train a new head until this answer is clear.
2. **Choose the identity repair from that audit.** If the requested face and rail context are visible, test one observation-only method that explicitly detects card instances in row order and the two ports on each card, using original-resolution crops and real occlusion/rejection labels. Train on complete reset groups and compare to the frozen selected-only and crop heads. Require zero wrong accepted ports and useful coverage on newly frozen nonfinal scenes. If the face is genuinely hidden, have the robot seek a demonstrably clearer, force-safe view; repeating the failed ±15 mm translation is not enough. Do not use true geometry to choose an evaluation crop.
3. **Repair transport as a separate bottleneck.** Give the shared actor a clearer observation-derived route stage or next safe subgoal and check that it follows the SC around-card route and reaches the requested SFP card from normal starts. Score measured closed-loop arrival, route contact and task identity, not just teacher-action error. The 104 verified SC route episodes and balanced SFP teacher collection are existing starting data. Avoid another undirected BC sweep.
4. **Test the full pause-and-check handoff.** Transport proposes a pause; the robot holds; perception either verifies the requested opening and safe posture or rejects and resumes/looks again. Challenge known wrong-card, wrong-port, hidden-port and wrong-route cases. Measure complete camera-to-command p95 below **300 ms**. Leave reserved final scenes sealed.
5. **Then attach and retrain the insertion actor** on the actual distribution of actor-generated handoff poses, including lateral and orientation errors, both task families, and full-depth teacher targets in the estimated port frame. Evaluate complete autonomous Gazebo episodes from their normal starts, reporting full insertion, wrong-port events, force and latency.
6. **Only after supervised full insertion becomes repeatable, revisit RL and hierarchical backtracking** for blocked insertion and cable-route failures. Compare BC, RL, local backtracking and hierarchical backtracking on the same episode-grouped starts. Keep the world-model branch parked unless a specific dynamics prediction test shows it adds value.

My immediate recommendation is **Step 1**. It is a small, read-only diagnosis of existing images and determines whether the next fix should be a better card/port identity model or a genuinely better viewpoint. In parallel planning, transport remains an equally serious blocker; neither should be hidden by a successful CheatCode score.

Sources within this repo: [current status](../STATUS.md), [handoff design](../SELECTED_PORT_HANDOFF_PLAN.md), [two-family perception and local BC](../experiments/2026-10-02-two-family-perception-local-bc.md), [SFP fresh identity failure](../experiments/2026-10-04-sfp-fresh-anchor-shadow.md), [side-view result](../experiments/2026-10-04-sfp-side-view-diagnostic.md), and [local RPDP/RL history](../experiments/2026-09-22-serl-mixture-recovery.md).
