# Proposed direction: rule transport, requested-port perception and hierarchical recovery

## Overall recommendation

I agree with this direction, with three qualifications:

1. Treat rule-based transport for both families as the next candidate to validate, not as an already proven replacement for both. The SC bypass has useful restricted-layout evidence; an equivalent SFP rule controller has not passed a comparable broad test.
2. Audit coverage and labels before deciding to retrain every perception component. The selected coarse locator was not trained on only a handful of episodes, and the refiner is conditionally accurate. Requested-port identity, deployment integration and scene coverage are concrete gaps.
3. First establish a broader-start BC insertion policy, then add and compare local versus hierarchical recovery, and then train recovery with RL. A hardcoded retreat controller and an RL-trained insertion actor have different responsibilities.

No training, new rollout or policy modification was started for this discussion. The two questions below concern the insertion/transport responsibility boundary.

## 1. Rule-based transport for SC and SFP

Use separate route templates for the two task families, with endpoints conditioned on the requested port and, for SFP, requested NIC card. Keep one shared task-conditioned learned insertion actor downstream. SC should retain the lateral bypass rather than reverting to the over-card CheatCode route. SFP can start from its established CheatCode-style transport route; do not assume the SC detour is appropriate for SFP.

The rules should use measured waypoint completion and camera-based acquisition adjustment. Nominal coordinates are a route prior, not the current target's exact location. Substantial board-pose changes may require board-relative adaptation or separate validated templates; moving a waypoint by 10–20 mm cannot be assumed to solve an approximately 79° SC board rotation.

Audit prior tests before repeating them. The current 140-run matrix did not test a rule-based SFP pipeline. Earlier SFP side-route proposals also had cable-load failures, so those are evidence to preserve, not passed solutions. Reuse good trajectories/videos and investigate a small waypoint sweep only where traces establish a concrete clearance or endpoint failure. Freeze the selected route before a fresh five-per-group development comparison. Measure correct requested-port acquisition and physical near-port arrival, not official score alone.

The approach controller also needs a bounded convergence repair: 17/50 primary SC candidate episodes reached roughly 30 mm stand-off from the correct port but failed the tight readiness test. More transport demonstrations are not automatically the remedy for that controller issue.

## 2. Perception coverage and retraining

Checked saved manifests:

| Component | Actual reported training | What it does and what is missing |
| --- | --- | --- |
| Selected joint coarse locator | 319 training episodes, 53 validation episodes; 25,686 training images | Includes SC and SFP. Older internal multi-slot architecture; its projection flags do not certify physical visibility. Counts do not prove complete layout/start coverage. |
| Selected high-resolution crop refiner | 83 training SC episodes, 21 validation SC episodes; 28,149 training camera views | SC-only checkpoint. It is not an already validated shared SC/SFP refiner. |
| Selected spatial identity check and orientation/grasp helpers | Separate checkpoints/contracts | The live verifier and held-plug adapter used in the matrix are SC-specific; a joint locator alone does not provide a connected SFP pipeline. Audit each helper's training and supported inputs separately. |

The correctly identified SC acquisitions had median 0.17 mm and p90 0.23 mm opening-position error. Preserve this refiner as a baseline. The three wrong-port approvals cannot be fixed merely by making its pixel predictions sharper.

Build a coverage table by family, requested card/port, card count/layout, board pose, initial robot/grasp state, route phase, distance, occlusion and neighboring visible ports. Audit image provenance, action/reset identity, crops/calibration, camera preprocessing and episode-grouped splits. Physical occlusion labels must not be replaced with “the projected point falls inside the image.”

The intended coarse model remains requested-target and history-conditioned: task code plus distance-spaced camera history/current robot state, with crop-location and handoff-eligibility outputs. Include wrong-target negatives, hidden/partial targets and ambiguous close views; mask pose/crop supervision where the target is not validly observable. No additional pose-confidence head is proposed. Use the original-image crop refiner and calibrated geometry afterward. Retrain with all suitable training data and targeted gap samples; retain validation and final scenes outside training. If joint SC/SFP fine refinement regresses SC accuracy, compare a shared backbone with family-specific heads rather than discarding the known baseline.

Use new held-out development scenes for the next promotion test. The inspected 140-run matrix remains frozen historical evidence, not a source of unseen-test claims after repairs.

## 3. Broader-start insertion and hierarchical backtracking

### Correct the earlier 8 mm interpretation

The old Isaac SFP experiment used an **8 mm seating-depth success target**. That was not a statement that all training starts were exactly 8 mm before the opening. Its recorded 1/8 versus 3/8 result involved local Isaac starts and a selected RL actor with versus without a local controller; it did not demonstrate hierarchical recovery or official full-depth Gazebo insertion. The older RL-only comparisons also exposed categorical policy collapse, so smooth BC-to-RL transfer must be checked rather than assumed.

### Supervised warm start

Define initial insertion starts from the distribution of correctly acquired transport endpoints. A proposed first envelope is **10–60 mm before the opening**, lateral offsets up to **±10 mm**, and orientation offsets up to **±10°**, sampled feasibly within each scene. These are design candidates, not proven capabilities. Include combined offsets and recovery states, not only easy centered examples, and expand if transport statistics justify it.

Collect sufficient successful complete insertion and post-retreat re-alignment demonstrations across both families. Use predicted perception outputs at the actor interface during training, including their realistic errors. Simulator geometry can generate/check training labels, but autonomous actions and evaluation crops may not consume it. Gate successful teacher labels by physical depth, lateral/orientation alignment and force/contact, not score alone. Preserve originals and clean derivative datasets.

First test wider-start BC with recovery off. This distinguishes a weak approach/alignment policy from a helpful recovery controller. A model that already centers and inserts from a meaningful range gives RL a better warm start.

### Controller responsibilities

Keep hierarchical retreat deterministic and present during both training and deployment:

```text
persistent blockage / excessive load / no measured progress
    -> retreat along measured SE(3) history to a local safe checkpoint
    -> confirm force release and clearance
    -> let the same insertion actor make a coherent lateral retry
    -> repeated failure: retreat to an earlier/deeper checkpoint
    -> outside reliable visual range: return to rule transport/reacquisition
    -> repeat perception and bind a fresh port/plug estimate
    -> stop when distance, force, retry or episode budgets are exhausted
```

A larger retry should return to a stored earlier position/orientation; it should not blindly add more millimeters along a guessed direction. Track the failure region and retry count, so repeated collisions do not simply oscillate between the same two states. Give the learned actor causal motion/force history and recovery context so it can distinguish nominal insertion from a retry. Do not reuse an invalid stale pose after a large retreat or possible grasp change.

Force triggers require orientation/load-aware calibration and measured progress. The roughly 21 N resting wrench seen in past Gazebo traces means raw force norm alone is not a cable-snag detector. Another boundary is simulator completion: if Gazebo has already declared completion and immobilized the arm, the policy cannot retreat afterward. Trigger recovery before that event where possible; do not change official physics/completion rules merely to improve a success count.

### RL and comparisons

After broader-start BC is working, run matched **no-backtracking / local-backtracking / hierarchical-backtracking** comparisons with the same actor and starts. Then use bounded SAC/SERL-style online recovery learning, with expert replay and BC anchoring, preserving the Isaac-to-Gazebo validation order. First verify the image/task/action/force interfaces and relevant collision cases agree enough across environments.

Hardcoded retreat commands must be logged as controller actions, not falsely treated as actions sampled from the actor. Record the actually executed actions and elapsed recovery duration so the critic learns the return of the complete intervention. The actor learns what to do after release and during re-alignment; the retreat rule itself does not need to be rediscovered through penalties. Use complete target-action policies, not a delta correction on a frozen base actor.

The promotion result must be full physical insertion across fresh grouped settings with correct requested-port identity, reported orientation and acceptable forces. An 8 mm Isaac proxy or score 75 on shallow Gazebo seating is insufficient.

## Suggested execution order

1. Audit existing routes, current failures and component/data coverage; preserve frozen comparisons.
2. Validate rule transport and convergence for SC and SFP; use targeted waypoint changes only where supported by traces.
3. Repair/integrate requested-port perception for both families; test correct identity and original-image pose accuracy on held-out scenes.
4. Collect/curate broader-start insertion and recovery demonstrations, then establish a shared BC baseline using predicted geometry.
5. Integrate and instrument hierarchical measured-path recovery; compare off/local/hierarchical controllers before attributing gains to RL.
6. Run bounded recovery RL with replay/BC anchoring and matched controls, then validate in Gazebo without privileged policy inputs.

## Questions to settle

1. Should the insertion actor initially own **10–60 mm** before the opening with the proposed lateral/orientation variation, expanding from measured failures, or own the full approximately **150 mm approach gap** immediately? Recommendation: start with the bounded envelope.
2. After hierarchical retreat leaves the reliable near-port visual region, should rule transport reacquire the target and rerun perception, or should the insertion actor remain responsible throughout? Recommendation: return to rule transport/reacquisition while retaining the same learned insertion actor for subsequent retries.
