# Summary: insertion repairs, scores, and remaining work

## 1. What works now

The working pipeline is:

**Rule-based transport → camera-based target acquisition → visually guided approach → insertion.**

Transport follows the desired route around the cards for SC. Coarse perception identifies the requested port, and the high-resolution refiner estimates its pose. We also estimate the held plug's pose relative to the gripper: the gripper and plug are not necessarily pointing in exactly the same direction.

**The important recent change:** the successful recent insertion experiments use an explicit controller that moves according to these image-based estimates and measured robot state. They do **not** use the learned insertion BC actor to choose insertion actions. No new RL was trained. Ground-truth geometry is used afterward to measure results, not to steer these autonomous rollouts.

## 2. Complete three-trial evaluation scores

These runs use the original public `sample_config.yaml`: two SFP trials and one SC trial, starting from the original home position. Maximum score is300.

| Version | Score | Passes both physical insertion and toolkit completion |
|---|---:|---:|
| Earlier learned BC insertion baseline |165.42/300|0/3|
| Explicit image-based insertion controller |152.61/300|1/3|
| Same controller, longer seating hold |**176.85/300**|**2/3**|

The scores alone are misleading: the older baseline earned credit despite failing our independent physical checks. The newer controller actually seats both SFP connectors.

After the seating-hold repair:

| Trial | Measured insertion depth | Lateral error | Orientation error | Result |
|---|---:|---:|---:|---|
| SFP1 |46.77mm|0.24mm|0.21°|Pass|
| SFP2 |46.76mm|0.20mm|0.49°|Pass|
| SC |2.01mm|0.23mm|2.18°|Fail: shallow insertion|

These are one complete three-trial evaluation before the repair and one afterward—not repeated full evaluations establishing broad reliability. Final scenes remain sealed.

## 3. What we fixed

### SFP: the robot finished before the simulator acknowledged success

Both SFP plugs already reached the correct depth. However, the simulator requires one second of uninterrupted qualifying contact. Our controller sometimes returned after only0.5seconds of estimated stable seating.

**Fix:** hold stable seating for2seconds. A matched development comparison improved from**1/5 to5/5** combined successes. This repaired completion timing; it did not teach a new insertion motion or change simulator physics.

We also removed duplicate image JPEG conversion between the plug-position and plug-orientation estimators. Predictions stayed identical in the equivalence check. Three live SFP repeats all inserted, with inference p95 around**220–228ms**. Those repeats cover one development configuration, not all settings.

### SC: backtracking commands initially did not produce real backtracking

Several issues appeared in the traces:

- The original force threshold did not trigger on low-force blockage.
- A slower blockage detector reacted after the simulator's one-second contact latch had already frozen the connector.
- Under the robot's impedance controller, commanding a previous measured position did not necessarily move the tool there: command and measured position differ under load.
- One compensation implementation accumulated error and overshot during lateral retry.
- Ordinary insertion control then steered back toward the original center, undoing the lateral exploration.

**Fixes:** detect sustained near-port blockage earlier; compensate using public command-versus-measured tracking; reset accumulated control error between phases; move toward fixed retry goals; preserve the selected lateral/orientation goal during the next insertion attempt. Real retreats and lateral retries now occur. A sensitive detector is restricted to the contact region so it does not interrupt normal free-space alignment.

## 4. How close SC got

The latest single development pilot, using local backoff and a persistent pose goal, reached:

- **15.43mm insertion depth**
- **0.37mm lateral error**
- **1.12° orientation error**
- Toolkit insertion credit: **75points** for the insertion category, not a complete300-point evaluation.

This is substantial physical progress from the earlier roughly2mm stop. However, it still fails our unchanged physical criteria of at least15.5mm depth, at most0.5mm lateral error and at most1° orientation error. It is also only one pilot. **SC reliability and the benefit of hierarchical backoff are not established.**

The retry calibration was fitted from training episodes, not the current test episode's geometry. Several controller changes were combined, so we cannot attribute the progress to backoff alone. Simply increasing insertion speed6→8mm/s did not reliably help.

## 5. What did not work

- The learned BC actor's matched controller-context comparison achieved**0/8** combined insertions.
- Its no/local/hierarchical-backoff comparison achieved**0/12**. No retreat actually triggered, because these policies mostly stopped short with low force. This does not demonstrate that backoff is ineffective after genuine contact.
- Multiple SC plug-pose estimators trained through validation plateaus still failed held-out accuracy. They were not promoted. The issue was not simply stopping training early.
- Edge-based pose registration and a simpler landmark readout did not consistently fix SC generalization.

The BC training set contained499usable episodes, including132SC episodes. The expanded SC perception dataset contained174episodes, but only**four training and two validation episodes** covered the nominal grasp at the actual near-acquisition viewpoint. Most new nominal captures were at home. Thus total dataset size overstates coverage of the particular view where runtime SC estimates are needed.

## 6. Remaining issues and recommended fixes

**First: finish ruling out simulator-version/order effects.** Other teams reported partial scores and confirmed historical phantom-collision problems. Both announced fixes are already in our source history, but the frozen simulator binary's exact patch provenance remains unverified. If necessary, compare the identical SC scene alone versus after two SFP trials, without changing policy or physics. Our fresh standalone failures already show that trial-order bugs cannot explain everything.

**Second: improve SC plug-pose accuracy at the actual acquisition view.** The roughly2° orientation error measured during shallow failures is a concrete problem. If the simulator check does not explain it, collect a bounded set of perception-only captures at that viewpoint, including different grasps and both SC ports. Split by complete episode/reset, inspect labels and images, and require held-out translation/orientation accuracy before deployment. This is a targeted coverage repair, not another indiscriminate training sweep.

**Third: validate the repaired controller across fresh development scenes.** Test both ports and different card counts, count actual retreats, and compare no backoff, local backoff and hierarchical backoff under matched conditions. Then rerun the unchanged full three-trial sample. Do not lower physical thresholds to turn the latest near-pass into a success.

**Current conclusion:** SFP insertion and score acknowledgement have improved. SC now demonstrates real withdrawal, retry and near-full insertion, but it is not yet a reliable solution. Nothing is currently training or evaluating in the background.

Detailed commands, artifacts and failures: [experiment record](../experiments/2026-10-18-three-trial-insertion-repair-cycle.md).
