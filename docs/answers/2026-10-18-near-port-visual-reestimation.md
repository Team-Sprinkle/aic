> Completed follow-up: [close-up re-estimation results](2026-10-18-close-up-reestimation-results.md). Earlier running/queued paragraphs below are historical.

# Re-checking plug orientation at 25 mm

## Why this can help

The controller knows the robot's TCP pose, but estimates the held plug's pose from images. An error in that estimate can make the real plug tilted while the controller thinks it is straight. A later close-up gives another opportunity to estimate it. In four checked Gazebo episodes the actual plug-to-gripper pose hardly changed; the present evidence points to visual estimation bias rather than slip. Gazebo fixes this endpoint to the gripper, so these tests cannot establish real-world slip recovery.

## Proposed runtime behavior

1. Use the existing transport, requested-port acquisition and approach to reach approximately25mm before the opening.
2. Hold the **last commanded robot pose**. Holding the measured pose instead can change the physical equilibrium of the compliant controller.
3. While nearly still and unloaded, collect five fresh, synchronized three-camera observations. Reject stale images and duplicate timestamps.
4. Require the estimated plug positions to agree within0.2mm and orientations within0.5°. This tests consistency; it does not prove accuracy.
5. Accept a change from the initial estimate only within1mm and3°. A larger change triggers a separate five-observation window. If the large disagreement persists, stop safely; automatic retreat/reacquisition in that rejection branch is not yet implemented.
6. Update the held-plug pose once, leave the accepted port pose unchanged, and restart the smooth command generator from the same issued reference. Keep this estimate fixed during the insertion attempt. A fresh route acquisition requires another close-up check.

The close model uses RGB only. Simulator poses provide training labels and independent scoring, never runtime predictions or control targets. Translation and orientation may come from separately qualified models; the bad orientation output of a position-qualified model is discarded.

## Current evidence

The largest completed orientation fit uses80 independent near training grasps and reaches a validation plateau after18,400updates. Its held-out orientation p95 improves1.587→1.026degrees on four unchanged near validation episodes. It still fails the1degree qualification gate. Five-observation averaging removes little of the systematic error, so it is not promoted by rounding or selecting a favourable statistic.

Two bounded training variants now test finer image features on the same data: jointly adapt the existing model, or freeze it and train only the additional detail path. Both must finish their LR-floor plateau and pass accuracy, runtime-parity and image-review checks before a live25mm update is enabled. No new collection is queued.

The prepared live test reuses the failed zero-card/SC-port0 development start, then proceeds to the unchanged public three-trial sample after a physical insertion success. Latest complete public scores remain257.76/300and257.65/300, with bothSFP insertions passing andSC stopping around15.48mm, below15.5mm. No new RL training is involved. Final scenes remain sealed.

[Full experiment and artifacts](../experiments/2026-10-18-sc-explicit-controller-continuation.md).


## Independent replay and limited live test

The80-grasp model was subsequently replayed on four different frozen development rollouts, checked disjoint from training. At25mm it gives0.717°orientationp95 and0.129mmpositionp95. This supports one bounded live diagnostic even though the original validation subset still fails1.026°. The experiment is explicitly unqualified: certificate `pose_gate_passed=false`, one exact failedSC start, no general deployment or final-scene access. A physical insertion result is still pending. [Explicit plan revision](../experiments/2026-10-18-sc-explicit-controller-continuation.md#explicit-plan-revision-one-unpromoted-live-diagnostic-after-independent-replay).


## First live result

The close-up update worked on the previously failed SC development start:15.524mm depth,0.114mm lateral error,0.688°orientation error,toolkit75and no scored harm. It updated the visual plug estimate by2.574°and0.339mm after five agreeing observations, then completed one local retreat/lateral retry. Complete update p95was102ms. This is one explicit-controller success, not trained-actor/RL learning or broad reliability. The unchanged three-trial public sample is running next.


## First full public sample and ablation

The unchanged public sample now passes **3/3 strict insertions**, scoring257.36/300. SCreaches15.518mm depth,0.239mm lateral and0.520°orientation,toolkit75/noharm. The score barely changes because the old shallowSCrun already received toolkitcredit; independent physical depth is what improves. Two identical repeats are underway.

In one matched development pair, disabling only the close-up pose update makes the sameV21controller fail at9.30mm depth/2.27°orientation. With the update it passes15.52mm/0.69°. This supports the full pose update, but does not isolate orientation from translation or establish a success rate. Three remaining matchedSCcoverage cases are queued with a stop-on-first-failure rule.
