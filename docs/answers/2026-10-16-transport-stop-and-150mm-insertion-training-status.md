# Transport stopping and the 150 mm insertion actor

## 1. When do we check the 130–180 mm distance?

**After transport/acquisition, before any learned insertion action.** No PoseInsert-inspired insertion actor ran in the current evaluation.

The current runtime is:

1. Follow the rule-based transport waypoints toward an approximate 150 mm stand-off.
2. Run requested-port coarse perception and high-resolution refinement during transport, including at the final acquisition stage.
3. At that stage, use a fresh accepted visual pose to adjust the final holding position. If acquisition is missing, perform a bounded viewpoint search; SC can retreat along its measured path and try the other corridor.
4. Declare handoff only after repeated consistent visual estimates and a stationary hold at the acquisition goal. Otherwise report failure.
5. In this transport-only experiment, end the episode. Independently use ground truth afterward to check the actual distance, lateral offset, orientation and port identity.

Thus transport does not simply finish a fixed movement list and automatically declare success. It includes a visual acquisition controller at the end. The 130–180 mm band is an assessment tolerance around the intended 150 mm handoff, not a distance reached after insertion.

## 2. Are the components all proven reliable?

They are promising enough to prepare the insertion comparison, but not proven for every scenario. At the checked snapshot, the frozen transport/perception evaluation is **97/102 qualified**, with 130 planned. Failures remain, and the batch has not finished. Coarse and refiner success is primarily measured together in this live pipeline; earlier accurate pose errors were conditional on acceptance and do not guarantee that every episode obtains an accepted pose.

Keep the current frozen batch running unchanged. It evaluates transport plus acquisition, not autonomous full insertion. Reserved internal final scenes remain sealed.

## 3. Was the insertion actor retrained for the 150 mm approach?

**Yes. Two new matched shared SC/SFP insertion BC actors completed training.** One starts with fresh action weights and ImageNet vision; the other initializes vision from transport while training its insertion action policy. Each received 30,000 updates. Both use the same corpus: **67,039 action-chunk rows from 500 source episodes / 377 physical reset configurations**. The actor has approximately **13.73 million parameters**.

The intended responsibility is the whole 150 mm approach through alignment and insertion, with wider teacher starts up to 180 mm. Wider teacher demonstrations include lateral and orientation perturbations; those successes are privileged training data, not learned-policy results.

Although both jobs ended at their update caps, the subsequent check found no practical late validation improvement in either arm, so it did not launch extra training. This supports using their selected checkpoints for evaluation; it does not establish optimal training or successful autonomous insertion.

The completed checkpoints have **not yet passed a completed-model autonomous insertion comparison**. The previously queued comparison stopped at its perception gate. Earlier integration rollouts used intermediate checkpoints: neither fully inserted. They are not final results for the completed actors.

## Next step

Finish and archive the current frozen acquisition evaluation. Then run the completed insertion actors in a separate, explicitly versioned integrated comparison using the selected acquisition stack: start from normal reset, transport to approximately 150 mm, reacquire port/grasp, and let the actor own the remaining approach. Begin with BC and no recovery, then compare the same actor with local and hierarchical backtracking. Preserve the original failed gate and distinguish this new pipeline revision from the older blocked comparison. More training or RL should follow a concrete diagnosis of these completed-model rollouts, not an assumption that training has already succeeded.
