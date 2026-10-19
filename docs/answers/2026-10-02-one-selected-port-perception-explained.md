# Train perception for one selected port

**Recommendation: do not train the new near-port model to output poses for all ports.** Give it the camera images **and the task code**, and ask it for the pose of **only the requested port**.

For example, an image shows two SC ports:

| Training input | Training answer |
| --- | --- |
| Same image + task says `SC port 0` | Position and orientation of **port 0**. |
| Same image + task says `SC port 1` | Position and orientation of **port 1**. |

The model has **one selected-port output**, not one output for every port. Showing it the same image with different task requests teaches it to distinguish the ports. The other visible port is a *wrong answer*, not another pose that it must output.

If the task requests port 1 but port 1 is hidden while port 0 is visible, the correct training answer is **“requested port not reliably visible; no valid pose.”** We train that rejection explicitly. When the requested port is visible, the training answer includes its location in the camera images and its pose relative to the robot tool. The simulator can provide those labels during training; its port coordinates are never supplied to the deployed model.

At deployment, transport brings the tool near the requested destination and pauses. The perception model receives the three camera images, measured tool pose, and task code. It either returns **the requested port's estimated pose** or says **“cannot identify it reliably.”** Only a verified estimate allows insertion to start.

The **current** coarse locator is different: it predicts candidate locations for all 12 possible port identities and selects one afterward. I recommend replacing that as the *precise near-port decision* with the task-conditioned, **one-requested-port** model described above. A small internal detector may still notice another nearby opening to avoid confusing it with the target, but it does not need to output that opening's 3D pose.

True visibility/rejection and calibrated accuracy are **training goals**, not reliable outputs of the current checkpoint. We would use close-up examples from both task families, including neighboring ports, hidden targets, and wrong-card requests, and evaluate on held-out complete scenes.
