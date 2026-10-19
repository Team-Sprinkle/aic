# Learned insertion actor versus explicit image-based controller

**Yes—the “BC insertion actor” means our PoseInsert-inspired learned policy.** They are two different ways of choosing insertion movements. I should have made this change clearer when reporting the results.

## 1. BC insertion actor: a neural network chooses the movement

BC means **behavior cloning**: train a policy to imitate teacher actions.

Our recent insertion actor is a trained transformer. It receives camera images, robot-state history and task information, with image-estimated port/plug poses used to express and decode its port-relative action targets.

At runtime:

```text
Images → perception models → estimated port and held-plug poses
                                |
Images + robot history + task → learned insertion actor
                                |
                         predicted action chunk
                                |
            convert port-relative targets to robot commands
```

For example, the network predicts a sequence of movements that should align and insert the plug. Those choices come from training on demonstrations. It is **PoseInsert-inspired**, not an exact reproduction of the paper; this recent actor is a transformer, not a diffusion actor.

## 2. Explicit image-based controller: code chooses the movement

Here we retain the trained perception models, but **bypass the learned insertion actor**.

```text
Images → perception models → estimated port and held-plug poses
                                |
Measured robot pose + force → programmed alignment/insertion rules
                                |
                           robot commands
```

The controller explicitly does things such as:

1. Estimate where the plug tip is relative to the opening.
2. Move sideways and rotate toward the estimated alignment.
3. Advance once estimated lateral and angular errors are small enough.
4. Check measured progress and force; stop or recover when appropriate.
5. Hold estimated stable seating long enough for completion acknowledgement.

For example: **“The estimated tip is2mm left of the opening: move toward the opening center. Once aligned, advance along its axis.”** The movement follows code and control equations, rather than an action network's prediction.

“Image-based” describes where the target and held-plug geometry come from. **It does not mean we know the true simulator pose.** For the recent controller, those image-derived poses are normally locked for an insertion attempt. The control loop updates measured robot state and force about every50ms; it does not rerun and replace the visual pose every50ms. Route-level recovery can reacquire it.

## 3. What the reported success actually proves

| Component | Recent successful explicit-controller tests |
|---|---|
| Rule-based transport | Used |
| Learned coarse/refiner and plug-pose estimation | Used |
| Learned PoseInsert-inspired insertion actor | **Bypassed** |
| Explicit alignment/insertion control | **Used** |
| Ground-truth geometry for autonomous movement | Not used |
| New RL training | None |

**The176.85/300 result and both successful SFP insertions belong to the explicit-controller pipeline. They are not evidence that our learned insertion actor improved.**

I introduced this as a diagnostic alternative because the BC actor was failing: can our perception estimates support insertion if a simple controller chooses the actions? For SFP, the answer is encouraging. For SC, pose accuracy and contact handling remain problems.

This is a change in the insertion action-selection method, not merely a smoothing wrapper around the original actor. We should keep both approaches clearly named and evaluated separately. It does not establish that the explicit controller can handle all cable dynamics or replace future learned recovery.
