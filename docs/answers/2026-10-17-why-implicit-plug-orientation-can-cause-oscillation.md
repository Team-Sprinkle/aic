# How gripper orientation could contribute to jerky insertion

**Gripper orientation is valid information. The possible problem was leaving the plug's orientation implicit—not that gripper orientation is inherently wrong or that the old coordinate conversion was proven incorrect.**

## A concrete example

Imagine the plug is held **2° tilted** relative to the gripper. These numbers are illustrative.

1. **The gripper looks aligned, but the plug is not.** The actor sees the gripper facing straight toward the port and commands forward motion. The tilted plug catches an edge.
2. **The next prediction reacts to the changed situation.** From the images and robot motion, the actor produces a sideways or rotational correction.
3. **Rotating the gripper also moves the plug tip.** The wrist rotates around the robot's tool reference, not around the plug tip. Correcting the angle can therefore introduce a sideways tip displacement unless the translation compensates for it.
4. **The following prediction tries to fix that displacement.** If the actor has not learned a consistent relationship between plug angle, gripper angle and translation, the new short plan can disagree with the previous plan. Repeating this can produce alternating corrections.

For scale, a point40mm from the rotation reference moves about1.4mm sideways under a2° rotation. This is an illustration, not our measured lever arm or an explanation of the entire15mm command jump.

## Why explicitly estimating plug orientation might help

Now the actor is explicitly told: **“The plug is tilted2°, even though the gripper looks straight.”** Its targets describe the actual plug pose. Command conversion accounts for the estimated plug-to-gripper relationship, coupling the required rotation and translation consistently.

That can make it easier to learn one coordinated correction, instead of repeatedly compensating for consequences of a poorly inferred grasp relationship.

The previous runtime already accounted for the estimated tip-position offset; it was not simply rotating the wrist and ignoring all tip motion. The full-connector change makes connector orientation explicit throughout conditioning, training targets and decoding. The example describes a possible learned-control inconsistency, not a demonstrated missing-translation bug.

## Why this is only a hypothesis

Different episodes can hold the plug at different angles. Without an explicit estimate, the actor must infer that hidden relationship from images and history. If that inference is inconsistent, short plans may disagree. But a sufficiently trained model can learn the relationship, and a constant2° offset could instead cause steady misalignment with no oscillation.

**Therefore, missing explicit plug orientation could contribute to jerk; it does not inevitably cause jerk.** Our SFP improvement is consistent with this explanation, but does not prove it. Controller handover jumps, teacher-command ambiguity and blocked forward demand remain separate explanations in the recorded traces.
