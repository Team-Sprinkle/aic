# Transport orientation, image correction, and the original verifier

## 1. What does “facing 6.7° away” mean?

Picture the plug almost at the port opening, but turned slightly relative to the direction in which it should enter. The **6.7° was that rotation difference** between the measured tool orientation and the desired insertion orientation inferred from the board edges in camera images. It was not a 6.7° error in the direction of travel, nor evidence that transport went to the wrong port. In the known port-0 run, the closest measured distance to the intended standoff was about **0.66 mm**.

You are right that the transport actor has **no separate head that explicitly estimates the target port's orientation**. It does, however, output a tool-rotation command as part of every 9D body-frame action. Near the port, the tested pipeline switches to a visual standoff servo. That servo replaces the actor's action chunk with small position and rotation steps, using an orientation probe blended with a learned/default orientation. So the precise diagnosis is that the **late approach's chosen orientation was wrong for this particular board pose**. The robot followed the issued rotation closely, within roughly 0.2–0.3°; the issued orientation itself was about 6.9° from the independent image-derived target. The 6.7° number describes the resulting posture at the inspection point, not necessarily an error made by the early transport actor.

## 2. Which model did the correction change?

It changed the **experimental handoff/controller logic**, not the weights of the perception model. The existing three-camera verifier can locate the selected opening and estimate the board's orientation from real images. Previously, if its selected-port identity check failed, it stopped before calculating a usable orientation correction. I added an optional path that still calculates that geometry after an identity failure, provided image agreement, distance, force, and angle checks pass. It then holds position, rotates the TCP toward the image-derived insertion orientation in steps of at most 2°, takes fresh camera images, and checks again. The failed identity check remains failed; rotation alone cannot authorize handoff to insertion.

There was already a correction path for the case where identity passed but posture was wrong. The new five-card run used **that existing path**. The known port-0 run exercised the newly added path after an identity miss. Both paths brought measured posture close to the image-derived target: about 6.73° to 0.13° in the known run, and 8.93° to 0.03° in the new run. **Neither run passed the subsequent identity check or inserted.** These are diagnostics, not a promoted deployment behavior.

I also trained a **separate offline test identity head** that ignores the TCP quaternion. That was an experiment to understand the identity failure; it was not the correction used by the live controller and was never deployed.

## 3. Did we reconstruct or recreate images?

**No.** Gazebo rendered fresh images from the three real simulator cameras after the robot physically rotated. We saved and inspected those before/after views. The opening and board-edge measurements used the observed RGB images; there was no decoder, image reconstruction, or predicted future frame in this test.

One diagnostic took an **unchanged saved image** and changed only the numerical TCP quaternion fed to the identity head. Its score changed sharply. This exposed sensitivity to the quaternion, but the image/state combination was physically inconsistent because the cameras move with the tool. We did **not** use those artificial combinations for training or control. The offline identity-head refit used captured images.

## 4. Why not use the original model if it had 0/100 errors?

The **0/100** result answers only one question: on one held-out reset, when the requested port was physically absent, the original verifier never falsely accepted it in 100 synchronized three-camera samples. That is good evidence for that particular rejection test, but it is not a universal error rate or a measure of whether it recognizes a *present* port.

The other side of the gate failed: after the tool was correctly rotated in two real-port runs, the original identity head rejected the visible requested port. In the known port-0 run, its three camera scores changed from roughly **0.190/0.645/0.938** before rotation to **0.003/0.045/0.848** after rotation. It requires all three cameras to pass, so there was no insertion handoff. Keeping it as a conservative reference is useful, but using it unchanged for automatic handoff would leave the robot waiting at some valid ports. Ignoring its rejection would defeat the wrong-port safety check.

The offline quaternion-masked alternative recognized more present-port views but falsely accepted a physically absent port in **11/100** three-camera samples on the new reset. So neither version currently passes both requirements: **reject absent or wrong ports, and accept the correct visible port after alignment**. The next bounded test uses physically matched before/after-rotation camera views to train and check that distinction. Final scenes remain sealed.

See the [orientation and identity experiment](../experiments/2026-10-03-sc-orientation-identity-repair.md) for traces, images, commands, and exact comparison files.
