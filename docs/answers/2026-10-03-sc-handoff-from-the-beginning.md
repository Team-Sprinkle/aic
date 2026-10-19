# SC handoff problem, from the beginning

## What the robot is trying to do

In the SC-to-SC task, the robot holds a cable plug and must insert it into a **particular requested port**. There can be two similar-looking SC ports. Its three cameras move with the robot's tool.

We first collected simulator demonstrations and trained a behavior-cloned transport policy to follow a route around the cards. We trained camera-based port perception separately, using simulator geometry as supervised labels. The longer-term plan is to let a separate insertion policy take over near the opening, and only later train recovery with RL. The recent runs deliberately isolate the **transport-to-insertion handoff**.

Our intended controller has three stages:

```text
Task says which port to use
          |
          v
Transport policy moves the plug near that port
          |
          v
Camera-based check locates and verifies that exact port
          |
          v
Insertion policy makes the final precise motion
```

The transport policy has reached the neighborhood of the port on some development runs. The camera-based check is supposed to decide whether the robot is close enough, facing the opening correctly, and looking at the **requested** port. Only then would the insertion policy take over. That automatic handoff has **not** passed its safety and reliability tests, so the insertion policy is not being used in these recent runs.

## Why “I can see an opening” is not enough

Imagine the task requests **port 0**, but only **port 1** exists in the scene. A camera model might accurately point to port 1's opening. That is still the wrong answer. The check must say **“requested port absent”**, rather than hand a precise pose for the wrong port to the insertion policy.

The current check first predicts possible port locations in the camera images. It then uses the requested-port code, the images, and the robot's measured tool pose to judge whether the requested port is actually there. If that passes, it combines camera views to estimate the opening's 3D position and board angle, and checks the robot's position, orientation, and force. The simulator's true port coordinates are used later to score experiments; they are not given to this live check.

## The problem we just investigated

On a port-0 run, transport got close to the desired waiting position. Its orientation controller thought it was aligned, but the images showed that the tool was facing about **6.7° away** from the port. The robot was following its command accurately; the **commanded orientation was wrong**.

I added an experimental rule: while holding near the port, use the three camera views to measure the board angle and rotate the tool toward it. This reduced the measured orientation error from **6.73° to about 0.1°** on the known scene, and from **8.93° to about 0.1°** on a second five-card scene. The rule still required a fresh camera check and did **not** authorize insertion by itself.

After this useful rotation, the existing port-identity model often said that the real port was **not** there. So we solved the facing-direction error but still could not safely hand control to insertion.

## Why the identity model changed its mind

The identity model receives the robot's measured orientation as an input. I tested its sensitivity by keeping one saved camera image fixed and changing only that orientation input. Its port-0 scores dropped sharply. Changing only the robot position barely affected them. This tells us the model relies heavily on orientation when making its port-identity decision.

That fixed-image test is a **diagnostic**, not a valid training example: the cameras are attached to the tool, so a real tool rotation also changes what they see. The real before-and-after camera images nevertheless showed the same practical problem: after the tool rotated into a better alignment, the identity model rejected the requested port.

I also trained a version of the identity model that does not receive tool orientation. It recognized the corrected real port in saved images. But on a new scene it incorrectly said a **physically nonexistent port** was present in **11 out of 100** three-camera moments. The old model made **0 out of 100** such mistakes there. That new model cannot safely replace the old one.

## What I recommend next

Collect a small, targeted training set where the robot really rotates near a port and records the **new three-camera images and matching measured tool pose at each angle**. Include both requested ports, scenes where the other port is nearby, and scenes where the requested port is physically absent. If the requested port remains visible before and after a rotation, its identity label should stay the same; if it is absent, the answer should remain “absent.”

Train two otherwise matched versions of the current spatial identity model on those real view pairs: one with tool orientation available and one without it. Use new complete scene configurations for evaluation. Require **zero accepted absent-port requests on every held-out scene**, at least **95% recall on manually reviewed clear arrivals**, preservation of the earlier good and wrong-port checks, and recognition of the corrected port-0 views. Budget the two fits to at most **two GPU-hours**. If absent-port confusion remains, stop refining this classifier and test a more explicit camera-based detector of the board rails and requested port.

The robot's measured orientation will still be used where it belongs: converting camera measurements into a 3D port pose and commanding motion. We are testing how much it should influence the separate **“is this the requested port?”** decision.

There has been **no autonomous full insertion or RL improvement** from these recent tests. One new-reset baseline run also took a different route and stopped far away, so its difference from the corrected run does not prove the correction improves transport. The handoff, insertion and RL gates remain closed, and the reserved final scenes remain sealed. The [experiment record](../experiments/2026-10-03-sc-orientation-identity-repair.md) links the inspection images, traces and exact scores.
