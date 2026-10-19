# How the current transport system actually works

## What happens in an autonomous rollout today

```text
Three robot-camera images + measured robot state + requested task code
                         |
                         v
            Transport actor with image history
                         |
                         v
        Four short target gripper poses in the gripper's frame
                         |
                         v
                 Robot controller moves
                         |
                  Repeat next decision
```

The **task code** says which task is being done and which SFP card and port is requested. The measured state tells the actor where the gripper is and how it moved. The actor also receives prior commands and a history of image features. Its vision backbone compresses each full camera image into a feature vector; a small temporal transformer combines those vectors with state, task code and motion history. It predicts four short future gripper targets **relative to the current gripper**, rather than absolute world coordinates. The controller executes the commanded motion, new images arrive, and the actor repeats.

The current actor has **no explicit output saying “this pixel is the requested port”**. It must learn the requested route indirectly from the camera vectors and task code. That is the part that currently works poorly: it improved its average imitation error, but missed the safe requested-port approach region on all three recent autonomous starts.

## What the port-locator test did

After those rollouts, we took their **saved camera images** and ran separate, frozen perception models on them:

```text
Saved original camera images + requested card/port code
                         |
                         v
        Port locator predicts a pixel and in-frame score
                     in each camera
                         |
                         v
        Three-view geometry estimates one coarse 3D position
```

One locator predicts the requested port directly. Another predicts a structured set of ports internally, then selects the requested one using the task code. The measured gripper pose and camera calibration let us combine the three image points into a coarse 3D estimate. The acceptance check requires an in-frame score of at least 0.5 in all three views and agreement between their projected rays. That score is **not a reliable measure of whether the port is truly visible through occlusion**. True simulator port position is used **afterward to score the answer**, never to make the prediction.

On the three one/two-card actor routes, these separate locators found the requested port early. **Their estimates were not fed back to the actor during those rollouts.** Thus the result says “the images contained enough information for these locators in these scenes,” while the actor still failed to use its own visual input to choose a good route.

## What is connected and what is still experimental

| Piece | Runs in today's autonomous transport? | Purpose |
| --- | --- | --- |
| Transport actor | Yes | Chooses short gripper movements from images, state, task code and history. |
| Frozen port locator | No; replayed separately | Estimates which image point is the requested opening and a coarse 3D position. |
| Learned transport stop | No | Was meant to propose when to pause; its timing gate failed. |
| Original-image pose refiner | No | Would estimate insertion-grade position and orientation after a safe coarse proposal. |
| Near-port insertion actor | No | Would align and insert after the refiner is trusted. |
| RL/recovery | No | Deferred until transport, identity and handoff work reliably. |

The main gap is therefore the **connection and reliability**: the actor does not consume the locator's spatial answer, and the locator itself is not yet reliable across harder middle-card and five-card views. The transport-to-insertion handoff also has no trusted automatic stop. We are checking those upstream pieces before joining them into one automatic system.

For the detailed scores and saved images, see the [spatial prerequisite audit](../experiments/2026-10-05-spatial-transport-prerequisite.md).
