# The SC port-identity experiment, from the beginning

## What the robot is trying to do

The robot holds a cable plug and must insert it into the **requested** socket. There are two task families: SFP-to-NIC and SC-to-SC. The task code tells the robot which kind of task it has and which card and port to use. Several similar-looking sockets may appear in its three wrist-camera images.

Our planned controller has two movement models and a perception step between them:

```text
Start far away
   |
   v
Transport model moves toward the requested socket
   |
   v
Transport proposes: "I am close enough to inspect"
   |
   v
Perception checks the requested socket's identity, visibility and pose
   |
   +-- uncertain or unsafe --> do not start insertion
   |
   v
Insertion model makes the precise final movement
```

The transport model must work even when the socket is initially offscreen. Near the socket, perception should locate **the requested opening**, rather than whichever opening looks easiest to see. The insertion model would then express its actions relative to that visually estimated opening. Gazebo's true socket coordinates are used to make training labels and score experiments; they are **not given to the model during its predictions**.

## Why we ran this particular test

An older vision model could estimate the position of a clearly visible SC opening quite precisely. But we found a serious identity error. In a scene with only SC port 0 physically present, we changed the request to the nonexistent port 1 while keeping the camera images identical. The old three-camera system accepted all **16 of 16** such requests and pointed to visible port 0. Its camera predictions agreed with one another, but they agreed on the **wrong port**.

That counterfactual request is a stress test. A normal task requests a port that exists. The test asks whether the vision system truly follows the requested identity or merely locks onto a nearby blue connector. The older SC training images all showed both ports, so they had not taught it this distinction well enough.

## What we changed

We made **12 new one-port SC simulator configurations** with different board positions, rail positions, card arrangements and requested ports. Eight were assigned to fitting; four different complete resets were held out for checking the result. We also used existing two-port images so the model had to deal with neighboring sockets. These runs supplied camera images and geometry labels for **perception**. They were not successful expert insertion demonstrations.

We compared two small models under the same data and training budget. Both receive the image, the requested port code, camera identity and the robot's measured tool pose:

- The **pooled model** compresses the image to one overall summary before deciding whether the requested port is in view.
- The **spatial model** keeps a small grid of image features before deciding. It can use *where* a port-like object appears and how the surrounding scene is arranged.

The models were trained to predict whether the requested port's opening **projects inside the image**. That is an imperfect substitute for visibility: a port may project inside the picture while the gripper or cable hides it. We have **not** trained a certified “I can clearly see the opening” output yet.

## What the numbers mean

For every saved image, we tried both requests: the real port and the physically absent port. At a decision time, the robot has three camera views. We used a strict rule: **all three cameras must score the requested port as present**. The scores below come from four held-out simulator resets; many decisions are neighboring frames of the same episode.

| Question on those held-out frames | Old-style pooled model | New spatial model |
| --- | ---: | ---: |
| How often did it wrongly accept a request for the absent port? | 32 of 109 decisions | **0 of 109** |
| How often did it accept the real port when its opening projected into all three views? | 34 of 100 decisions | **96 of 100** |

So the spatial information greatly helped the model tell **which SC port was requested**. “0 of 109” does **not** mean zero failures in production: it is a small development test on four reset configurations. “96 of 100” is also not an insertion success rate; it only measures acceptance of the real requested port under a projection-based label.

The strict three-camera rule matters. The spatial model still made mistakes in individual camera images. In a two-out-of-three version, it falsely accepted the absent port **2 of 109** times. All-three agreement removed those false accepts in this test, but it can reject a real port when one camera is blocked.

## Identity is different from visibility

We also replayed 20 older examples where the target was provisionally judged hidden. The new identity model **alone accepted 4 of 20**. The existing check that compares precise keypoint predictions across cameras rejected all 20; requiring **both** checks also accepted none.

This is why the new result is best described as an **SC port-identity improvement**. It has not proved that the system can always tell when the opening is actually visible, especially near contact when the plug or gripper covers it. Those 20 hidden labels are from a small older review, so they are not a complete hard-occlusion benchmark.

## What happens now

The SC identity subtest looks promising, but the automatic switch from transport to insertion remains **off**. We still need to check reviewed clear and obstructed SC images, the known wrong-route case, and the saved **SFP-to-NIC wrong-port** case. Then we need to test the entire transport → perception → insertion handoff live in Gazebo, including safe robot posture and force, and measure complete inference latency below 300 ms. The measured **11.19 ms p95** so far covers only preloaded vision components, not the full live robot loop. No new insertion actor or RL model was trained from this identity experiment, and the reserved final scenes remain sealed.

The [experiment record](../experiments/2026-10-02-sc-absent-port-identity.md#spatial-selected-port-continuation-on-new-reset-groups) contains the exact scene manifests, model checkpoints, per-frame errors, commands and latency measurement.
