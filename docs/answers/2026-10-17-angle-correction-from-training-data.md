# Why would the next prediction correct the angle? A training-data explanation

**It would do so only if the current observations resemble training examples whose recorded next commands correct the angle.** The actor does not have an explicit rule saying “I hit something, so rotate,” and it does not learn new behavior during the rollout.

## What the teacher demonstrates

The SFP approach teacher uses privileged geometry to align the actual plug before advancing. Its recordings therefore contain pairs like these illustrative examples:

| Recorded situation | Recorded teacher command |
| --- | --- |
| Plug is tilted relative to the port | Rotate toward alignment; restrict forward advance |
| Plug is aligned | Advance toward/into the opening |

BC trains the actor to predict those recorded commands from its available inputs. Teacher geometry supplies the collection controller's decisions; the autonomous actor must use images, robot state and estimated poses.

So a later prediction might produce a rotation because its inputs resemble the first kind of example. **It is not necessarily recognizing contact or deliberately recovering from a collision.**

## Where gripper-only orientation can make this ambiguous

Consider two training episodes:

- Episode A: the gripper faces straight toward the port, and the plug is held straight. The teacher commands forward motion.
- Episode B: the gripper also faces straight toward the port, but the plug is held2° crooked. The teacher commands a rotation.

If the plug angle is not explicitly represented, the orientation-related robot inputs can look the same while the labels differ. Images can still distinguish them, so the actor is not fundamentally unable to solve it—but it must learn that visual distinction implicitly.

If its interpretation changes between observations, successive outputs could switch between advancing and correcting. Explicit plug orientation makes the relevant distinction easier to represent. This remains a possible mechanism, not a demonstrated explanation of our recorded jerk.

## Correction to my previous example

My earlier sequence—“push, catch an edge, then correct the angle”—was hypothetical. **We have not shown that the dataset contains this exact contact-recovery sequence or that the successful rollout used a learned version of it.** The teacher's alignment gate often avoids that sequence by aligning before advancing.

The supported claim is narrower: the data contains ordinary alignment supervision, and the new representation improved SFP command consistency in the small comparison. That does not prove learned collision recovery or establish the cause of every correction.
