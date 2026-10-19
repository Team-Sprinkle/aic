# Requested-port camera-history crop experiment

I implemented the coarse stage you described for **SFP-to-NIC**. It receives the requested card/port code, all three current camera images, earlier images from the same episode at approximately 1.5, 3 and 6 seconds back, and measured TCP motion between views. It predicts **only the requested port**: one candidate crop center and an in-frame score per current camera. It does not emit ten SFP port poses. A future pose-only refiner would receive a 160×160 crop from each original-resolution image at those centers.

I kept the refiner disconnected while testing whether the coarse crop was safe. A wrong crop could show the neighboring port and cause a precise refiner to return a precise pose for the *wrong* opening.

## What was tested

- A pooled-history model trained on **50** complete teacher episodes.
- A version that predicts one shared 3D requested-port point, then projects it into all cameras.
- A spatial-history version that preserves a coarse image grid from earlier views.
- The spatial version retrained with **ten additional, existing five-card teacher episodes** (one for each requested SFP port). These filled a real training gap: the initial 50 episodes had no five-card example for some requests.
- The same 60-episode spatial version with explicit pixel coordinates added to its decoder.

All variants used one GPU, at most 1,000 updates each, the same ten complete development episodes for checkpoint choice, and a separate ten-scene stress replay that had already been inspected in prior work. No final scenes or new simulator episodes were used.

## Result

Adding five-card data helped, but the coarse model still confused repeated ports. On the inspected stress scenes, the final coordinate-aware version proposed **572** crops; **173** had a center closer to another port than the requested one, and only **303/572** put the requested opening within 60 pixels of the center in all three cameras. Its in-frame scores often stayed high on these mistakes, so that score is not reliable permission to hand off.

A strict check based on whether the three predicted points agree with measured camera geometry rejected all wrong centers in this replay. It also rejected most correct opportunities: the final version acquired only **3/10** requested ports within the early three-second window, the same count as the earlier selected-port model. The five-card spatial version without explicit output coordinates reached **4/10** early, still far below the **9/10** target. Turning history off changed some later accepts, but did not produce a reliable early gain.

The attempt therefore did **not** establish a safe coarse crop. I did not connect the refiner, insertion actor, or RL. The new one-requested-port design remains the right interface to test; this checkpoint is not ready to deploy.

## Why history alone did not solve it

The recorded camera history begins with the teacher observation episode, when the TCP is already about **18 cm** from the requested opening on average. In at least one inspected five-card first view, only part of the card row is clear. Later images cannot restore a wider view that was never recorded. Other hard scenes do show the row, so this is not the only source of errors. There is also limited variety across complete five-card reset groups.

The available training label says whether the simulator's port location *projects inside the image*. It does **not** certify that the requested opening can actually be identified under occlusion or among similar card faces. Training an in-frame score with that label does not produce trustworthy visual confidence.

The next upstream experiment should first verify a safe, wider observation pose that shows enough of the board to identify the requested numbered slot. We should annotate target identity and actual visual availability independently of model predictions, including empty slots and occlusions. Then train the same requested-port crop interface and evaluate on **new episode-grouped nonfinal scenes**, requiring zero accepted wrong ports and at least 9/10 early acquisitions before attaching a pose-only refiner. The previously inspected ten-scene replay must not be used for promotion.

Commands, per-variant artifacts, failure details and the machine-readable comparison are in the [experiment record](../experiments/2026-10-04-requested-history-crop.md) and [summary](../experiments/2026-10-04-requested-history-crop.summary.json).
