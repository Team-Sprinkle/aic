# How the current system works, and what is missing

## The intended flow

1. **Transport** moves the plug from its starting position toward the requested port. It sees three camera images, robot measurements and a task code identifying the requested card and port. It predicts short movements in the gripper's local frame.
2. **Coarse perception** should recognize the requested port in the camera views and propose where to look more closely. It should also help decide when transport is close enough to pause. It must be allowed to say “I cannot identify it.”
3. **Precise perception** should use original-resolution crops around that proposal to estimate the opening's pose accurately enough for insertion.
4. **Insertion policy** should use the estimated port pose to align and insert. Recovery and RL would come later if the supervised system works reliably enough.

These pieces are **not yet connected as an automatic end-to-end system**. The current autonomous transport actor uses camera features that compress each whole image into one vector. It receives the task code, but it does **not** currently receive the frozen port locator's predicted position. The locator tests described below were run separately on saved actor images, after the actor had already moved. A successful locator test therefore does not mean the actor used that visual information.

## What the recent test showed

On three saved autonomous routes with one or two cards, two existing RGB locators each found the requested port early in all three episodes, without an accepted wrong-port prediction. They are promising as **coarse** locators for those particular scenes. Their accepted 3D position errors were about 3–4 mm at the 95th percentile, which is useful as route context but is not precise enough by itself for the roughly 0.5 mm insertion corridor.

The transport actor nevertheless reached the safe requested-port approach region in **zero of those three routes**. This means the ports could be found in those images by a separate model, while the actor's movements did not bring the plug close enough. One likely explanation is that pooling each whole camera image discards the spatial relationship between the plug, the row of cards and the requested port. The actor's response to changing only the requested card/port code was also small in earlier tests. Those observations support the explanation, but they do not prove it is the only cause of the misses.

The easy sparse-card results do not generalize to the harder identity cases. In the earlier ten-request replay, the selected-port locator acquired an early target in only **3/10** episodes. It acquired **none of the six requests for cards 1, 2 or 3**. The requested opening projected into all three camera frames on all seven early samples in each of those six episodes. Projection says where geometry would appear; it does **not** prove the opening was unobscured in the real RGB. However, an inspected card-2 example shows all five card faces in the center image while the model's estimated point was 26 mm from the requested opening. That is concrete evidence that identifying the correct physical slot across views is a problem, not merely that the board never enters the images.

## The gaps, in order

1. **Requested-port identity in hard scenes.** Similar-looking ports sit next to each other. Some camera views clip the end of the card row, making it difficult to tell which visible card is card 0, 1, 2 and so on. The current locators sometimes abstain correctly, but they miss too many early middle-card requests. A projection-based label also cannot tell us whether the port is truly visible behind the plug, cable or gripper.
2. **Transport action choice.** The current actor gets a task code but mostly compressed whole-image features. It has not learned a reliably different early route for each requested port, and it reached 0/3 safe approach regions in the latest autonomous comparison. The separate locator's output has not been fed into this actor.
3. **Safe stopping and handoff.** A learned transport stop was timely in only **2/10** held-out teacher episodes under its strict rule. A broad base-coordinate region can reject obviously distant stop proposals, but it cannot confirm the correct port or safe alignment. Automatic handoff remains disabled.
4. **Fine pose and insertion are untested as a connected autonomous chain.** Earlier precise perception and near-port insertion studies produced useful partial results, but this transport route has not reliably handed them a safe, correctly identified starting position. Millimeter-scale coarse estimates are insufficient for final insertion, so the original-image refiner still matters.
5. **Training-image mismatch.** Much of the older transport feature cache contains images already reduced to 256×288 pixels. The more successful visual locators use original camera images before resizing. A future actor trained on locator outputs needs matching original-resolution images or a clearly measured lower-resolution alternative.

## What to do next

First review early original images from the harder cases and mark whether the requested port is clearly identifiable, visually ambiguous because the board reference is missing, or genuinely obscured. Then test one bounded model that represents the **whole card row and the requested slot** while retaining spatial image information and a short movement-sampled visual history. It should be able to abstain. Check it on complete, new development episodes covering all ten card/port requests: **no accepted wrong port and at least nine of ten early correct acquisitions**. If safe views do not show enough information, change where transport looks rather than fitting another model to ambiguous images.

Only after that identity gate should we train the matched spatial transport actor and compare it with the current compressed-image actor on the same data, update budget and new starts. The actor must then reach a safe requested-port approach region reliably before we turn on automatic handoff, the precise insertion policy or RL. Reserved final scenes remain sealed.

See the [spatial prerequisite experiment](../experiments/2026-10-05-spatial-transport-prerequisite.md) and [current status](../STATUS.md) for exact metrics, commands and artifacts.
