# Which perception stages use camera history?

**Among coarse perception, the crop refiner and the final verifier, none currently has a promoted model that reasons over a sequence of camera images to establish requested-port identity.** The transport actor is the main model with a genuine long image-feature history.

| Stage | Earlier camera images used by the model? | Temporal behavior that does exist |
| --- | --- | --- |
| **Transport actor** | **Yes.** Its temporal head can use 17 sampled three-camera feature snapshots across up to about 60 seconds, plus prior proposed actions and measured motion. | This supports route imitation; it is not a verified requested-port identity tracker. |
| **Coarse port perception** | **No.** The current locator predicts from the current three-camera observation, each camera encoded separately. | An experimental SC visual stop keeps a median of up to five recent **predicted 3D targets** and requires several good decisions. That is smoothing and a streak rule, not a model reading earlier RGB to identify a card. |
| **Original-image crop refiner** | **No.** It predicts from the current original-image crop for each camera and the requested target code. | It is run again on later frames, but its network does not consume the earlier crops. |
| **Final camera check / verifier** | **Normally no image-sequence identity model.** It inspects a fresh three-camera triplet with measured robot state and task request. | It can require current three-view agreement and repeated good stop decisions. An SC request-specific anchor/history check exists only as an opt-in **shadow diagnostic**: it logs what it *would* accept without changing control. The SFP selected-anchor result was an **offline replay**, not an integrated runtime verifier. |

The distinction matters: **three simultaneous cameras are multiple views, not camera history**. Likewise, averaging five predicted positions or demanding three consecutive accepts does not reveal which numbered slot an earlier wide view showed. A stable wrong-port prediction can pass those simple temporal checks.

The missing capability is a request-specific memory that first identifies the target from a sufficiently informative view, records the evidence and measured TCP pose, tracks that identity through later motion, and abstains if it becomes ambiguous. It must reset at an episode/task change and must never initialize from a prediction whose identity was not well established. The existing SFP replay showed that such an anchor can preserve a correct estimate in 52 later frames but did not increase early target acquisition.

Sources: [transport history inputs](../experiments/2026-10-01-transport-five-way-comparison.md), [SC sparse-history shadow test](../experiments/2026-10-03-sc-sparse-history-handoff.md), [SFP anchor replay](../experiments/2026-10-04-sfp-context-sufficiency-audit.md), [current handoff design](../SELECTED_PORT_HANDOFF_PLAN.md).
