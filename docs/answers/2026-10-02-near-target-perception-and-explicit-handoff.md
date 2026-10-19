# Near-target perception and an explicit transport handoff (October 2, 2026)

## Recommendation

I agree with the proposed **order of operation**: let transport move toward the requested destination; when it believes it has reached a safe pre-insertion neighborhood, pause; then run precise perception for the **requested nearby port** and let that result either approve the insertion handoff or reject it. This should simplify the precise pose problem and avoid spending most of the episode estimating distant or hidden openings. It does **not** by itself make transport reliable, and “nearest visible opening” cannot replace the task's selected port identity.

```text
Initial state, target may be offscreen
       |
       v
TRANSPORT actor + its coarse arrival/stop head
       |
       | proposes "I am near the selected destination"
       v
HOLD measured TCP pose; run precise selected-port perception
       |
       +-- selected port not found / ambiguous / unsafe pose --> resume transport
       |                                               or hold safely
       |
       +-- selected port verified near safe standoff --> INSERTION actor
```

## 1. Is this better than predicting every port pose?

**Likely a better deployment design for precise insertion.** A single current image cannot reveal the exact 3D pose of a port that is offscreen or fully hidden. A long image history can sometimes remember a previously seen, stationary port, but it cannot create visual information about a port never observed, and memory errors grow when robot motion or occlusion is uncertain. At submillimeter insertion precision, we should not ask a model to hallucinate those poses. If the requested opening is hidden even after arriving nearby, the controller should change viewpoint, wait, or reject the handoff rather than confidently invent its pose.

There is one nuance about the **current** implementation. Its shared coarse locator has 12 output slots for the two SC ports and ten possible SFP card/port identities. It produces a candidate pixel location and an “projected in-frame” flag for each, then the task code selects one. It does **not** triangulate a valid 3D pose for every absent or offscreen slot; unsupported candidates are masked. Nevertheless, those broad output heads and their projection-based labels do not solve true occlusion or reliable target identity. The SFP path already gives most of its precise computation to the selected port through original-resolution crops. Your proposal makes that separation stricter: do not run the precise estimator until transport proposes arrival, and train/evaluate it primarily on the nearby selected target plus hard negatives such as the neighboring port.

Even locally, we should not throw away **identity**. SC port 0 and port 1 are close together, and an SFP scene can show several similar openings. The local estimator should receive the requested family/card/port code and explicitly distinguish the chosen opening from visible alternatives. It can detect nearby alternatives to check identity; it does not need to report millimeter-accurate 3D poses for every port in the scene.

The likely gain is a simpler, better conditioned precise estimator and less wasted computation. It is a hypothesis to test with grouped scenes and autonomous rollouts, not a demonstrated performance improvement over the current locator. Transport's ability to arrive near the **correct** port remains the first dependency.

## 2. Exactly how would transport decide to stop?

There **is** an experimental output from the temporal transport model: a **handoff probability** estimated from camera features, measured robot state, the task code, and causal motion/command history. In the frozen SC comparison, the tested transport-only rule proposed a stop when this probability was at least **0.95 on two consecutive decisions after 15 simulated seconds**. On seven saved development paths, that gave only **two valid stops, two false stops, and three missed arrivals**. The numerical threshold is a diagnostic choice, not a reliable deployment rule. The current general transport runtime does not automatically start the insertion actor when a stop rule fires; it returns from its transport run.

We also tested a separate SC **visual standoff** rule. That one already runs a port locator late in transport, then steers to a predicted waiting pose and stops after repeated closeness checks. It gave **five valid stops, one false stop, and no missed arrivals** on the same seven saved paths. Its false stop had good position but wrong orientation and a poor route. It therefore remains development-only. SFP has no validated observation-only transport stop. The reported **10 mm / 3°** arrival criterion is scored using post-run truth; it is not a quantity that deployment can directly read.

Under your proposed design, I would make stopping a **two-stage decision**:

1. **Transport proposes arrival.** A lightweight head that already accompanies transport uses its own images, history, ordinary robot state, and selected-target task code to predict whether the measured TCP is in a safe *pre-insertion region* for that target. It should also predict/assess orientation and route safety, not just position. Require agreement over several observations. This head runs during transport; the full precise port-pose estimator does **not** have to run then. An elapsed-time or fixed-distance guess alone is inadequate across different starts and layouts.
2. **Pause and verify.** On a proposed arrival, command a measured-pose hold and run the precise **selected-port** estimator on fresh high-resolution camera images. Check that the requested identity is supported, the opening is actually identifiable in enough views, the estimate is stable, the TCP is in the safe waiting region with acceptable orientation, and there is no unacceptable force/contact or route condition. If it fails, do **not** start insertion: continue transport, actively acquire a better view, or safely stop, according to a tested policy. If it passes, hand control to the already-loaded insertion actor without a command jump.

This resolves the apparent chicken-and-egg problem: the transport model supplies a **coarse invitation to inspect**, while precise perception supplies the **accept/reject decision**. Precise perception runs only at plausible arrivals, and perhaps for short repeated observations during verification and insertion. During insertion it should keep checking or safely carry a recent locked pose through brief occlusion; a lock must not be mistaken for certified correctness.

## What must be built and proved

The necessary experiment is to train or repair the transport arrival head with episode-grouped, task-conditioned labels for safe pre-insertion arrival, including wrong-port, wrong-orientation, and wrong-route negatives. Test it on the existing frozen SC paths, the known SFP wrong-target case, and new development starts. The proposed stop must reject the known false stops without losing valid arrivals. Then test a **selected-port-only local estimator** against the current all-candidate coarse locator on the same near-target scenes, especially when the sibling port is visible and the selected opening is hidden. Report selected-target identification, pose error, false acceptance, full pipeline latency, and ultimately autonomous full insertion. Keep final scenes sealed.

The key operational rule is: **transport may suggest “we have arrived,” but only verified selected-port perception can authorize insertion.** Today neither half has passed that full gate, so this is the recommended revised architecture rather than a claim of deployed performance.

See [current stop comparison](../experiments/2026-10-02-transport-stop-comparison.md), [perception experiment](../experiments/2026-10-02-two-family-perception-local-bc.md), and [current transport runtime](../../aic_example_policies/aic_example_policies/ros/RunTemporalTransport.py).
