# Did the temporal transport models plateau? Is longer history better?

**They were stopped at the planned 3,000 training updates, before we saw a clear plateau.** This was an update limit, not a wall-clock timeout. Both shared action-only models chose their final checkpoint as the best one on the episode-grouped validation sample.

| Model | SC first-action p95 error at update 2,000 → 3,000 | SFP first-action p95 error at update 2,000 → 3,000 |
|---|---:|---:|
| Current frame only | 2.39 → 2.32 mm | 3.21 → 2.31 mm |
| Up to 60 s of causal history | 2.91 → 2.27 mm | 3.56 → 2.62 mm |

These held-out action errors were still improving, especially for the history model. Thus we cannot say the models were fully trained, nor that simply giving them more updates would solve the live failures. The other branches had different late behavior: the route model selected update 2,750 and the diffusion model update 2,250, with considerably worse action or route errors.

**Longer history helped markedly on the one SFP development rollout we tested.** The history model stopped about **8 mm** from the teacher's pre-insertion TCP pose; its matched current-frame model stopped **315 mm** away. The old frozen model was about **130 mm** away. The history model followed the intended direction and reached roughly the right axial plane. It still had about **8 mm lateral** and **6.9° orientation** error, so this was **not an insertion**.

**It did not help on the one SC five-card development rollout.** The history model ended about **158 mm** from the teacher standoff, versus **85 mm** for the current-frame model. Both initially followed part of the around-card route; the history model then deviated badly near the return/tilt phase. Offline, the history model's SC first-action p95 was only slightly better (**2.27 vs 2.32 mm**) and its SFP p95 was slightly worse (**2.62 vs 2.31 mm**). Those offline numbers do not predict the outcome of a full rollout, where small errors compound.

So the precise answer is: **history is a promising SFP transport input, but it is not yet a proven improvement for the shared SFP/SC policy.** The live comparison covers one development scene per task, and neither policy autonomously inserted. The final scenes remain sealed.

If we test whether undertraining explains part of the gap, a bounded experiment would continue **only the shared history head** from update 3,000 to at most 6,000 on the same data and split, check grouped validation every 250 updates, and stop after five checks without a meaningful gain. We would then rerun matched SFP and SC development starts, including a new start for each. The deciding result would be improved SC route completion and SFP standoff without a regression; lower first-action error alone would not be enough. This continuation is a proposal, not an experiment already run.

Detailed measurements and artifact locations: [five-way transport comparison](../experiments/2026-10-01-transport-five-way-comparison.md).
