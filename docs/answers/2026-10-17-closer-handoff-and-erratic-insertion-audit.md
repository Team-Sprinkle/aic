# Closer handoff and insertion audit

I agree with bringing the plug closer before the learned insertion stage. The earlier150mm handoff was intentional under the previous plan, but the rollouts show that asking this BC actor to own that whole approach is not working.

## Closer handoff

I implemented a separate vision/proprioception controller targeting **25mm before the opening**. It aligns before advancing, slows near the port and retains force guards. It uses the estimated port and gripper-to-tip offset plus measured TCP motion, not true geometry.

The first SC test reached **24.2mm before the opening,0.55mm lateral error**, independently confirmed afterward from the raw bag. Plug orientation error was3.68degrees. This demonstrates one closer approach, not reliability across settings. Four matched fresh/warm SC/SFP pilots are running. Recovery is off only in this isolated diagnostic; the earlier hierarchical results remain preserved.

The unchanged fresh actor still failed physical insertion afterward: **5.95mm lateral error and0.51mm axial depth**. Official score was75, but this is not a full physical insertion under our gate. Shortening the approach alone does not solve the actor.

## What caused the erratic motion?

The strongest measured finding is **disagreement between successive action chunks**:

| Task | Within-chunk target changes, p95 | Changes when replanning, p95 |
| --- | --- | --- |
| SC |4.5mm /1.9degrees|**32.2mm /17.2degrees**|
| SFP |1.3mm /0.13degrees|**12.3mm /2.0degrees**|

These are commanded target changes, not actual instantaneous robot motion. The actor is a deterministic transformer, not diffusion or random exploration. It is producing inconsistent successive plans.

### Chunk execution

It predicts **four50ms steps and executes all four**. There were **zero insertion-stage stale-step drops** across37actor-executing episodes. Therefore this was not caused by intentionally executing just one predicted step.

However,200ms is a short prediction horizon, and inference adds pauses: commands are50ms apart inside a chunk, but the median gap between chunks is100ms(p95150ms). Longer-horizon models are worth a controlled test, not an assumed fix.

### Data and training

Training was substantial: **421training episodes—102SC and319SFP—plus79validation episodes**. Every requested task code is represented. There are far-approach examples in95SC and309SFPtraining episodes, so the150mm range was not entirely absent.

There are meaningful gaps:

- Far approach receives only about12–14%of each family's effective training weight.
- About55%of SC0–30mm chunks are almost stationary. Some are valid contact settling; we must not delete them blindly. SFP approach labels mostly move, so pauses alone do not explain SFP failure.
- Training supervises individual chunks and internal motion, without directly requiring successive plans to agree.
- Runtime resets history at actor takeover; its initially empty history may differ from preceding teacher-motion context.
- The actor receives port orientation and TCP orientation, but **not an approved gripper-to-plug rotation estimate**. That is an additional fine-alignment gap under randomized grasps, not a complete explanation of the large jerks.

A numerical audit of42,080predicted target conversions found essentially zero error in the logged port-anchor and plug-point-to-TCP conversion. A gross bug in those conversions is not supported. Other timing/label issues remain worth checking.

## Next experiments

1. Finish the four25mm pilots and separate approach success from actor insertion success.
2. Check actual-history initialization and successful-teacher command/load compensation at matched states.
3. Prepare copied recorded-target data with phase balancing and reviewed pauses; compare a4/4baseline with **8predict/4execute** and **12predict/6execute**. Longer horizons require new labels/model outputs, not stretching the existing four predictions.
4. Test cross-plan continuity and fix measured-path recovery progress separately before reconnecting hierarchical backtracking.

No new training has started yet. The current experiments isolate the causes instead of changing data, architecture and execution simultaneously. Original evaluations remain intact; no official-scene tuning or reserved-final access.

[Detailed audit and commands](../experiments/2026-10-17-near-handoff-and-insertion-continuity-audit.md).
