# Does transport stop just because a coarse model runs periodically?

**No.** There are three separate ideas:

1. **Inference cadence:** the transport policy receives new observations and makes repeated action decisions. This only says *when it gets a chance to decide*.
2. **Stop proposal:** a learned arrival signal or visual standoff check judges whether the robot appears to have reached a useful waiting posture. It generally needs the condition to hold over consecutive decisions. Some development modes also ignore the signal until an initial elapsed-time guard; that guard is not a stop timer.
3. **Final handoff:** after holding still and taking fresh images, the verifier must identify the requested opening and check pose, tool posture, route, force, and view consistency. Only then may the insertion actor take control.

The existing implementation has several **experimental** stop modes, not one proven general stop controller. The learned transport head can use a probability threshold and a streak when explicitly enabled. The SC visual stop uses an RGB-estimated selected-port standoff and orientation, also requiring repeated good decisions. The SFP stop signal is not approved for autonomous handoff; its diagnostic mode is disabled by default. The intended extra coarse requested-target check is a proposed early reject / pause gate, not a deployed safety gate.

An accurate requested-port pose **does help decide when to pause**. With the measured tool pose and the desired pre-insertion offset, we can calculate: “How far is the tool from a safe waiting pose?” and “Is it facing the opening correctly?” A coarse answer can trigger a held, high-resolution inspection; the final insertion decision needs a more precise answer. But position accuracy alone does **not** establish the correct port, unobstructed view, safe approach route, acceptable force, or suitable orientation. In a matched SC development replay, a visual stop was only **1.02 mm** from the desired position but was **6.71° misoriented** after an unsafe route. It was a false handoff despite its good position number.

```text
regular transport decisions
   ↓
observed arrival / coarse requested-target check
   ├─ poor evidence → keep transporting or seek a view
   └─ plausible waiting posture → pause and hold
                                ↓
                    fresh original-resolution images
                                ↓
                    final identity + pose + posture + force verifier
                       ├─ invalid → no insertion
                       └─ valid → insertion actor
```

The crucial current gap is that SFP coarse perception sometimes chooses the wrong numbered port or abstains, while transport itself can drift away. So the system **does not yet know reliably when to make that SFP pause or handoff**. We should evaluate stop timing and identity separately on complete autonomous routes, then together; a scheduled inference tick or a near-looking pose alone is insufficient.

The [handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md) now separates the intended coarse pause gate from the final verifier. The [matched SC stop comparison](../experiments/2026-10-02-transport-stop-comparison.md) records the supporting successes and false stop. A [paired-port crop experiment](../experiments/2026-10-04-sfp-requested-pair-crop.md) has only been drafted; no new pair-model training has started while we clarify this design.
