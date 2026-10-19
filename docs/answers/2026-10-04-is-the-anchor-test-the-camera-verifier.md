# Is the anchor test the camera check / verifier?

**Yes, it tests the requested-port identity part of the camera check.** After transport proposes a pause, the camera check should inspect fresh images and decide whether it has found the **requested** opening. The selected-only model supplies a candidate position; the tested history rule keeps an earlier accepted candidate as an anchor and lets later estimates confirm it.

It is **not the entire verifier**. A safe handoff must also check that the opening's pose is accurate enough, the tool's standoff and orientation are suitable, the three camera views agree, visibility/occlusion is understood, force/contact is acceptable, and the complete decision is fast enough. The new anchored rule was only an **offline replay on teacher camera observations**. It has not been connected to an automatic transport-to-insertion switch.

In short:

```text
transport proposes a pause
        ↓
camera check: requested-port identity  ←  this is the part we tested
              + pose, view, posture and force checks
        ↓
accept → insertion actor; reject → hold, move or seek another view
```

The identity subtest had zero wrong accepts on ten reused development episodes but found the target early in only 3/10. That is why the full camera check is still **not approved for autonomous handoff**. [Detailed result](../experiments/2026-10-04-sfp-context-sufficiency-audit.md).
