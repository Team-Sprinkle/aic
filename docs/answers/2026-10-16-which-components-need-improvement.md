# Which components need improvement?

**Mostly yes—but we should not declare the refiner fully solved.**

The official-scene repeats have now finished:

| Official setting | Successful transport + verified handoff |
| --- | --- |
| SFP, one card | **5/5** |
| SFP, two cards | **1/5** |
| SC, three cards, new refiner | **0/5** |
| Same SC setting, previous refiner | **0/5** |

The broader card-count/port sweep is still running: **64/120 total runs completed** at this check. These numbers measure transport/handoff, not insertion.

- **Transport rules: definitely need improvement across board placements.** SC fails with both refiners; its route does not reach the correct acquisition corridor. Changing only the refiner did not repair it.
- **Coarse localization and acquisition logic: need diagnosis/improvement.** They must locate the requested port and guide the robot into a view where precise estimation works. A failure to obtain an accepted pose is not automatically a wrong-port classification.
- **Refiner: promising, but not cleared across official settings.** It has strong conditional accuracy on tested views. However, the failed SFP example reached a useful physical location and then failed the multi-camera pose-consistency check. That could involve coarse crop placement, refined landmark errors, occlusion, or geometric/calibration assumptions. We have not isolated which one caused it.

The priority is to fix SC transport/acquisition across board placements, and diagnose the SFP visual-fit failures before deciding whether the coarse model, refiner, or pose-fitting assumptions need repair. Keep rejection thresholds unchanged during the frozen comparison; accepting an inconsistent pose would hide the problem.
