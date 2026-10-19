# Board landmark data-size follow-up

I ran the matched test we discussed. I trained the same original-image endpoint refiner with **20, 40, and 70 complete training episode/reset groups**, using the same ten development groups, architecture, seed, initialization, 2,000-update budget and checkpoint rule. The sets were nested and balanced over the ten requested card/port identities.

| Training groups | Labeled crops | Held-out pixel median | Held-out pixel p95 | Accurate early board references |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 7,129 | 11.85 px | 34.58 px | 1/10 |
| 40 | 9,742 | 11.87 px | 34.67 px | 0/10 |
| 70 | 16,145 | 9.92 px | 32.17 px | 3/10 |

The 20-to-70-group p95 improvement is only **7%**, with no improvement at 40 groups; the planned scale-up signal was a consistent **20%** reduction. All runs remain far from the **3 px median / 8 px p95** landmark target and **9/10** accurate early board-reference target. In the 20- and 40-group runs, training loss fell, but the best held-out p95 came after only the first update. This experiment does not prove that additional *different* data cannot help; it shows no strong return from adding more of the **current projected-label data**.

I also built and inspected a fixed [30-panel original-image packet](../../outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/visibility_review/). It shows the CAD projected endpoint without model predictions. In my single-reviewer assessment, **none** was clearly a unique, numbered physical landmark; 23 were ambiguous/repeated or obscured, and seven were uncertain. This is not an independent visibility certification, and the separate independent complete-public-observation review remains pending. It reinforces the concern that the current target is a geometrically defined pixel that the camera may not uniquely reveal.

**Decision:** do not collect more episodes with this same endpoint label yet. First identify and independently verify a visually recognizable fixed board origin, or find a safe view that exposes one. Then a new grouped learning curve would be meaningful. I did not enable handoff, insertion, RL, or use reserved final scenes. SFP transport's separate safe-arrival gap also remains open.

[Experiment record, exact manifests and commands](../experiments/2026-10-07-sfp-rail-data-scale-and-visibility.md) · [learning-curve plot](../../outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/learning_curve.png).
