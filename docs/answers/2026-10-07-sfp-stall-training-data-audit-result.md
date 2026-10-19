# SFP actor training-data audit and cleaned-data experiment

Your stall concern was valid, but it applies mainly to the **older SFP
source**, not the newly collected routes. I had not applied the earlier SC
stall-cleaning pass to the new SFP data. The SC pass removed explicit
waypoint holds; the SFP data were collected differently. I audited the raw
SFP labels instead of assuming that pass was appropriate.

The new scored SFP training routes had almost no sustained no-motion holds:
only **1 of 115** diverse training episodes had a continuous ≥1 s interval
with a >5 mm commanded target and <0.1 mm measured TCP movement per raw
step; the two new paired-request **training** batches had none. In contrast, **68 of
236 legacy SFP training episodes** had such an interval. One checked
example held near 90 mm in front of the selected port for several seconds,
then continued to a successful insertion. This is a real command-versus-
motion plateau, though its exact physical cause is not established.

I made a **separate immutable training mask**, without changing the original
episodes, images, action targets or timestamps. It excludes 689 cached
training rows from 70 episodes, nearly all legacy rows. It excludes no
development rows and no requested-route contrast pairs. Two otherwise
identical actors then trained for 2,000 updates each, one with the original
sampler and one with this mask. The GPUs were occupied, so both used CPU.

| Held-out measure | Original sampler | Stall-filtered sampler |
|---|---:|---:|
| Old SFP action error, p95 | 1.493 mm | 1.498 mm |
| New SFP action error, p95 | 1.569 mm | 1.583 mm |
| SC action error, p95 | 2.155 mm | 2.193 mm |
| Requested-route pair error, p95 | 3.810 mm | 4.172 mm |
| Same-image task-code action diameter | 1.568 mm | 1.554 mm |

**The cleanup did not improve the actor**, so I did not promote the new
checkpoint or spend a fresh live rollout on it. Most removed rows were
already near the port; the autonomous actor had diverged from the successful
route earlier. Stall labels alone therefore do not explain its current
transport failure.

Other data checks found no new-source target-frame conversion error in a
2,000-row reconstruction audit. The training code also checks finite
values, task encoding and scene-group split separation. The main remaining
data gaps are: the legacy set is highly skewed (**227/236** episodes request
card 0/port 1), newer held-out routes have limited fresh 3–5-card coverage,
and nominal successful trajectories rarely show how to correct an
**actor-visited wrong route**. A prior ten-episode, two-second actor-prefix
CheatCode correction fit already failed a 0/3 live arrival gate. I will not
repeat that same collection recipe or add routine nominal episodes without
a distinct route-repair hypothesis.

The [full audit and training record](../experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md)
contains commands, checkpoints and scores. The [raw-trace plot](/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/stall_raw_examples.png)
shows a legacy plateau beside a new route. The [machine summary](../experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.summary.json)
records the decision.
