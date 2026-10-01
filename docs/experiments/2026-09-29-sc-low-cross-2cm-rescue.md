# Fixed-scene SC low-cross ±2 cm route diagnostic

Date: 2026-09-29. Status: bounded privileged Gazebo teacher experiment;
no autonomous actor, BC/RL update, final-scene opening, or MoveIt-executed
trajectory. The experiment revisits the **one** `forward_minus10` partial
insertion in the [September 26 low-cross seven-run sweep](2026-09-26-sc-far-return-route-sensitivity.md).

## Fixed case and test

The archived failure used seed 51500, five NIC cards at their allowed
leftmost rail positions, SC port 0, `outside_left_low_cross_return`, and
`AIC_CABLE_ROUTE_FORWARD_EXTRA_M=-0.01`. It scored Tier 3 = 39.67, about
10 mm partial insertion. Its exact `eval_config.yaml`, `world_audit.sdf`,
`bridge_audit.yaml`, and policy code were retained at
`/var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926/`.
The [machine summary](2026-09-29-sc-low-cross-2cm-rescue.json) records the
SHA-256 of all four files; they stayed byte-identical in every new run.
The selected port pose did not move, and the new starts differed by at most
0.06 mm per TCP coordinate. Every run used the same stock CheatCode final
alignment/insertion handoff. The route planner here remained the scripted
Cartesian diagnostic; the ±2 cm tests did **not** use MoveIt.

The [runner](../../scripts/run_sc_low_cross_2cm_rescue.sh) varied the three
route margins. `forward_plus20` means +20 mm **relative to the failed
forward setting**, i.e. its absolute `forward_extra_m` is +0.01. The other
variants kept the failed −0.01 m forward setting. Runs were sequential on
rootless Docker GPU 1; active jobs were left alone. The first exact repeat
fully inserted, so this is a stochastic outcome comparison rather than a
deterministic rescue test. We interleaved later exact repeats and the most
promising variant instead of interpreting one success as a fix.

| Route on the same scene | Full insertions | Median sampled cable-center gap in new runs | Interpretation |
| --- | ---: | ---: | --- |
| Failed setting repeated | **4/7**, including archived failure | 16.15 mm | Same route sometimes works, sometimes partially inserts. |
| 20 mm farther behind | 2/2 | 18.15 mm | Feasible pilot; too few runs for reliability. |
| 20 mm farther outside left | **4/5** | **19.24 mm** | Best tested candidate for sampled clearance, but one partial insertion. |
| Both 20 mm farther behind and outside | 2/2 | 17.02 mm | Feasible pilot; too few runs for reliability. |
| 20 mm farther forward than failed route | 1/2 | 14.44 mm | Moving forward alone is inconsistent. |

These are fixed-scene teacher rollouts. Even 4/5 versus 4/7 is insufficient
to claim a general insertion improvement. The sampled gap is between cable
link **centers** and simplified main PCB boxes; it omits cable radius,
protrusions, and named contact. The 20 mm outside-lane partial run had the
*largest* sampled gap in its group, 22.32 mm, and about 21.76 N peak wrist
force with no scored excessive-force/contact penalty. At the port-opening
frame, its plug proxy crossed with roughly 0.57 mm lateral offset, emitted
no insertion event, and stopped about 3.2 mm beyond the entrance. A matched
full run crossed at about 0.04 mm lateral offset, emitted an event, and
reached about 15.6 mm. Orientation errors were small in both. This supports
a near-opening alignment/stall hypothesis for this partial, not a diagnosed
cable snag. Another unchanged-route partial reached about 10.5 mm with
0.37 mm final lateral offset; one mechanism does not explain every partial.

I inspected both outside-lane side and overhead contact sheets. The orange
cable remains close to the card tops at sampled times, and the arm occludes
some of it. The visual record does not certify the user's stricter
"never over the cards" route condition, even for successful insertions.

## Reproduction and artifacts

Run one case with, for example,

```bash
bash scripts/run_sc_low_cross_2cm_rescue.sh lane_plus20 3
```

The script uses `DOCKER_HOST=unix:///run/user/1008/docker.sock` and the
official pinned evaluation image. To regenerate the aggregate JSON:

```bash
.pixi/envs/default/bin/python scripts/summarize_sc_low_cross_2cm_rescue.py \
  --root /var/tmp/chmin_aic_20260918_act/sc_low_cross_2cm_rescue_20260929 \
  --historical /var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926 \
  --output docs/experiments/2026-09-29-sc-low-cross-2cm-rescue.json
```

Each new case has a score, exact scene, policy copy, route plan, scored bag,
sampled cable geometry, port-relative trace, and command log. The compact
[summary](2026-09-29-sc-low-cross-2cm-rescue.json) lists all 17 new runs,
including failures. All raw folders were copied to
`/home/nas/chmin/aic_archive_20260929/sc_low_cross_2cm_rescue_raw/`;
the [archive manifest](2026-09-29-sc-low-cross-2cm-archive.json) verifies
SHA-256 for all 17 scored MCAPs, scores, route plans, and summaries.
The selected H.264 20 fps review videos are:

- [Unchanged-route partial](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/baseline_partial/all_views_20fps.mp4)
- [Outside-lane +20 mm partial](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/lane20_partial/all_views_20fps.mp4)
- [Outside-lane +20 mm full](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/lane20_full/all_views_20fps.mp4)

The [video manifest](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/review_manifest.json)
maps videos, side/overhead contact sheets, scores, geometry summaries, and
port traces with hashes. Wide cameras are diagnostic only.

## Decision

Stop blind route-margin expansion on this scene. The outside-lane change is
a useful **candidate** for more cable clearance but did not make insertion
reliable or prove a clean around-card cable path. Diagnose and repair the
near-opening handoff separately, then compare planner-generated free-space
routes with the same final insertion controller on new SC ports and starts.
The host pixi environment has MoveItPy, but the official evaluation image
does not. An offline MoveItPy smoke test also failed to initialize its
current-state monitor without a usable live joint-state source; no MoveIt
path was generated or executed. A future MoveIt pilot must add the actual
card/fixture collision objects to its planning scene, because the current
backend calls `plan()` without doing so. Keep step 5 and the supervised
route/actor gates open; do not admit these diagnostics to BC or start RL.
