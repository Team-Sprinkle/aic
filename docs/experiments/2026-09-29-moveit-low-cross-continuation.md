# MoveIt SC low-cross continuation

Date: 2026-09-29. These are **privileged Gazebo teacher diagnostics**, not autonomous actor rollouts. The selected SC port pose was available to the insertion teacher. No BC or RL training ran, and the reserved final scenes stayed sealed. The [machine summary](2026-09-29-moveit-low-cross-results.json) identifies every scored attempt, trace, plan hash, height measurement, and review video.

## Hypothesis and measured repair

The prior strict-return route kept the plug too high while moving beside the NIC row. Its final plug tip was only about 4.4 mm below the card tops, and at around 25 s it was 0.6 mm above them. Per the user's clarified criterion, **touching a card is tolerable if the cable does not snag or stop progress**. The repair sends the tool away from the cards, left, and downward together; MoveIt then keeps a lower outside corridor until the row has been passed. The final insertion standoff remains at its original height. The planner still generates collision-checked joint paths against actual rigid Gazebo shapes. The cable is not part of MoveIt's planning scene.

The highest corner of any NIC collision shape is **1.331 m in world Z** in each tested layout. The [height script](../../scripts/measure_sc_route_heights.py) uses the exact collision-shape poses and scored `/scoring/tf` plug-tip pose; each result has about 12,800–13,100 TF samples. This is a plug-tip measurement, not a claim that every point on the cable is below the cards.

| Layout, route-only run | First tip below top | Tip relative to top at 20/25/30 s | Reached planned standoff | Review |
| --- | ---: | ---: | --- | --- |
| Five cards, SC port 0 | 11.10 s | −26.1 / −34.5 / −32.1 mm | Yes; 5.01 mm TCP endpoint error, 496 joint commands, no Cartesian fallback | [Five views](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/low18_diagonal_video/all_views_20fps.mp4) |
| Five cards, SC port 1 | 8.24 s | −24.4 / −34.3 / −33.1 mm | Yes; 5.71 mm endpoint error, 506 joint commands, no fallback | [Five views](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/port1_low18_video/all_views_20fps.mp4) |
| Three cards, shifted middle card, SC port 0 | 11.72 s | −25.3 / −35.3 / −30.8 mm | Yes; 5.03 mm endpoint error, 498 joint commands, no fallback | [Five views](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/threecard_mixed_low18_video/all_views_20fps.mp4) |

I inspected the overhead and side contact sheets for these runs and the [five-view full-insertion recording](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/low18_diagonal_insert_success_video/all_views_20fps.mp4). The orange cable approaches and can touch the cards. The arm and plug continue along the route in all three scored route-only scenes; no sustained snag or loss of progress is visible. All three route-only official score files have no scored off-limit contact or excessive-force event. We lack a named cable/card contact stream, so this does **not** prove zero contact or absence of brief cable tension. It does support the revised route gate on these tested starts. The port-0 full insertion independently shows that this route can reach a usable insertion handoff.

## Separate insertion handoff diagnosis

Route completion is distinct from full insertion. The route-only runs deliberately stop before insertion. For port 0, the initial online CheatCode handoff stopped at its force/tracking gate. A bounded retry allowing a 6× force margin at the preinsert tracking gate and requiring a sustained/median **2 N** contact signal yielded one **full insertion**: official Tier 3 = 75, total = 87.07, with an insertion event and no scored off-limit contact or force penalty. The [success video](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/low18_diagonal_insert_success_video/all_views_20fps.mp4) is synchronized 20 fps H.264 from three wrist, overhead, and side cameras. An exact repeat with the same plan and gates stopped on a confirmed 2.30 N near-opening contact at 6.5 mm axial offset. Thus the current teacher is **1/2 full insertions** under this limited matched condition; the route alone is not reliable full-insertion supervision.

For port 1, a route with the original wrist orientation reached standoff, but the insertion handoff had about 13 mm lateral tracking error and stopped roughly 122 mm axially from the opening. A targeted repair rotated the wrist toward the insertion orientation **after** passing the card row. Doing it before the traverse failed MoveIt collision/IK planning; doing it after the row produced an actual 236-point joint trajectory and a scored route to standoff with about 5.23 mm endpoint error. Its nominal insertion attempt realigned an initial 2.62 mm lateral error to 2.18 mm, advanced to 3.9 mm axial offset, and then stopped on a confirmed 4.11 N contact. This is a near-opening insertion/contact failure, not a transport snag.

Two port-1 recovery diagnostics exposed a different gate problem. A 5 mm initial lateral limit missed by 0.048 mm, triggering a retreat. At 6 mm, the recovery-mode teacher got through alignment, but its trace shows a **planned recovery intervention** immediately after insertion began, at only 0.35 N force change. The measured retreat was 7.64 mm against its 8 mm release gate and failed. This deliberate intervention is not evidence of an observed cable snag. Both failed results are retained in the machine summary. Do not admit either as a successful teacher insertion.

At the nominal port-1 contact, the four last trace samples show plug lateral error falling from **2.60 to 0.87 mm** while force rises from 0.53 to 2.46 N; the confirmed stop follows at 4.11 N. The target corridor is 0.5 mm. I tested the existing guarded-insertion lateral servo with the same route and gates, but that run stopped at its **preinsert** gate (2.86 mm lateral), so it never tested insertion servo. The 12 mm preinsert command-bias limit had saturated. A separate 16 mm bias-limit diagnostic still stopped at the preinsert gate (5.86 mm); its final bias was only about 11.5 mm, so the higher cap did not isolate a cap effect. These are negative/inconclusive teacher-controller tests, not evidence against the MoveIt transport route. No further blind bias sweep is justified.

## Reproduction and admission decision

The rootless replay accepts a previously saved MoveIt plan and exact one-trial scene. For example, with `DOCKER_HOST=unix:///run/user/1008/docker.sock`:

```bash
AIC_MOVEIT_PILOT_GPU=4 \
  AIC_MOVEIT_PILOT_SCENE_DIR=/var/tmp/chmin_aic_20260918_act/moveit_sc_port1_generalization_source_20260929 \
  AIC_MOVEIT_PILOT_ONLINE_INSERTION=true \
  AIC_MOVEIT_PILOT_TRACKING_GATE_FORCE_FRACTION=6.0 \
  AIC_MOVEIT_PILOT_FORCE_CONFIRM_THRESHOLD_N=2.0 \
  bash scripts/run_moveit_sc_route_pilot.sh \
  outputs/experiments/2026-09-29_moveit_gazebo_pilot/plan_port1_low18_prealign_after_row_insert \
  /var/tmp/chmin_aic_20260918_act/NEW_UNIQUE_AUDIT_DIR
```

For the 6 mm recovery diagnostic, add `AIC_MOVEIT_PILOT_EXPERT_MODE=recovery AIC_MOVEIT_PILOT_RECOVERY_INITIAL_LATERAL_GATE_M=0.006`. Each raw audit directory has the exact config, plan, `inputs_sha256.txt`, scored bag, score, policy log, and causal runtime trace. The [planner](../../scripts/plan_moveit_sc_fixed_route.py) accepts `--diagonal-egress --low-cross-z-m 0.18`; for the port-1 prealignment it also accepts `--aligned-orientation-xyzw` and `--prealign-from-goal-index 4`. The [handoff script](../../scripts/make_moveit_insertion_handoff.py) adds a no-motion phase marker so the privileged online insertion teacher starts only after the MoveIt joint trajectory.

The [NAS archive manifest](2026-09-29-moveit-low-cross-archive.json) verifies exact configs, plans, traces, scores, and scored MCAPs for all **13** completed low-cross runs by SHA-256 under `/home/nas/chmin/aic_archive_20260929/moveit_low_cross_raw/`. Captured source frames and volatile logs were excluded from the compact archive; the review videos remain in `outputs/`.

**Gate decision:** Step 3's revised snag/progress gate passed on these three tested route-only configurations. Step 4 has encouraging but **limited** generality evidence: both SC ports and two card counts/positions, with one scored route per configuration. It has not passed the planned several-start, matched scripted-control, repeatability gate. Step 5 has one verified complete port-0 teacher insertion; this is insufficient for robust SC local BC or a shared autonomous policy. Keep route-only and full-insertion labels separate. The perception visibility source also lacks true occlusion labels, wrong-port negatives, and SFP coverage. Therefore two-actor training, autonomous handoff, RL, and final evaluation remain gated.

The next targeted test should address the **near-opening port-1 contact**, not widen the transport corridor again. The existing causal trace already measures port-relative plug pose, commanded versus measured TCP, and force at about 20 Hz. Diagnose why the preinsert controller leaves 2–6 mm lateral error despite a large command bias, then test **one explicit alignment/standoff controller correction** against the frozen prealigned route on matched development starts. Stop that repair if it does not reduce lateral error toward the 0.5 mm corridor and improve full-insertion frequency without excess force. Only after repeatable full SC teacher insertions should its segments enter local BC. Independently, obtain real selected-port visibility labels and wrong-port negatives before training the two-actor handoff.
