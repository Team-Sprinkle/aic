# SC start-side cable route comparison (October 19)

## Question and scene provenance

The five-card video in `2026-10-19_transport_local_recovery_sc_settle_v4/development_sc_n5_p1_r0/` shows the cable crossing the card row before the arm has reached the port. In an overhead sketch, the start resembles `|.||||`, while the previous SC detour assumed `.|||||`. Contact with a card alone is acceptable; sustained cable loading that stops measured TCP progress is not.

This scene is **sample-derived development**, not a released production evaluation scene. `scripts/prepare_sample_conservative_transport.py --phase development` reads the pinned predeadline public `configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml` (SHA-256 `21f6eb9ac0d71f7dddac3657c41b5fa212e5aaa1fd68528b08ec219585c8d`) and uses RNG seed `2026101627` to create new card, SC rail, board, and cable configurations. Its `official_trial: trial_3` field identifies the sample task template, not a copy of a released final scene. The exact scene and manifest are under `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/`. No released `eval_config.yaml` scene was used for route changes here. The per-run `eval_config.yaml` files in the archive are generated configs for these development scenes.

## Routes and controlled setup

- **sc old:** move backward to the behind-row plane, move sideways while descending, traverse beside the row, sweep toward the selected port, tilt, then request a visual handoff.
- **sc skip back:** omit the first backward leg; move sideways and down directly from the measured start plane. This is a candidate for starts already amid the row.
- **sc ascend:** lift 100 mm at the current horizontal location, move backward at that height, then move sideways and down before traversing.
- **early tilt ablation:** perform the tilt while still outside the row, then sweep toward the port at the target orientation. This is an ordering change within `sc skip back`, not a fourth start-side route.

The implementation selects the three named routes with `AIC_SC_TRANSPORT_ROUTE`. `AIC_SC_EARLY_TILT=1` selects the ordering ablation; default behavior remains the old route and old tilt order. The first variants were compared on the same five-card scene, then the selected candidate was screened on seven SC scenes. All live comparisons use one rootless container at a time on GPU 4, the same `RunSampleTransportRecovery` policy, frozen SC perception weights, oracle contacts for **offline diagnosis only**, and no actor training or insertion. Runtime decisions use public images, measured robot state, force, task code, and fixed sample-derived waypoints; oracle contacts and true port geometry do not enter control. SFP route code is unchanged.

The original video's five cameras were sampled at one-second intervals under `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/frames_1hz/`. The old overhead first-30-second sheet is `old_overhead_every_second_first30.jpg`. Each candidate rollout retains 20 fps raw camera frames, runtime trace, scored stream, and oracle contacts. The skip-back successful pilot has `sc_skip_back_short_side_v2/overhead_every_second_first30.jpg`; the lift-first opposite-side pilot has `sc_ascend_short_side_case_v1/ascend_overhead_every2s.jpg`. The video and contact evidence show cable links crossing and touching the NIC row as the tool travels; the contact trace in the failing skip-back repeat includes cable links against several card mounts and one PCB during the tilt. This supports cable load as a cause of the stalled progress, although the exact distribution of tension along the cable is not directly measured.

## Results on the difficult five-card, port-1 development scene

| Variant | Outcome | Diagnostic |
| --- | --- | --- |
| `sc skip back`, original outside corridor `x=-0.27 m` | No visual handoff | Force/progress gate fired three times; recovery budget exhausted. |
| `sc ascend`, original corridor `x=-0.27 m` | No visual handoff | Lift and elevated back completed, then sideways descent loaded the cable and local retreat aborted. |
| `sc skip back`, opposite-side `x=+0.12 m` | One qualified correct-port handoff | At about 155 mm axial standoff; no local recovery in the pilot. |
| Same opposite-side route, seven-case repeat | Six of seven qualified | The repeat of this exact five-card case stalled at the tilt with cable/card contacts; a single good pilot was not reliable evidence. |
| `sc skip back`, opposite-side `x=+0.18 m` | No handoff | Stalled earlier while traversing beside the row. |
| `sc skip back`, `x=+0.12 m`, beyond-row `y=+0.20 m` | No handoff | Reached the tilt, then stalled again. |
| `sc ascend`, opposite-side `x=+0.12 m` | No handoff | Reached the tilt without a local recovery trigger, then timed out for lack of measured progress. |
| `sc skip back`, `x=+0.12 m`, **early tilt** | Qualified correct-port handoff in first pilot | The orientation change happens before the sweep toward the port. |

The seven-case screen includes sample SC no-card, development no-card ports 0/1, one-card ports 0/1, and five-card ports 0/1. `qualified` means correct requested-port visual acquisition with an offline opening error at most 1 mm, axial standoff 130–180 mm, lateral error at most 25 mm, orientation at most 15°, active inference p95 below 300 ms, and no scored harm. It is **transport/handoff**, not insertion. Brief force excursions below the toolkit's one-second penalty threshold are reported separately.

The early-tilt seven-case screen reached the correct visual handoff in **7/7**, with no scored harm and no local-recovery trigger. **6/7 met the full qualification gate**. The five-card port-0 case acquired the correct port with 0.086 mm estimated opening error, 155.8 mm axial standoff, 1.11 mm lateral error, 3.43° orientation error and no excessive-force report, but active perception p95 was **390.8 ms**, above the 300 ms limit. An exact same-case repeat met the full gate with active p95 **189.5 ms**, no excessive-force report and correct acquisition. Its first run's active timings were slow across the rollout (median 257.1 ms versus 106.9 ms in the repeat), which points to a runtime performance fluctuation; the trace alone does not establish its external cause. The five-card port-1 case passed twice with the early tilt, including a new seven-case repeat at 154.94 mm axial and 3.13 mm lateral standoff. This is stronger evidence for this particular start than the earlier one-pass skip-back result, but it does not certify all SC starts.

One-second overhead review of the early-tilt pilot is `sc_skip_back_short_side_early_tilt_v1/overhead_every_second_first35.jpg`; its synchronized five-camera 20 fps recording is `sc_skip_back_short_side_early_tilt_v1/development_sc_n5_p1_r0/five_cameras_20fps_grid_h264.mp4`. The cable still touches the card region. The observed improvement is continued measured tool motion and successful handoff, not zero cable contact or a full insertion.

## Exact commands and artifacts

The root of all diagnostics is `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/`. Each run folder contains source hashes, a source snapshot, generated scene YAML, command JSON, metric JSON, contact trace, and raw camera frames. Example command:

```bash
.pixi/envs/default/bin/python scripts/run_sample_rule_development.py \
  --root /home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3 \
  --output /home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/sc_skip_back_short_side_early_tilt_v1 \
  --case development_sc_n5_p1_r0 --family sc \
  --policy RunSampleTransportRecovery --sc-route sc_skip_back --sc-early-tilt \
  --spec /home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/transport_spec_SC_opposite_side_v4.json \
  --gpu 4
```

The source and target artifact hashes are captured in each run's `source_hashes.json`. No frozen final scene was opened. Promotion requires repeated SC cases with measured progress, acceptable force, correct visual handoff, and no evidence of cable snag; a single handoff does not satisfy this gate.

**Decision:** retain `sc skip back` plus early tilt as an opt-in SC transport candidate. Keep `sc old` as the default until a broader independent sample-derived SC scene screen passes the physical and p95 gates. The tested route resolves the specific blocked five-card start in two runs; the seven-case screen has an unresolved one-run latency miss. SFP has not been changed or retested in this experiment.
