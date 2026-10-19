# Exposed released SFP and SC repair — October 19, 2026

## Scope and decision

The user authorized reading the videos and iterating against the publicly released `eval_config.yaml`. It was published after the competition deadline. These results measure repair on a **known configuration**, not an unseen competition score. The separate reserved internal final scenes remain sealed. No model weights were trained or changed; the runtime remains rule transport, RGB perception, and an explicit visual insertion controller. BC, SERL and RL actions were not used.

The first complete repaired replay scored **272.701942/300; all three trials received the 75-point physical-insertion award**. Three matched repeats scored **275.435576, 273.143938 and 276.264227/300**, each with all three physical insertions. That is **12/12 successful rounds across four complete replays of this one exposed configuration**. Earlier frozen replays with the old SC route and no SFP repair scored about **98.6–98.75/300, 1/3 insertions**. Repeating one known configuration does not establish broad reliability.

## Causal failure audit

| Trial | Observed failure before repair | Evidence and repair |
| --- | --- | --- |
| SFP 2 | The sample-derived nominal opening was 254 mm away in base-frame y from the scored port. The initial three-camera estimate was roughly 12 mm from truth, but separate transport and fresh insertion checks rejected it because both required the pose within 200 mm of the sample nominal. The fallback route then made high cable/card contact. | Guarded 3-view visual transport waypoint; widen each of the two nominal search envelopes to 350 mm; lift the SFP fallback by 60 mm. The precise native pose fit and temporal agreement still gate insertion. Four matched sample starts inserted 4/4 with the guided waypoint. |
| SC 3 | The actual board translated about 150 mm and rotated about 85° from the sample board. The old route treated the sample-board row as physical truth, descended beside the wrong row, and contacted a real NIC card or pulled the cable across it. The failed video shows the gripper move down over the card around 50–58 s. | Look at the colored board marker in native wrist RGB at safe height. Fit board x, y and yaw from camera projection, require two consistent frames, then route outside the **observed** card row. The first released full replay inserted. |

The marker is a visually observed feature of the asset, with board-local dimensions calibrated from the asset. It is not simulator board TF at runtime. Square symmetry is disambiguated against a broad requested-port prior, with ambiguity and reprojection rejection. No true geometry selects crops or waypoints during execution. The SC route is: shift at starting height; look for marker; move outside observed row; descend outside; travel beside the row at low height; tilt beyond it; sweep toward requested port; acquire the port using the existing native RGB refiner; approach and insert with the existing explicit controller. An absent marker times out after six seconds rather than causing a blind descent.

## Controlled sequence

| Variant | Released three-round score | Physical insertions | Decision |
| --- | ---: | ---: | --- |
| Frozen old candidate, `sc_skip_back` plus early tilt | 98.62–98.75 | 1/3 | Baseline failure preserved. |
| SFP visual waypoint, fresh envelope and fallback lift; old SC route, cycle 5 | 191.661253 | 2/3 | SFP repair supported; SC still fails. |
| SC sideways shift at start height, cycle 6 | 191.721859 | 2/3 | SC still descends over real card. |
| SC camera scout but stale-board route, cycle 7 | 179.482346 | 2/3 | Marker visible, route still wrong; force harm. |
| SC marker-derived route at low travel height, cycle 8 | **272.701942** | **3/3** | First complete pass; repetition needed. |

The marker-guided route was also run on four independent, sample-derived three-card starts. All four reached correct-port visual handoff, and **2/4** completed insertion. The two misses occurred after handoff: load rose with near-zero measured axial progress; local insertion retry and measured-path backtrack did not resolve them. Thus transport acquisition improved, while SC insertion reliability remains open. The hard five-card port-0 development case failed before marker acquisition and is a separate coverage gap.

## Artifacts and exact commands

The cycle roots below contain frozen source copies and SHA-256 manifests, command environments, `runtime_trace.jsonl`, scored bags, camera frames, scoring YAML, and machine-readable score summaries:

- SFP repair: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle5_fresh_envelope/`
- Failed SC route variants: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle6_sc_level/` and `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle7_sc_scout/`
- First passing released replay: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/`
- Matched sample-derived SC starts: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_route_repair_dev_v2/sc_marker_routed_low_full_four/`
- Three-repeat reliability run: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle9_sc_marker_repeat3/`
- Review video of the first passing run, all five views at 10 fps: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/repeat_1/rollout/five_cameras_10fps_grid.mp4`. The diagnostic still images at 50, 57, 67 and 82 s are in its sibling `review_frames/` directory. Visual review confirms the arm leaves the card row sideways, travels beside it, then approaches the selected SC port; the orange cable remains visible during this motion. The scored bag, rather than appearance alone, establishes insertion.

Run a released-config local diagnostic with:

```bash
.pixi/envs/default/bin/python scripts/run_sc_sfp_repaired_released_eval.py \
  --output /home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker \
  --repeats 1 --sc-route sc_marker_scout --sfp-lift-m 0.06 --gpu 4 --domain 178
```

The exact run is preserved; choose a new output directory and ROS domain to reproduce. The runner checks that the downloaded official YAML equals the scene YAML, verifies source hashes before each repetition, and records the active route inside the container. It uses the rootless Docker host socket. On the first passing run, active transport-decision p95 including image work was **187.2 ms (SFP 1), 133.8 ms (SFP 2), and 266.7 ms (SC)**; the SC marker fits took 109 and 117 ms on their two accepted frames. Across the repeats, the maximum per-trial active-decision p95 was **299.4 ms**, just under the 300 ms target. Individual tail calls exceeded 300 ms, so there is little latency margin in some runs. Separate shared insertion-perception p95 ranged from 117.9 to 231.5 ms.

## Interpretation and next gate

The result demonstrates that wrong sample-board geometry was the dominant cause of this SC transport failure, and that the SFP failure involved rejecting an otherwise useful visual pose. It does **not** establish that every card count, port and reset now inserts. Test frozen code on new independently generated board poses and card layouts. In particular, target the four/five-card marker acquisition gap and SC insertion stalls after a valid handoff. Do not tune the reserved internal final split.
