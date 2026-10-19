# Can measured global coordinates replace the coarse visual handoff?

**Decision, October 5, 2026:** Robot TCP coordinates in `base_link` are available through the official observation's controller state. The selected port's exact global pose is **not** a permissible evaluation input: the CheatCode obtains it from simulator ground-truth TF, and the challenge rules explicitly restrict backend state leaking and configuration hardcoding during evaluation. A training-set nominal port location is permitted as a learned prior, but this replay does **not** support using it alone to stop transport or replace visual port identity. It can be tested later as a weak scheduling cue, never as an insertion pose.

## Official interface and randomization

The [official observation message](../../aic_interfaces/aic_model_interfaces/msg/Observation.msg) includes `controller_state`; the [controller](../../aic_controller/src/aic_controller.cpp) publishes measured `tcp_pose` in the robot base frame. The task request supplies a named target component and port, not its measured pose. [CheatCode](../../aic_example_policies/aic_example_policies/ros/CheatCode.py) explicitly waits for the target and cable TF frames with `ground_truth:=true`. The robot base is fixed in these scenes, so base-frame TCP is suitable for the proposed global-position proxy without needing a separate `world` transform.

The [official qualification setup](https://github.com/intrinsic-dev/aic/blob/main/docs/qualification_phase.md) randomizes the task-board pose and yaw, NIC rail/card translation and yaw, and SC-rail translation. Its released [evaluation configuration](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml) lists NIC rail translation bounds −21.5 to +23.4 mm and SC rail bounds −60 to +55 mm. The three published example trials use different board poses and target rails; their exact numeric settings are examples, not allowed runtime lookups. The [challenge rules](https://github.com/intrinsic-dev/aic/blob/main/docs/challenge_rules.md) permit ground-truth TF during **training** while prohibiting backend state leaks and exploitative hardcoding during **evaluation**. A target TF or parsed per-trial evaluation YAML therefore cannot be a deployed port position.

## Reproducible replay

Fit a median `base_link` port position per public SFP card/port code from **66 complete training reset groups only**. The decision sees only the measured TCP position and public requested code: stop at the first saved sample where TCP is within a fixed 60, 80, 100, or 120 mm sphere about that median. Only **afterward** use the saved Gazebo scene to measure true port distance, axial/lateral error, and nearest port. All thresholds were reported; none was selected from the inspected diagnostic.

```bash
.pixi/envs/default/bin/python scripts/audit_global_coordinate_handoff.py \
  --train /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1/rows.jsonl \
  --dev /home/nas/chmin/aic_archives/2026-10-05_sfp_coarse_dev_only_labels_v1/rows.jsonl \
  --diagnostic /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten/rows.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_global_coordinate_proxy_v2/summary.json
```

Nominal target-position error (median / p90 / maximum): **34.1 / 71.8 / 121.6 mm** on train, **44.6 / 62.5 / 63.7 mm** on ten grouped development resets, and **47.6 / 55.7 / 63.6 mm** on ten previously inspected, distinct nonfinal Gazebo reset configurations. This is far larger than the 0.5 mm insertion corridor and can approach the separation between neighboring cards. The development label stream was limited to about the first 15 s; its missed threshold crossings are **not** full-episode miss rates.

On the ten complete inspected **CheatCode teacher** rollouts, the 100 mm coordinate sphere triggered in all ten; 9/10 first crossings happened to satisfy the real buffered handoff geometry, axial −100 to −70 mm and lateral ≤10 mm. At 80 mm, it triggered in all ten but only 4/10 first crossings satisfied that gate; at 60 mm, it triggered in eight and **0/8** satisfied it because it was too late. This favorable 100 mm teacher result reflects a trajectory already steered by exact target geometry, not evidence that the coordinate rule finds the port or makes an autonomous actor align.

The decisive replay used **three recorded autonomous SFP policy rollouts**; each had the same observation-only actor and public task code, and the coordinate rule was evaluated without changing control:

```bash
.pixi/envs/default/bin/python scripts/replay_sfp_nominal_coordinate_on_actor.py \
  --nominal-summary /home/nas/chmin/aic_archives/2026-10-05_sfp_global_coordinate_proxy_v2/summary.json \
  --calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --run /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/live_card0_port0_cards5_frozen_history \
  --run /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/live_card2_port0_cards5_audit420 \
  --run /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/live_card4_port1_cards4_frozen_history \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_global_coordinate_proxy_v2/autonomous_three.json
```

| Requested port | First 100 mm nominal crossing | Actual lateral / axial | True buffered handoff? |
| --- | --- | --- | --- |
| Card 0, port 0 | None | — | No |
| Card 2, port 0 | 23.4 s | 31.0 / −96.4 mm; TCP nearer a card-1 port | No |
| Card 4, port 1 | 37.95 s | 99.2 / +18.5 mm | No |

The 80 mm sphere crossed in only card 2/port 0 and was still 31.0 mm laterally off. The 120 mm sphere crossed in two rollouts, neither a valid handoff. Thus the coordinate rule would have caused false stops in the cases where the autonomous actor reached its nominal neighborhood. The original scorer also found **0/3** real buffered visits for these actors. A coordinate alarm cannot repair a transport route that misses the port or tell which neighboring opening it faces.

## Limits and next use

These are **replays** of saved nonfinal Gazebo rollouts, not newly controlled experiments. A later audit corrected the launch note: `117/117` was the instantiated-key field of `/proc/key-users`, **not** the user-key quota. The actual quota had been raised to **1024**, and a corrected preflight launched six later paired-request Gazebo trials successfully. This does not change any coordinate-replay result. No final IK scene was opened, no training or active job was interrupted during this replay, and no threshold was tuned on the inspected diagnostic. The SC task was checked against official randomization but has not had the same coordinate-only rollout replay; its ±60 mm rail shift gives no reason to assume a tighter nominal prior.

Use the public task code plus measured base-frame TCP as a cheap **prior** or cadence trigger for looking more closely. Keep visual selected-port evidence, an observation-only identity check, pose refinement and force/standoff checks for handoff. A later matched test could compare a wide coordinate cue versus no cue on new nonfinal autonomous starts, with **zero unsafe stops** and at least 9/10 safe early handoffs as the gate. Do not promote it from the teacher replay.

Artifacts: [training/development/teacher replay](</home/nas/chmin/aic_archives/2026-10-05_sfp_global_coordinate_proxy_v2/summary.json>) · [autonomous three-run replay](</home/nas/chmin/aic_archives/2026-10-05_sfp_global_coordinate_proxy_v2/autonomous_three.json>) · [small machine summary](2026-10-05-global-coordinate-arrival-proxy.summary.json). All per-episode decisions and posthoc geometry are retained there.
