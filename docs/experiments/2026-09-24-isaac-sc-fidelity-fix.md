# Isaac SC collision and grasp repair, September 24

Status: **one constructed five-card development scene now passes a normal-collision scripted mechanics check.** Two deliberate near-port offset commands produce recorded plug/port blockages. These are diagnostic Isaac traces, not autonomous visual-policy rollouts, a scored insertion result, or accepted RL replay.

Follow-up: [corrected-asset card-route probes](2026-09-24-isaac-sc-contact-routes.md)
show that a deliberately poor across-card route does create a named
gripper/card jam even though the centered route below did not. Cable links
also touch a card during that gripper jam; an independent cable snag is not
yet established.

## What was wrong

The earlier five-card Isaac contact was not a reliable cable-snag example. Several independent import and reset errors produced collisions that the scored Gazebo SC episode did not have:

1. The prepared Isaac robot gripped the wrong cable end. The scored Gazebo bag 251 gave a stable `ati/tool_link -> sc_plug_link` transform; the rebuilt robot now uses it. Bag 252 agreed to about 0.025 mm in translation. The first cable body also sat beside the port in Isaac, although Gazebo put it behind the plug. Its endpoint and first-link seed now come from the scored Gazebo TF. The downstream cable shape and its ball-joint behavior still need validation.
2. The SC target orientation was nearly 180 degrees wrong, and the pose controller's old XY/Z sign corrections moved this SC topology away from the target. The live port-base orientation is now the generated SC orientation; the probe uses `--no-fix_isaac_ik_xy_sign --no-fix_isaac_ik_z_sign`. The joint-replay probe also now selects joint-position action mode before creating the environment.
3. A 5 kN gripper-base contact came from its fixed neighboring `ati_base_link`, not a card. The builder filters that exact pair. In the calibrated first-link seed, the grasped cable collided with its own gripper at up to 3.6 kN even when the port was absent. The selected asset filters gripper-to-held-plug/first-link pairs, while retaining plug, gripper, cable, board, card, and port scene collisions. This is an explicit grasp-contact approximation; it does not prove Gazebo and Isaac have identical cable contact physics.
4. The distributed SC-port USD used visual-mesh convex hulls that close the opening. A new builder copies all 15 source SDF boxes/cylinders. Its first version still blocked the plug 7.7 mm before the scored Gazebo tip pose. A collider-group ablation isolated `cube_collider_box.002` (the bottom plate): bottom plate alone blocked, while top plates or walls alone did not. The scored Gazebo TF showed why: Isaac's port rigid root is aligned with Gazebo's **port-base** frame, while those colliders were placed as if it were the **port-model** frame. The builder now includes the missing 180-degree Y model-frame alignment. All source SDF collision shapes are retained.
5. The old Isaac SC depth was based on first contact with the wrong asset. Scored Gazebo bags 251 and 252 put the tip near `sc_port_base_link`, 15.64 mm beyond its entrance. The generator now targets this depth (`root-local z=+2 mm`). A reset check also rejects a board world position that disagrees with the configured board default, preventing a silent scene mismatch.

The transform calibration files are [tool-to-plug](../../configs/hierarchical_recovery/sc_tool_plug_gazebo_tf_251.json) and [first cable links](../../configs/hierarchical_recovery/sc_cable_first_links_gazebo_tf_251.json). The input scene is a **constructed, unsealed five-card development scene** using bag-251 SC geometry and extra cards: [episode YAML](../../configs/hierarchical_recovery/isaac_sc_fidelity/episodes/episode_000001.yaml), SHA-256 `354ff524705aa2ea23f46ae799ed54d1539dc23755d960a4c8c456c8070fde5a`. The source scored bag had three cards. No reserved final configuration was opened.

## Measured probes

All rows below use the same corrected robot and all corrected SDF port colliders, except the historical wrong-frame control. They are one run each on one scene. The controller uses the simulator target to diagnose mechanics; it is **not** the deployable actor. At 20 Hz, the complete rows and contacts are in local `outputs/experiments/2026-09-24_isaac_sc_fidelity/`.

| Probe | Start and command | Closest distance to commanded tip target | Final axial / lateral error to true port-base target | Peak plug/port contact | Peak gripper/first-link contact |
| --- | --- | ---: | ---: | ---: | ---: |
| Wrong port frame, filtered grasp | Near port; centered | 7.66 mm | 7.73 / 0.35 mm | 40.0 N | 0 N |
| Corrected port, filtered grasp | Near port; centered | **0.074 mm** | **0.073 / 0.016 mm** | 1.10 N | 0 N |
| Corrected port, filtered grasp | Full start, five cards; centered | **0.078 mm** | **0.073 / 0.054 mm** | 1.04 N | 0 N |
| Corrected port, filtered grasp | Near port; +3 mm lateral command | 5.94 mm | 5.27 / 0.38 mm | 54.4 N | 0 N |
| Corrected port, filtered grasp | Near port; -3 mm lateral command | 12.86 mm | 12.67 / 0.75 mm | 92.2 N | 0 N |

The offset commands are measured against their own commanded targets in the third column. Their final lateral values are measured against the **true centered port-base target**, because contact prevented the full requested sideways motion. Final orientation errors for the two blocked runs were 0.74 and 0.88 degrees. The centered runs ended within the geometric target with about 0-degree orientation error. No scorer event was available in these Isaac probes, so “geometric target reached” is the supported conclusion. The corrected **centered** full-start route did not reproduce the old gripper/card hit; the previous item-2 trace should not be treated as a valid training incident. Deliberately crossing routes have separate named card-contact evidence, reported in the follow-up above.

## Build and replay

Run [the asset builder](../../scripts/build_sc_isaac_fidelity_assets.sh) inside the existing rootless IsaacLab container, with the repository mounted at `/workspace/isaaclab/aic`:

```bash
docker exec aic_sc_fix_gpu1 bash /workspace/isaaclab/aic/scripts/build_sc_isaac_fidelity_assets.sh
```

It writes ignored sibling USDs under `Intrinsic_assets/`. For this run their SHA-256 values were:

```text
d797c81de7510f90efa0c0d92bfa67478894aca50555548c1c67f55c49fa606d  aic_unified_robot_cable_reversed_gazebo_firstlink_aligned_filtered.generated.usd
3ea2034e05c50bfbd0bcab3c2f618ed8b207cde21f7bb73b559ad7d88c24795f  assets/SC Port/sc_port_sdf_aligned.generated.usd
```

The original exact commands are retained as `run_filtered_aligned_probe.sh`, `run_offset_plus3_x.sh`, `run_offset_minus3_x.sh`, and `run_full_start_five.sh` under `/var/tmp/aic_sc_fix_20260924/` and copied with the output JSON and selected logs to the ignored local `outputs/experiments/2026-09-24_isaac_sc_fidelity/`. The maintained [probe runner](../../scripts/run_sc_isaac_fidelity_probe.sh) resolves tracked scene and generated asset paths from the repository:

```bash
docker exec aic_sc_fix_gpu1 bash /workspace/isaaclab/aic/scripts/run_sc_isaac_fidelity_probe.sh full_start
docker exec aic_sc_fix_gpu1 bash /workspace/isaaclab/aic/scripts/run_sc_isaac_fidelity_probe.sh near_center
docker exec aic_sc_fix_gpu1 bash /workspace/isaaclab/aic/scripts/run_sc_isaac_fidelity_probe.sh near_plus3
docker exec aic_sc_fix_gpu1 bash /workspace/isaaclab/aic/scripts/run_sc_isaac_fidelity_probe.sh near_minus3
```

The maintained `full_start` runner was rerun once: it reproduced the original
161-row trace's 0.078 mm closest distance, 0.073 mm final axial error,
0.054 mm final lateral error, and 1.043 N peak plug/port contact.

The original centered full-start command was:

```bash
export AIC_ISAAC_EPISODE_CONFIG_DIR=/workspace/isaaclab/aic/configs/hierarchical_recovery/isaac_sc_fidelity
export AIC_ISAAC_ROBOT_USD_PATH=/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/aic_unified_robot_cable_reversed_gazebo_firstlink_aligned_filtered.generated.usd
export AIC_ISAAC_SC_PORT_USD_PATH='/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/assets/SC Port/sc_port_sdf_aligned.generated.usd'
# Run inside the rootless container with IsaacLab's documented PYTHONPATH.
python.sh aic_utils/aic_isaac/aic_isaaclab/scripts/serl/probe_target_reward.py \
  --headless --task_family sc_to_sc --target_body sc_tip_link \
  --controller target_body_pose_root \
  --no-fix_isaac_ik_xy_sign --no-fix_isaac_ik_z_sign \
  --target_position_offset 0 0 0.002 --insert_offset 0 0 0.002 \
  --steps 160 --descent_start_step 80 --descent_end_step 130 \
  --debug_named_contacts --output /sc_fix/full_start_five.json
```

For the near-port variants, the retained scripts also pass the Gazebo joint seed, 600 physical interpolation steps, 300 hold steps, and parked scene objects during interpolation. Use the stored scripts for exact reproduction; a direct joint teleport snaps the cable into contacts and is not equivalent.

## What can be collected now, and what remains

The corrected assets allow **physics-only diagnostic Isaac episodes** with the five-card scene, including centered approach and true plug/port blockage. Save causal action, measured TCP/plug/cable motion, wrist force, named contacts, reset identity, and terminal observation for any replay. Do not train a recovery policy on the old wrong-frame or unfiltered internal-contact traces. No cable/card snag has been causally reproduced with this corrected setup, and the distal cable motion is not yet source-validated.

Before the container fix below, RGB collection was blocked on this rootless Isaac Sim 5.1 / driver 535.104.05 setup. A separate three-step camera smoke test on the **repaired** assets failed before rollout: the local renderer log states installed 535.104 is within its unsupported `[0.0, 535.129)` range and recommends 535.161.07; it then reports `rtx driver verification failed`. The [smoke log](../../outputs/experiments/2026-09-24_isaac_sc_fidelity/camera_smoke.log) and script are saved with the probes. The earlier camera attempt also hit a CUDA illegal-memory-access when reading output. The no-camera probes complete. Treat these as structured mechanics incidents, **not visual BC or online SERL episodes**, until the same scenarios are rerun with images. No unrelated active jobs were interrupted; all new probes used GPU 1 only.

### Camera follow-up, September 24

The immediate regression was container configuration: the earlier working
`aic_isaac_world_policy_20260920` container had the repository's local
[RTX driver-check override](../../aic_utils/aic_isaac/aic_isaaclab/scripts/patch_isaac_rtx_driver_check.sh),
while the newer `aic_sc_fix_gpu1` container did not. Both use Isaac Sim 5.1
and host driver 535.104.05. I applied that existing, reversible override
**inside the new rootless container only**:

```bash
docker exec -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 aic_sc_fix_gpu1 \
  bash /workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/patch_isaac_rtx_driver_check.sh
```

The four-frame repaired-scene smoke test then completed and produced distinct
224×224 left, center, and right wrist-camera H.264 videos. I decoded and
visually inspected their first frames. A separate **diagnostic-only** 512×512
overhead camera shows the five-card row and an orange cable segment; the
generated rope link's authored display color is `(1, 0.22, 0)`, matching the
source Gazebo cable's diffuse color. The external camera is not an actor
input. [Wrist frames, diagnostic frame, videos, exact scripts, JSON, and
logs](../../outputs/experiments/2026-09-24_isaac_sc_fidelity/camera_smoke_repaired/)
are retained locally. The maintained probe supports
`--debug_overview_camera` for this visual audit.

This resolves the **camera initialization/smoke-test blocker in this
container**, not the host driver incompatibility. A recreated container
needs the same override, and a longer moving-cable visual episode plus
Gazebo/Isaac appearance comparison is still required before SC visual BC/RL.
The earlier failed smoke and logs remain preserved above.

A follow-up 61-frame, 20 Hz clear-lane scripted motion clip also completed
without renderer failure. The [four videos and 0/30/60-frame sheets](../../outputs/experiments/2026-09-24_isaac_sc_fidelity/camera_motion_clear_lane/)
show changing wrist views, the five-card row, and orange cable. Its peak wrist
force was 21.6 N; the 60-step clip intentionally stops 34.2 mm from the final
tip target and is **not an insertion evaluation**. On the overview and left
camera frames the moving distal cable can look sharply kinked and looped,
unlike the smoother Gazebo cable in the retained wide-view video. The source
USD has the right orange color, but distal cable shape and dynamics remain a
visual/physics fidelity concern. Do not use this one clip to certify
Gazebo-to-Isaac appearance parity or train a cable-recovery policy.

## Cable self-collision diagnosis and corrected moving view

The September 24 follow-up extracted all 20 cable-link poses from the scored
three-card Gazebo bag 251 at six times, then recorded the same 20 link centers
in a constructed five-card Isaac development scene. These are **not
time/action-matched** trajectories, so the [shape comparison](2026-09-24-sc-cable-shape-comparison.json)
is a plausibility diagnostic, not a numerical sim-to-sim fidelity score.
The original Isaac moving run compressed at least one adjacent link-center
distance to 4.4 mm and made four turns above 90 degrees; the scored Gazebo
samples had minimum distance 41.2 mm and no turns above 90 degrees. The bad
shape also appeared with a stationary arm. It was not caused only by the
transport command.

The original Isaac articulation had robot self-collision enabled, although
the source Gazebo cable SDF does not request it. At the end of a stationary
Isaac probe, the two gripper fingers each reported about 20.8 kN of internal
contact and distal rope links 19/20 about 151/160 N. With robot self-collision
disabled, those internal forces were zero. The moving run then had a minimum
adjacent link-center distance of 37.1 mm, no turn above 90 degrees, and a
visibly smoother orange cable. The [no-self-collision shape summary](2026-09-24-sc-cable-shape-no-self-collision.json),
[61-frame 20 Hz videos and comparison frames](../../outputs/experiments/2026-09-24_isaac_sc_fidelity/cable_self_collision_ablation/)
preserve the result. The earlier kinked clip remains preserved for comparison.
The rendered overhead camera is diagnostic-only, not an actor observation.

Alternative tests did not explain the failure as well. Raising articulation
solver iterations from 16/8 to 64/32 gave inconsistent compression and shape.
The USD import's ball-joint drives/limits differ from the source SDF, but a
bounded joint-physics variant either made the rope rigid or made the kinks
worse. Neither variant was selected. The selected asset remains the previous
normal-collision USD with `AIC_ISAAC_ENABLE_ROBOT_SELF_COLLISIONS=0`; **scene**
collisions remain enabled. On the same five-card full-start scripted motion,
the SC tip reached 0.074 mm from its seated target, with 0.88 N final
plug/port contact. On a deliberate gripper/card route, contact with
`nic_card` still reached 364 N and wrist force 395 N before the scripted
controller subsequently inserted. This is a high-force diagnostic incident,
not a safe teacher trajectory or learned recovery success. The maintained
`scripts/run_sc_isaac_fidelity_probe.sh` now sets the chosen self-collision
mode explicitly, and probe JSON records the simulation contact setting.

The source Gazebo bag had three cards, whereas this Isaac scene has five. A
controlled derivative changed only card presence: scripted full-start motion
with zero, three, and five cards each ended 0.074 mm from the seated target,
with 2.26 N peak wrist force. A separate 3 mm lateral near-port target
remained 12.79 mm from its seated target and produced 47.4 N peak plug/port
contact. [Exact trace pointers and metrics](2026-09-24-isaac-sc-self-collision-gate.json)
are retained. These probes clear the **bounded mechanics/image gate** for
SC pose/data development. They do not prove isolated cable snag or exact
Gazebo dynamics, and the extreme card-route force is not a safe policy target.
