# SC-to-SC Isaac mechanics and routing bring-up

Date: 2026-09-23  
Status: mechanics partly validated; multi-card result is diagnostic proxy evidence

## Why this experiment was required

The approved SC continuation begins by proving that Isaac has the correct plug,
port, grasp, insertion depth, card geometry, cable topology, and terminal
accounting. Training BC or RL before that check would teach a policy to work
around simulator setup errors.

No BC, critic, or actor was trained in this bring-up. Every motion below is a
privileged joint-space mechanics probe. True simulator geometry and IK targets
are not deployable actor inputs.

## Corrections made before interpreting a rollout

Several earlier SC probes were invalid for different reasons.

1. The SC ports were mounted 5 mm above the board. The source task-board xacro
   mounts them at 16.5 mm; the old value embedded the receptacle in the board.
2. The prepared cable had LC at the gripper and SC at the free end. The source
   reversed SC task has SC at rope endpoint 0 and LC at endpoint 20. The new USD
   builder rewires those fixed joints and verifies their residual transforms.
3. A builder early return ignored requested collision disables. This made old
   “no collision” conclusions unreliable. The builder now applies and reports
   every requested disable.
4. Teleporting the arm while leaving the cable behind caused a large first-step
   snap. Near-gate reset can now use actuator-driven interpolation and records
   the solved and realized joint states.
5. The old target was the center of Gazebo's solid SC touch-sensor box. A plug
   cannot physically move its tip to the center of that collider. The current
   empirical stable-contact target is port-local `(0, 0, -4.65 mm)`, or 8.99 mm
   from the nominal entrance. A recorded Gazebo terminal pose should refine
   this value later.
6. The reachable Isaac grasp put the gripper housing into the board. A 30 mm
   connector extension provides board clearance while preserving exact fixed
   joint transforms. This remains an Isaac mechanics approximation.

## Zero-card physical insertion and local recovery

With all normal collisions active, a direct approach stopped 2.60 mm from the
contact target. A 14 mm retreat followed by a 4 mm lateral retry in one
direction reached 0.084 mm, ended at 0.099 mm lateral and 0.008 mm axial error,
and remained inside the component gate for 53 rows. The opposite lateral retry
briefly reached 0.505 mm but later diverged.

This is useful evidence for the proposed recovery structure:

```text
blocked approach
      |
      v
retreat far enough to unload contact
      |
      v
try a coherent lateral direction
      |
      v
reapproach and hold contact
```

It also shows why the direction cannot be arbitrary. The positive 4 mm retry
succeeded; the matched negative retry did not remain successful.

## The first five-card “snag” was actually gripper-card collision

The normal full-start five-card probe stopped about 13.2 mm away with about
12.5 mm lateral error and a 165.8 N peak wrist load. The left-camera video made
the cable look suspicious, but a collision ablation gave the causal answer:

- disabling rope, plug, LC, and SFP collisions did **not** change the failure;
- a contact sensor on `gripper_hande_base_link` measured about 128 N against a
  NIC card;
- disabling only the gripper-base colliders reduced the best miss to 2.14 mm;
- disabling all ten gripper colliders exposed a different, smaller interaction.

Therefore the original five-card failure is not valid cable-snag data. The
reachable vertical grasp makes the physical gripper cross a horizontal NIC
card. The source reversed Gazebo grasp rotates the gripper differently, but it
is not physically reachable in the current Isaac board/base arrangement: the
arm hits its scene or limit envelope before attaining that pose. Roll-offset
grasp probes at plus and minus 90 degrees also failed to provide a stable,
reachable full insertion.

The [invalid original-grasp video](../../outputs/experiments/2026-09-23_sc_cable_bringup/videos/invalid_original_grasp_nic5_left.mp4)
is retained so this failure is not later mistaken for learned cable behavior.

## Bounded cable-routing proxy

For a diagnostic only, all gripper collision shapes were disabled while the SC
plug, rope, remaining connector, cards, board, and port collisions stayed
active. This approximates the clearance that the source grasp should provide;
it is not a deployable scene fix.

Three seeds compared the same scripted path with and without a large route
around the card edge. The gate is evaluated by components: lateral at most
0.5 mm, axial at most 1 mm, and orientation at most 2 degrees.

| Method | Stable final gates | Final lateral, mm | Final axial, mm | Peak wrist force, N |
| --- | ---: | --- | --- | --- |
| Direct high approach, seeds 1--3 | 1/3 | 3.215 / 0.095 / 7.783 | 4.394 / 0.635 / 3.213 | 62.4 / 179.6 / 65.3 |
| High approach, +100 mm side route, lower on the outside, then return, seeds 1--3 | 3/3 | 0.147 / 0.112 / 0.101 | 0.485 / 0.385 / 0.459 | 42.0 / 30.6 / 42.2 |

The route is temporally coherent: rise above the cards, move 100 mm around the
safe side, lower outside the card footprint, and then return toward the port.
This supports a hierarchical transport/recovery option. It does not establish
an autonomous policy result because its waypoints came from simulator IK.

Camera rendering changes the PhysX outcome enough to matter. The recorded
seed-1 route entered the component gate briefly at step 489, then ended at
1.89 mm lateral error. The metrics-only seed-1 run remained in the gate for 52
rows. Preserve the [recorded route video](../../outputs/experiments/2026-09-23_sc_cable_bringup/videos/proxy_x100_route_seed1_left.mp4)
as a separate rollout rather than overwriting the matched metrics result.

## Decision

The following claims are supported:

- zero-card SC insertion mechanics can work in the corrected scene;
- retreat plus directional lateral retry can resolve a local contact;
- the original multi-card failure was caused primarily by the grasp/scene
  contract;
- in a declared gripper-clearance proxy, a large routed transport is much more
  reliable than a direct approach.

The following claims are not supported:

- the faithful Isaac scene reproduces natural cable snagging;
- the proxy route is an autonomous or learned solution;
- existing SC perception, BC, or RL is ready for this scene.

Do not collect SC BC/RL data until the faithful grasp/scene contract is fixed.
The best next engineering task is to reproduce the source reversed grasp with a
reachable robot/base/board arrangement or an equivalent collision-faithful
gripper model. After that, rerun card counts 0--5 and require stable physical
insertions before beginning the bounded discovery set.

## Artifacts

- [Machine summary](../../outputs/experiments/2026-09-23_sc_cable_bringup/summary.json)
- [Artifact map](../../outputs/experiments/2026-09-23_sc_cable_bringup/artifact_map.json)
- [Checksums](../../outputs/experiments/2026-09-23_sc_cable_bringup/sha256.txt)
- Bulk diagnostics: `/var/tmp/chmin_aic_20260920_isaac_world_rl/sc_snag_20260923/`

## Reproduction entry points

All simulator work used the rootless container
`aic_isaac_world_policy_20260920`. The generated USD files are intentionally
ignored because they are derived from the distributed asset. Rebuild the
collision-faithful 30 mm clearance asset inside the container with:

```bash
cd /workspace/isaaclab/aic
ASSETS=aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets
/workspace/isaaclab/_isaac_sim/python.sh \
  aic_utils/aic_isaac/aic_isaaclab/scripts/build_sc_reversed_robot_usd.py \
  --headless --mode reversed_topology \
  --source "$ASSETS/aic_unified_robot_cable_sdf.usd" \
  --output "$ASSETS/aic_unified_robot_cable_reversed_clearance30.generated.usd" \
  --sc-grasp-extension-m 0.03 --print-topology-transforms
```

The diagnostic gripper-clearance asset adds these options to the same command:

```text
--deinstance-substring /World/aic_unified_robot
--disable-collision-substring gripper_hande
```

Run mechanics probes through
`aic_utils/aic_isaac/aic_isaaclab/scripts/serl/probe_target_reward.py` with
`AIC_ISAAC_ROBOT_USD_PATH` set to the generated asset and
`AIC_ISAAC_ARM_ACTION_MODE=joint_position`. The exact per-run JSON paths and
metrics are retained in `summary.json`; the corresponding stdout logs remain
beside them under the bulk diagnostics directory. The generated episode YAMLs
and requests are under that directory's `generated/` and `requests/` folders.
