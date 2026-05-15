# Isaac Near-Gate Handoff - 2026-05-15

This note summarizes the Isaac `start_near_gate` / near-gate insertion debugging done on this server. Treat these results as environment-specific. The server used for these experiments had visible asset/camera problems: the SFP tip/plug visual was missing or not reliably visible in camera images until asset references were repaired, and some camera views looked wrong. The next server is expected to be based on the latest remote `feat/hybrid-train`, where the cheatcode reportedly works and the tip is visible. Results below are useful mainly as diagnostics and experiment design notes, not as final tuning conclusions.

## Server-Specific Caveats

- The official NVIDIA `Intrinsic_assets.zip` was re-downloaded and installed according to the upstream Isaac README. In this runtime, the downloaded pack still left direct `.glb` visual references for `lc_plug_visual`, `sfp_module_visual`, and `sc_plug_visual` inside `aic_unified_robot_cable_sdf.usd`.
- Isaac Sim on this server failed to open those direct GLB references, producing missing plug visuals. This can make camera-based conclusions unreliable.
- The live asset directory was repaired locally by converting those GLBs to USD and patching the robot USD references. That repair is local/generated and should not be assumed necessary on the next server if assets already render correctly.
- Camera constants appeared unchanged from upstream: `PinholeCameraCfg`, `224x224`, ROS camera offset `(pos=(0,0,0), rot=(1,0,0,0))`, and optical prim paths were the same.
- A branch-local robot reset regression was found: `shoulder_pan_joint` had been flipped from upstream `0.1597` to `-0.1597`. Since cameras are robot-mounted, that can make views look wrong even when camera constants are correct.
- Physics did start. Cable/plug dangling was likely from dynamic cable joints and only the arm joints being actuated, not from a fully paused simulation.

## Near-Gate Experiments Run Here

The experiments focused on whether Isaac could produce physically visible insertion from near-gate starts before doing reward tuning.

- Restored `shoulder_pan_joint` to upstream `0.1597`.
- Added SFP geometry diagnostics and tests around target, entrance, and insertion axis.
- Replaced inconsistent hardcoded SFP seated target constants with a target derived from:
  - SFP port root-local entrance
  - port rotation / insertion axis
  - seated depth
- Added fail-fast validation for target geometry:
  - target must be collinear with entrance and insertion axis
  - seated depth must be within expected bounds
  - lateral error must be near zero
- Added all-body insertion diagnostics for `wrist_3_link`, `gripper_tcp`, `sfp_tip_link`, and `sfp_module_link`.
- Added `cheatcode_tcp` action-guide mode to compare Isaac guide semantics with the Gazebo cheatcode idea.
- Ran short near-gate / cheatcode probes with debug overlays and videos under:
  - `outputs/debug/isaac_cheatcode_fix_20260515/`

Important video artifacts from this server:

- Missing/asset visual probe:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r106_asset_visual_probe_env0_center.mp4`
- Module-link cheatcode probe:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r107_module_link_cheatcode_env0_center.mp4`
- Earlier tip-link cheatcode side-by-side:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r105_fixed_geometry_cheatcode_ikbody_no_xy_sign_env0_env1_center_side_by_side.mp4`

Observed on this server:

- Semantic metrics could show partial insertion or positive SFP tip depth while the camera did not show convincing physical insertion.
- With `target_reward_body=sfp_tip_link`, diagnostics could show the tip near/inside the target plane while the module body was still visibly outside. This suggests that tip-only metrics can overstate insertion if the visible module/plug body does not follow.
- Switching the guide to a module-link target did not fix insertion here; in one run it drove the module farther away.
- The global Isaac IK x/y sign fix was not sufficient by itself.
- Because plug visuals were missing for part of the debugging, any purely visual conclusion from these runs should be rechecked on the new server.

## What To Reuse On The New Server

If the new server has the latest branch, visible plug/tip, and working cheatcode, do not start by applying the local asset repair or old visual conclusions. Instead, reuse the diagnostics pattern:

1. Confirm camera and asset sanity first.
2. Run one no-learning near-gate cheatcode probe with overlays and save video.
3. Compare physical video against semantic diagnostics for:
   - `sfp_tip_link`
   - `sfp_module_link`
   - port entrance
   - seated target
   - insertion axis
4. Only tune insertion behavior after physical and semantic diagnostics agree.
5. Prefer minimal changes per iteration, with a short run and video after each change.

Most useful checks for the new server:

- Is the visible SFP module aligned with the port mouth at reset?
- Does the cheatcode move the visible module toward the port, not just the semantic tip?
- Do lateral error, axial depth, and orientation error improve monotonically during insertion?
- Does any positive insertion depth correspond to visible physical insertion?
- Does cable dynamics pull the module away after reset or during the first few sim steps?
- Does `sfp_tip_link` differ from `sfp_module_link` enough that the reward body should be changed or supplemented?

## Suggested Prompt For The Next Server

Use a prompt like this for autonomous iterative work:

```text
We are on the new server with the latest remote feat/hybrid-train where the Isaac plug/tip is visible and the cheatcode works. Do not reuse conclusions from the older server unless revalidated here.

Goal: improve SFP-to-NIC near-gate insertion in Isaac, not reward-tune blindly.

Work iteratively and autonomously:
1. Run a short near-gate cheatcode/guide probe with cameras, overlays, diagnostics, and saved video.
2. Inspect video and diagnostics together. Check whether visible SFP module insertion matches semantic metrics for sfp_tip_link, sfp_module_link, entrance, target, insertion axis, lateral error, axial depth, and orientation.
3. If physical and semantic metrics disagree, fix frames/geometry/target body/reset semantics before changing rewards.
4. If they agree but insertion fails, make one minimal change to guide/reset/action-frame/insertion-axis behavior.
5. Re-run the same short probe, compare against the previous run, and repeat.
6. Save every run's video path, diagnostics summary, exact command, and a concise explanation of what changed and whether it helped.

Use the Gazebo cheatcode as the semantic reference, but verify the Isaac adaptation body frame, root frame, x/y sign, target body, and insertion axis directly in Isaac. Do not assume tip-depth success is valid unless the video shows the visible plug entering the port.
```

## Recommendation

Share this note with the next server, but frame it as a cautionary handoff. The useful transferable parts are the diagnostic methodology, the suspected shoulder-pan/camera coupling, and the warning that `sfp_tip_link` metrics can disagree with visible module insertion. The actual failed videos from this server should not be used as evidence that the latest remote branch fails, because this server had asset/rendering differences and possibly stale branch state.
