# SC port-0 orientation correction and selected-port identity audit

**Date:** October 3, 2026. **Decision:** keep automatic handoff, insertion and RL disabled. The guarded image-derived orientation correction worked mechanically, but the existing selected-port identity head rejected the port after rotation. A bounded identity refit repaired that false rejection on saved images but failed the physically absent-port test on a new reset. Neither checkpoint nor runtime switch is promoted. The four reserved final IK scenes remain sealed.

## Hypothesis and trace

The [previous crop identity ablation](2026-10-03-sc-original-rgb-crop-identity.md) exposed a 6.71° posture error behind the original port-0 identity miss. I compared the transport controller's **body-frame orientation command**, measured TCP quaternion and fresh RGB-derived port orientation at the actual proposals. On the known port-0 pause, measured-to-commanded rotation was only about **0.2–0.3°**, while the command differed from the board-derived target by about **6.9°**. The controller was tracking its command; the transport orientation target itself was wrong for this board pose. Four saved accepted controls had measured posture errors of roughly 0.07–2.05°.

The previous verifier returned immediately on identity rejection, so it never surfaced a geometric orientation candidate. I added an **opt-in, observation-only** path that computes the existing three-view opening and board-line estimate after an identity miss. It may set a held-state orientation target only when the opening reprojection residual is ≤2 px, the board-view yaw spread ≤1°, predicted standoff discrepancy ≤15 mm, measured force ≤40 N, and posture error is between 3° and 10°. The identity miss still returns **invalid** and cannot hand off. The existing orientation correction path for a `wrong_tcp_posture` rejection is unchanged. All experimental switches default off.

An offline replay found the known port-0 miss eligible for this guarded rotation at **6.71°**. The other saved good controls and all 20 provisionally hidden requests retained their prior acceptance/rejection outcomes; no hidden example was an orientation-correction candidate. This is a JPEG proxy screen, not a native-buffer safety certificate.

## Live shadow checks

Rootless Gazebo used GPU 1 with the policy on GPU 2; no more than two project GPUs ran for a live rollout. Both runs used the same frozen transport and perception checkpoints, scene and selected port. No policy had oracle geometry, no insertion actor ran, and all motion came from the transport actor plus the explicitly gated held-state orientation servo.

| Development run | Closest true transport standoff, post-run | Fresh inspections | Result |
| --- | ---: | --- | --- |
| Known port 0, orientation option alone | 0.45 mm | 0 | Three-view proposal became stale; correction never triggered. |
| Known port 0, existing safe stale-pause option plus orientation correction | 0.66 mm | 3 | First posture error **6.73°**; after correction **0.13°**, then **0.08°**. Identity failed all three, so no handoff. |

The stale-pause option asks for a **fresh** verifier inspection even when the earlier three-view transport proposal is stale; it cannot accept from stale RGB. The first live run's visual target got within about 1.18 mm but had a >3 px reprojection residual and a stale last-good view, explaining its zero inspections. The second run exercised the correction. The [known port-0 inspection montage](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/p0_known_stale_orientation/review/all_inspections.jpg) uses RGB-locator crop centers, not true simulator port coordinates. The port remains visible in the sampled views after rotation.

I froze [one further five-card port-0 reset](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/development_new_p0_fivecard/manifest.json) before its matched runs. Its complete-reset hash differs from the prior one-port and crop-development manifests. The baseline and correction runs used the same scene and stale-pause setting, but followed different transport routes:

| New-reset run | Closest true standoff | Fresh inspections | Result |
| --- | ---: | ---: | --- |
| Frozen baseline, no identity-miss correction | 82.84 mm | 0 | Transport never reached an inspectable pose. |
| Opt-in correction | 1.04 mm | 3 | Existing `wrong_tcp_posture` path rotated from **8.93°** to **0.03°**, then **0.13°**; identity rejected the corrected view. |

The route divergence makes this **not a causal insertion or transport improvement**. The correction run's three inspection forces were about 20.6–20.9 N, below the unchanged 40 N gate; the scored tier-3 insertion result was zero, as expected for transport-only shadow work. The [new-reset inspection montage](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/new_p0_orientation_correction/review/all_inspections.jpg) again shows the selected opening in the RGB-locator crops. Both real requested ports stayed unaccepted, and the absent-port counterfactual never commanded motion.

## Why identity changed after a correct rotation

The current spatial identity head consumes RGB features, selected-port code and measured TCP pose **including its quaternion**. In the known port-0 live run its per-view selected-port scores went from **0.190/0.645/0.938** at 6.73° posture error to **0.003/0.045/0.848** after posture correction to 0.13°. The RGB locator coordinates, reprojection residual and board lines remained stable.

I held an exact saved RGB triplet and TCP position fixed and changed **only the input TCP quaternion** from the before-rotation sample to the after-rotation sample. The same-image scores fell from **0.214/0.651/0.951** to **0.023/0.118/0.677**. Swapping TCP position alone left the scores essentially unchanged (**0.210/0.646/0.949**). In the new reset, changing only the corrected image's quaternion back to its earlier value raised a view from **0.426** to **0.964**. This isolates the model's sensitivity to the quaternion input; changing the camera image or position is not required for that *diagnostic* score shift. Since the cameras move with the TCP, this artificial same-image quaternion swap is not a physically valid training pair. The live before/after image pairs are physically valid and also show rejection after rotation. Together they support an orientation-dependent shortcut in this checkpoint, without proving that every orientation-dependent feature is spurious. [Known-reset four-way state swap](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/p0_known_stale_orientation/identity_state_swap_v2.json), [new-reset replication](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/new_p0_orientation_correction/identity_state_swap_v2.json).

I trained one bounded spatial-head refit with the measured TCP quaternion masked to zero after normalization, leaving full-image features, camera and requested-port code, measured TCP position and frozen locator intact. Eight older one-port and 83 two-port training reset groups supplied its fitting examples; four older plus four crop-development one-port groups were held out from fitting. The **new five-card reset was frozen before this fit** and excluded completely. On saved JPEG proxy frames, the refit accepted both post-correction port-0 inspections, retained the previously accepted controls, rejected swapped and early requests and 20/20 provisional hidden cases. This supports the shortcut diagnosis, **not deployment**.

The independent new-reset absent-port challenge failed: the refit accepted the physically nonexistent SC port in **11/100** synchronized triplets, versus **0/100** for the frozen original head. Across the four earlier crop resets plus this new one, projected in-frame selected-port recall was 212/213 versus 184/213 original, but absent-port acceptance was 11/244 versus zero. Some absent scores were above 0.99 in all three views, so raising the threshold is not a reliable repair. Projected-in-frame is not certified visibility; physically absent is an unambiguous negative. The refit must not control a robot. [Machine comparison](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/identity_pair_four_plus_new_v1.json).

## Decision and bounded next hypothesis

The orientation correction itself is worth retaining as **experimental, off-by-default code**: it uses observation-only geometry, improved measured posture on two live runs and never bypassed identity or force checks. The handoff is still not safe or reliable because the original identity head fails after rotation, while the orientation-masked head falsely accepts an absent port on a new scene. Do not combine them in live autonomous insertion.

**Revised next experiment:** collect a small, targeted set of *real simulator observations before and after a held-state TCP rotation* on the same complete reset, with matching camera images, measured TCP pose and requested-port label at each instant. Include SC port 0 and 1, one-port absent counterfactuals, both-port neighbors and clear versus partly blocked views. Use a few such **training** resets, freeze separate new development resets before fitting, and keep the selected-port supervision constant across a physical rotation when the port remains identifiable. Compare the current spatial head with the same architecture trained on these physically consistent view pairs, giving absent requests explicit loss. Keep the TCP quaternion available to geometric triangulation and port-pose calculation; test whether the identity head can rely on the RGB spatial layout and task code without a brittle quaternion cue. Do **not** train on a fixed image with a fabricated different TCP quaternion.

Budget this as one bounded data/label pass, at most two GPU-hours of identity fitting and four new complete-reset development checks. Continue only if it retains all four good saved handoffs, accepts corrected known port-0 images, rejects all known wrong/hidden requests, achieves at least 95% *reviewed clear* arrival recall, and has **zero absent-port all-view accepts on each held-out reset**. Then repeat fresh native-RGB shadow runs and complete ROS-to-first-insertion-command latency before attaching insertion. Stop this identity branch if absent-port acceptance recurs; the fallback is an explicit observation-derived board/rail landmark selector, evaluated under the same gate. Keep SFP handoff, actor RL and final scenes gated. Spatial image representations and multi-view consistency are established design ideas in [Transporter Networks](https://proceedings.mlr.press/v155/zeng21a.html), [S3K](https://proceedings.mlr.press/v155/vecerik21a.html) and [RVT-2](https://roboticsproceedings.org/rss20/p055.pdf), but those papers do not establish that this specific SC identity gate will pass.

## Commands, checks and artifacts

```bash
# Offline cause check and saved controls:
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/replay_sc_shadow_handoff_negatives.py \
  /home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v2/native_absent_development_p0_a \
  /home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v1/direct_rgb_replay_llb_030{0,3,4,9} \
  --all-inspections --diagnose-identity-failure --output <new-replay.json> --device cuda:1

# Live known port-0 correction; change the scene root/case for new reset.
AIC_SC_SHADOW_DIRECT_NATIVE_RGB=1 AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST=1 \
 AIC_SC_SHADOW_ALLOW_STALE_PAUSE=1 AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS=1 \
 AIC_SC_SHADOW_SCENE_ROOT=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1 \
 AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=100 AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC=260 \
 bash scripts/run_sc_shadow_handoff_pilot.sh \
 /home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/p0_known_stale_orientation \
 development_p0_a

# Bounded identity fit (original head and locator remain frozen references):
prior=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1
crop=/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/train_sc_spatial_identity_probe.py \
 --two-port-labels outputs/experiments/2026-10-02_two_family_perception/projection_labels/rows.jsonl \
 --one-port-train "$prior"/train_*/projection_state_v2/rows.jsonl \
 --one-port-development "$prior"/development_*/projection_state_v2/rows.jsonl \
    "$crop"/crop_development_*/projection_state_v2/rows.jsonl \
 --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
 --output /home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/identity_no_tcp_orientation_v1 \
 --spatial-only --mask-tcp-orientation --updates 800 --device cuda:3
```

The [NAS archive](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/) contains launcher snapshots, scene manifest, rootless Gazebo captures, scored streams, per-inspection traces, RGB review montages, state-swap results, the failed identity checkpoint and both comparison JSONs. The [tracked machine summary](2026-10-03-sc-orientation-identity-repair.summary.json) links the decisive results. `python -m py_compile`, `bash -n`, JSON consistency and unique reset-hash checks were run. No active experiment jobs were left running.
