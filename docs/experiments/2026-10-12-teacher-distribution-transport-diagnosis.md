# Teacher-generator distribution transport diagnosis, October 12

**Follow-up:** [saved-input feedback and inference-cadence diagnosis](2026-10-12-route-feedback-and-inference-cadence.md) completed the corrected codec tests, two bounded training ablations, and a four-step live pilot. No safe insertion or checkpoint promotion resulted.

## Question and frozen test

Can the route-consistent diffusion actor follow the same starts and board layouts on which its two teacher generators succeed? This check uses the [October 12 checkpoint](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt), with no retraining, guide, true port pose, or teacher actions during rollout. Ground-truth geometry is used only after the runs to measure the path. The four reserved final configurations remain sealed.

The frozen [plan](/home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1/plan.json) selects three held-out SFP validation starts from the actual varied-start CheatCode generator and three held-out SC validation starts from the Low Lateral Bypass (LLB) generator. The SC set includes one, four, and five cards and both requested ports. Every original teacher completed full insertion and scored no prohibited contact on its exact scene. The actor was run from the same scene config in rootless Gazebo, with only its observations and task code. SFP runs lasted 40 simulation seconds; SC runs lasted 70.

## Matched results

| Held-out scene | Teacher result | Actor closest lateral offset in axial −100 to −70 mm band | Actor result |
| --- | --- | ---: | --- |
| SFP trial 1 validation 04 | Full insertion, no contact | 66.74 mm | No buffer, no insertion, no contact |
| SFP trial 2 validation 12 | Full insertion, no contact | 44.77 mm | No buffer, no insertion, gripper/card 4 contact |
| SFP trial 2 validation 14 | Full insertion, no contact | 44.01 mm | No buffer, no insertion, no contact |
| SC LLB 0005, one card, port 1 | Full insertion, no contact | Did not enter band | No insertion, no scored contact |
| SC LLB 0074, five cards, port 0 | Full insertion, no contact | 325.84 mm | No insertion, no scored contact |
| SC LLB 0183, four cards, port 1 | Full insertion, no contact | Did not enter band | No insertion, no scored contact |

Sustained measured TCP departure of more than 20 mm from the time-matched teacher appeared at simulation time 4.20–4.45 s for SFP and 6.05–6.30 s for SC. The first SFP teacher and actor TCP poses at 3.05 s were within 0.4 mm, so the path mismatch developed after autonomous actions. For one SFP case, at 3.6 s the actor's first body-frame target was approximately `[0, −3.1, −3.8]` mm whereas the recorded teacher target was `[−3.5, −11.0, −7.6]` mm. The actor therefore moved too little along the route almost immediately. Offline replay of that exact held-out teacher episode produced first-target errors near 0.7–1.6 mm; good teacher-state imitation did not survive actor-visited observations.

The SC actor temporarily reduced lateral distance around simulation time 40 s, but stayed roughly 20–45 mm too far back axially and then retreated as the teacher continued toward insertion. Five-card side and overhead views show a different tool/cable route, with the actor ultimately farther from the card row. No scored contact occurred in the three SC actor runs, so these traces do **not** establish a cable snag. They show route and phase-progress errors. In the five-card case the actor entered the chosen axial band while still about 326 mm laterally away; “entered band” alone is not a useful success measure.

The old SC `lateral ≤10 mm` TCP buffer is also an invalid universal teacher success gate: only 11/21 held-out, fully inserted LLB teachers meet it. The selected successful teachers have closest lateral offsets 10.89, 4.21, and 10.38 mm in that band. Future SC transport gates must use a geometry and phase definition that all successful teachers satisfy, or compare directly to safe teacher route progress. Actor failure here does not depend on this gate: none completed insertion, and the measured paths separate substantially.

## Image preprocessing provenance correction

The SFP training cache reads JPEG-quality-86 images from the new teacher recorder before its common resize/JPEG-95 transform. Live SFP inference previously passed raw camera images directly to the common transform; SC already applied the quality-86 step. An opt-in `AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86` setting was added to the policy, but a later launcher audit found the variable was **recorded in the eval manifest and omitted from the Docker environment**. The initial purported codec runs below did **not** exercise that setting. The launcher has now been repaired and its container environment checked directly. Do not attribute changes in the old runs to JPEG encoding.

On the same three SFP scene configs and checkpoint, the mislabeled repeat changed closest lateral offsets from `66.74, 44.77, 44.01` mm to `37.50, 9.55, 57.53` mm. This is evidence of rollout variability, **not** evidence that JPEG matching helps. The old metrics are preserved as a repeatability diagnostic. All three still scored zero insertion.

The initial SFP camera timestamps match simulation time. An exact cached-row ablation of prior-command feedback changed the predicted first action modestly, which does not by itself explain the live shrinkage. Remaining hypotheses include visually shifted actor-visited states, temporal action execution/conditioning, and route/phase representation. These require a bounded causal test before further routine collection or retraining.

## Released official evaluation check

The local released qualification config at `artifacts/prod_cheatcode_audit/official_qualification/eval_config.yaml` is byte-identical to the current upstream [eval_config.yaml](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml), SHA-256 `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5`. The [qualification documentation](https://github.com/intrinsic-dev/aic/blob/main/docs/qualification_phase.md) describes one SFP plug into NIC or one SC plug into SC, while the other cable end remains free. The released qualification distribution is a useful public check; it is not proof of any future private final mix.

The existing frozen public-family comparison of this same checkpoint reached 0/3 SFP and 0/3 SC safe buffers. The SC board rotation in those scenes is outside the LLB teacher coverage. The [frozen three-scene SFP rerun plan](/home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1/plan.json) uses different exact configs from the teacher-generator test, but the same nonfinal qualification-family validation starts as the earlier public-family comparison. The SFP route results were:

| Released qualification-family SFP scene | First run | Unchanged-policy repeat; codec flag not forwarded | Result in repeat |
| --- | ---: | ---: | --- |
| Trial 1 validation 04 | 68.86 mm | 72.77 mm | No buffer, no contact |
| Trial 2 validation 12 | 25.33 mm | **5.20 mm** | Buffer reached, no contact |
| Trial 2 validation 13 | 10.97 mm, gripper/card contact | 44.82 mm | No buffer, no contact |

**Correction:** the right-hand column is an unchanged-policy repeat because the Docker launcher omitted `AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY`. The one buffer visit and two larger misses show sensitivity to repeated Gazebo rollout details. All three still scored zero full insertion. The unchanged SC checkpoint's existing public-family result remains 0/3; an SFP-only preprocessing fix cannot resolve SC board-rotation coverage. The [machine-readable public-family repeat scores](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/official_codec86_analysis.json) retain per-scene contacts, task code, and inference values. Single-GPU latency p95 ranged 299.5–305.4 ms, but Gazebo shared that GPU, so this is not a clean deployment latency benchmark.

**Decision at this stage:** no checkpoint or preprocessing variant is promoted. The next input and first-action parity audit is recorded in a follow-up document. SC also needs a teacher-calibrated transport criterion and rotated-board LLB coverage before public-family promotion.

## Reproduction and artifacts

```bash
.pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1 --gpu 0
.pixi/envs/default/bin/python scripts/analyze_teacher_distribution_route.py \
  --comparison /home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1 \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/teacher_distribution_analysis.json
# This command was originally run before the Docker launcher forwarded the
# codec flag. Its saved output is an unchanged-policy repeat, not a codec test.
AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86 \
  .pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_sfp_codec86_v1 --gpu 0
.pixi/envs/default/bin/python scripts/analyze_teacher_distribution_route.py \
  --comparison /home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_sfp_codec86_v1 \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/teacher_distribution_codec86_analysis.json
.pixi/envs/default/bin/python scripts/run_official_sfp_codec_check.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1 --gpu 0
.pixi/envs/default/bin/python scripts/score_official_anchor_route_dev.py \
  --run /home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1/rollouts/sfp_trial1_validation_04 \
  --run /home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1/rollouts/sfp_trial2_validation_12 \
  --run /home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1/rollouts/sfp_trial2_validation_13 \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/official_codec86_analysis.json
```

The plans, exact scene configs, bags, camera frames, runtime traces, and scoring reports are under the two NAS roots above. Machine-readable per-proposal geometry and aggregate scores are in [baseline](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/teacher_distribution_analysis.json) and [unchanged-policy repeat](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/teacher_distribution_codec86_analysis.json). Videos: [SFP teacher](/home/nas/chmin/aic_archives/2026-10-12_sfp_start_variants_v1/rollouts/sfp_trial2_validation_12/five_cameras_10fps.mp4), [SFP actor](/home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1/rollouts/sfp_trial2_validation_12/five_cameras_10fps.mp4), [SC five-card teacher](../../outputs/experiments/2026-09-30_sc_llb_collection/continuation170/runs/llb_0074/five_cameras_10fps.mp4), [SC five-card actor](/home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1/rollouts/llb_0074/five_cameras_10fps.mp4).

All single-GPU diagnostic latency measurements were affected by colocating Gazebo and the actor on GPU 0; other users occupied GPUs 3 and 4. Do not use these measurements for the 300 ms deployment gate.
