# Phased Full-Insertion / Ablation Plan - 2026-05-30 v3

## Job State

Host and container process checks found no active AIC/Isaac insertion training or rollout jobs to stop. The only
long-running GPU jobs visible on the host were unrelated `protomotions` jobs outside this repository, so they were left
untouched.

## Current Evidence

Strict full insertion is still unachieved. The current best evidence says reward/architecture training is secondary to
reset/contact correctness:

- v699 no-guard randomized training reached shallow positive tip depth only: max `s=1.369 mm`, min theta `0.0160 rad`,
  but no module-consistent full insertion.
- v704/v705 ConvNeXt/history branch tested stronger critic vision and temporal state history. v705 improved shallow
  positive tip depth to max `s=1.867 mm` and theta min `0.0131 rad`, but still no strict insertion.
- v706 continuation was incomplete/weak and did not improve the v705 shallow result.
- The latest collision work shows correctly registered NIC cage aligned-cubes recover module consistency near full
  depth, but strict success still fails on marginal `r` and semantic-tip theta: v813 best `s=45.941 mm`, `r=0.505 mm`,
  theta `0.03686 rad`, consistency `0.960`, strict false.

## Reset / Randomization Audit

The v701 offset-fixed curriculum remains the current training substrate:

```text
outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v701_offsetfix_from_v642
```

Static audit:

```text
outputs/agentic_reward_curriculum_20260529/reset_audits/audit_v702_v701_offsetfix_train_mixed_static_distribution
```

Key finding: all 24 train-mixed episodes reset `gripper_tcp` while the semantic reward body is `sfp_tip_link`. That is
expected for the current wrist-IK path, but only valid if the reset-body-to-tip calibration is preserved. The current
generator records a constant calibrated reset offset of about `59.55 mm`; it no longer silently treats gripper and tip
as the same body.

Remaining risk: `variant_tip_preserving_rotation` is false for all train-mixed episodes, so orientation perturbations
are not yet guaranteed to preserve the realized semantic tip after settle. Long runs should not be promoted until
post-step reset validation confirms `s/r/theta/module` are in the intended bucket.

## Literature Priors

- SERL emphasizes that sample-efficient robotic RL depends heavily on controllers, resets, reward computation, and
  implementation details; it reports 25-50 minute learning windows on tasks including PCB assembly and cable routing
  when those pieces are engineered correctly. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL reports precise manipulation policies trained in about 1-2.5 hours using demonstrations/corrections plus
  efficient off-policy RL. For this task, the analogue is a trusted Isaac teacher or accepted trajectory replay before
  online SERL, not blind reward-only training. Source: https://arxiv.org/abs/2410.21845
- RL-VLM-F is relevant as a reward/feedback idea, but its feedback signal is visual/task-semantic. For this project it
  would be useful only as an auxiliary visual sanity scorer after geometric strict checks; it cannot replace
  post-step `s/r/theta/module` metrics. Source: https://arxiv.org/abs/2402.03681
- Vision pretraining evidence supports testing stronger features, but not assuming global features solve contact: a
  manipulation PTM study reports that local visual features are important and that DINOv2 can outperform robot-specific
  pretraining in some policy settings. Source: https://openreview.net/forum?id=9GKMCecZ7c
- Eureka-style reward synthesis is relevant only with strict geometric and visual acceptance gates; reward return alone
  is already known to be unsafe for this insertion task. Source: https://arxiv.org/abs/2310.12931

Literature review refresh on 2026-05-30 used the public paper/project pages above. The actionable conclusion is unchanged:
try reward/architecture variants only as bounded ablations, and prioritize controller/reset/contact validity before
HIL-SERL. SERL/HIL-SERL both assume a high-quality controller/data pathway; the current Isaac pathway cannot yet produce
post-step stable module-following full-depth trajectories.

## Seven-Hour Work Packages

Each package has a hard 7 hour wall-clock cap. Stop earlier when strict metrics reject the path.

| package | cap | work | promotion gate |
|---|---:|---|---|
| Reset/contact fixes | 7h | eliminate gripper-tip reset drift; audit orientation perturbations; continue NIC cage clearance/orientation probes | accepted post-step reset buckets and improved v813-like full-depth strict gap |
| Reward/curriculum | 7h | phase-conditioned reward schedule, module-depth loss/progress, theta retention after positive `s`, safe axial gates | improves `s`, `r`, theta, and module consistency together; no tip-depth false positives |
| Architecture | 7h | ConvNeXt/ResNet18/DINOv2-feasible critic, actor/critic state history, ACT ResNet50 offline config | stable smoke plus held-out metric improvement, not reward-only gain |
| Literature-derived HIL/SERL | 7h | teacher/replay or correction-style dataset only from valid/near-valid trajectories; conservative online updates | module-following insertion improves without hard hidden success criteria |
| Evaluation/reporting | 7h | held-out 40/10, randomized near-gate, randomized shallow/final videos and summaries | strict success or blocker with exact closest gap |

## Generated Commands

Fresh architecture smoke command set, derived from v699:

```text
outputs/agentic_reward_curriculum_20260529/commands/architecture_v814/architecture_ablation_manifest.json
outputs/agentic_reward_curriculum_20260529/commands/architecture_v814/critic_convnext_hist4.sh
outputs/agentic_reward_curriculum_20260529/commands/architecture_v814/actorcritic_hist4_convnext.sh
outputs/agentic_reward_curriculum_20260529/commands/architecture_v814/critic_resnet18_imagenet_hist4.sh
```

These are 80-step, 2-env smokes. They preserve the v699 reward/reset/checker setup while varying critic vision and
state history. They are not success criteria; they are compatibility and signal checks.

## Immediate Queue

1. Run one low-cost architecture smoke from the v814 set only after confirming no AIC jobs are active.
2. Prefer `critic_resnet18_imagenet_hist4` first because it is cheaper than ConvNeXt and tests the history/vision path.
3. Reject the branch if it only increases reward or tip `s` while module consistency, `r`, or theta regress.
4. In parallel with short smokes, continue the reset/contact path: target the v813 residual gap (`r≈0.505 mm`,
   theta `≈0.0369 rad`) with semantic-tip-preserving orientation reset diagnostics and exact cage-clearance probes.
5. Do not start long HIL-SERL until an Isaac teacher or reset trajectory can produce module-following insertion
   evidence beyond shallow positive depth.

## v814 Architecture Smoke Result

Run:

```text
outputs/agentic_reward_curriculum_20260529/architecture_ablation_runs_v814/2026-05-30_19-09-04_arch_critic_resnet18_imagenet_hist4_smoke
```

Decision: reject for promotion.

The 80-step, 2-env ResNet18 ImageNet critic/history smoke completed and wrote checkpoints, but did not improve the
insertion bottleneck. `cheatcode_phase_summary.json` reported max tip `s=3.565 mm`, mean `r=0.533 mm`, mean
theta `0.0223 rad`, `first_step_success_candidate=null`, and `axial_progress_positive_count=0`. Direct metrics parsing
showed mean post-step max tip `s=2.358 mm`, module max `s=-21.279 mm`, best-score row at step 80 with tip
`s=2.358 mm`, `r=0.409 mm`, theta `0.0197 rad`, module `s=-21.279 mm`, module `r=1.135 mm`, and consistency axial
error about `20.1 mm`.

This is a shallow tip-only improvement with substantial module lag, not a teacher trajectory and not a reason to run a
long architecture continuation. The next priority is the Gazebo-to-Isaac cheatcode transfer audit/fix path so the
teacher can generate module-following full-depth trajectories before HIL-SERL.

## v826 ConvNeXt Critic/History Smoke Result

Run was redirected to host/container scratch because `/data1` is nearly full:

```text
/tmp/aic_agentic_reward_curriculum_20260529/architecture_ablation_runs_v826/2026-05-30_19-42-31_arch_v826_critic_convnext_hist4_smoke_tmp
```

The command was the v814 `critic_convnext_hist4` smoke with output path changed to `/tmp`, checkpoint save intervals
raised to avoid periodic large writes, and image logging reduced. It completed 80 steps / 2 envs.

Decision: reject for promotion.

Strict success count was zero. Best post-step tip row was env0 step 80: tip `s=4.030 mm`, `r=0.280 mm`,
theta `0.0231 rad`, consistency gate `0.412`, module `s=-19.600 mm`, module `r=1.131 mm`. The best strict-like score
row was env0 step 40: tip `s=3.007 mm`, `r=0.126 mm`, theta `0.0213 rad`, consistency `0.611`, module
`s=-20.625 mm`, module `r=0.842 mm`. Final env1 was also shallow: tip `s=1.149 mm`, module `s=-22.492 mm`.

This confirms the same pattern as v814: stronger critic vision and temporal critic history can produce small shallow
tip progress and good theta, but it does not create module-following insertion. Architecture remains a secondary branch
until the Isaac teacher/contact path can produce deeper module-consistent trajectories.

## v850/v852 Tip-Preserving Randomized Curriculum

The previous v701 randomized curriculum still carried a reset bug risk: nonzero orientation perturbations were not
semantic-tip preserving even though resets were commanded on `gripper_tcp` and rewards/checks used `sfp_tip_link`.
I patched `build_randomized_near_gate_curriculum.py` to reconstruct the measured reset-body-to-tip vector from the
`lowtheta_metric_reset` metadata and apply rotation perturbations around the semantic tip. A static audit of the new
v850 train-mixed set confirms all sampled variants now have `tip_preserving_rotation=true`.

Validation run v852 accepted the first 8 sampled train-mixed episodes:

```text
outputs/agentic_reward_curriculum_20260529/reset_settle_validation/2026-05-30_20-53-17_v852_v850_tip_preserving_train_mixed_8env_zeroaction_repoout/accepted_episodes
```

Step-2 reset distribution was plausible for short smoke training: tip `s=-19.90..+1.17 mm`, tip
`r=0.265..2.532 mm`, theta `0.0162..0.0275 rad`, module `s=-43.54..-22.47 mm`, and module
`r=0.157..1.791 mm`. These are not success states; they are accepted randomized starts for a no-guard smoke because
theta is now within the strict orientation region while axial/lateral buckets remain varied.

## v855/v856 No-Guard Randomized Smoke Results

Compact artifacts:

```text
outputs/agentic_reward_curriculum_20260529/policy_train_runs_compact/v855
outputs/agentic_reward_curriculum_20260529/policy_train_runs_compact/v856
```

| run | change | strict | best/final post-step result | decision |
|---|---|---:|---|---|
| v855 | v850/v852 accepted randomized starts, no guard, ConvNeXt/history from v706, 80 steps | 0 | step 80 tip `s=-0.063..3.604 mm`, `r=0.194..0.400 mm`, theta `0.0162..0.0225 rad`; module `s=-23.696..-20.030 mm`, consistency final axial error `42.2..45.9 mm` | partial signal only |
| v856 | continue v855 with larger translation clip, stronger axial/module reward/loss, 160 steps | 0 | step 160 tip `s=0.049..5.833 mm`, `r=0.270..0.439 mm`, theta `0.0167..0.0352 rad`; module `s=-23.579..-17.798 mm`, consistency gate worsened to mean `0.434`, force mean `11.18 N` | reject |

v855 is useful evidence that policy-only actions can keep lateral/orientation tight under the tip-preserving randomized
curriculum. It is not insertion: the module remains roughly `40+ mm` axially inconsistent with full seating. v856
shows that simply increasing axial authority and module reward pressure is not the fix; it produced slightly more
shallow tip depth but destabilized reward/Q, worsened module consistency, and exceeded the strict theta threshold in
one env.

Storage note: Docker temp output filled `/data1` while saving v856. The invalid partial
`checkpoint_latest.pt.tmp.*` files were removed. Rejected smoke checkpoints were preserved off `/data1` under
`/tmp/aic_checkpoint_archive/agentic_reward_curriculum_20260529/`, and one older large replay artifact was moved to
`/tmp/aic_output_archive/agentic_reward_curriculum_20260529/replay/` with a symlink left at the original repo path.

Current decision: do not continue this exact no-guard reward-only branch with larger axial clips. The next useful
branch should either generate valid module-following teacher trajectories from the cheatcode/contact path or add an
imitation-heavy/HIL-SERL source. Reward-only tuning can still be used for short probes, but promotion requires module
`s` to advance with tip `s`, not only shallow tip-depth improvement.

## v857-v861 Teacher/Contact Follow-Up

I ran the next bounded teacher/contact probes rather than continuing reward-only training. These used the v846/v843
close-balanced module-body teacher setup as the baseline.

| run | result | decision |
|---|---|---|
| v857 | failed before rollout because cameras were spawned without `--enable_cameras` | infrastructure failure; rerun as v858 |
| v858 | stricter orientation trim reached only `s=40.12 mm`, theta `0.03627 rad`, consistency `0.004`; later swept laterally | reject |
| v859 | low-friction shell/cage material worsened best depth and consistency: `s=38.73 mm`, `r=0.666 mm`, consistency near zero | reject |
| v860 | tiny late trim improved the closest candidate: `s=45.96 mm`, `r=0.240 mm`, theta `0.03524 rad`, module `s=22.28 mm`, consistency `0.910` | closest near-success teacher candidate, not strict success |
| v861 | ultra-tiny lower-threshold trim failed to reproduce v860; best summary `s=35.05 mm`, `r=0.670 mm`, consistency near zero | reject |

Strict success is still false. The useful finding is that v860 preserves full-depth/module-consistent insertion while
moving theta closer to the strict threshold. The remaining gap is now specifically semantic-tip orientation:
approximately `0.0052 rad` above the existing `0.030 rad` threshold. The next bounded improvement should be a
module-consistency-aware final orientation correction or early-stop/trajectory capture around the v860 best frame, not
broader reward-only training.

## v862-v863 Module-Consistency-Gated Trim

I added an off-by-default `target_module_stabilize` diagnostic flag to gate orientation trim on current module
consistency:

```text
--target_module_stabilize_orientation_min_module_consistency
```

Validation passed with `py_compile` and the insertion/randomized curriculum tests (`32 passed`). The follow-up probes
did not beat v860:

| run | gate | result | decision |
|---|---:|---|---|
| v862 | `0.85` | best `s=46.39 mm`, `r=0.349 mm`, theta `0.03446 rad`, consistency `0.756`; trim was inactive at the best frame because pre-step consistency was below gate | reject |
| v863 | `0.65` | best summary `s=45.84 mm`, `r=0.273 mm`, theta `0.04120 rad`, consistency `0.880` | reject |

This narrows the blocker: preserving module consistency and lowering theta are coupled in the current wrist/contact
controller. A module-consistency gate prevents some destructive rotation, but it does not produce strict theta. v860
remains the best near-success teacher/contact state; strict success is still unproven.

## v864-v865 Near-Success Capture

I added an off-by-default early capture mode to preserve near-seated teacher trajectories before the diagnostic
retreats:

```text
--stop_on_near_success_capture
```

The capture thresholds are recorded in each run summary and do not alter strict success. Results:

| run | result | decision |
|---|---|---|
| v864 | new seed did not hit capture thresholds; best `s=45.79 mm`, `r=0.224 mm`, theta `0.03956`, consistency `0.867`; videos saved | reject for teacher promotion |
| v865 | exact v860-seed replay stopped at step `199`: `s=45.19 mm`, `r=0.240 mm`, theta `0.03521`, consistency `0.860`, module `s=21.55 mm`; center/left/right videos saved | accepted as near-success teacher artifact only |

v865 is the best preserved module-following teacher artifact so far, but not a success. It is suitable for
imitation-heavy/HIL-SERL as near-success data only if labels keep `strict_success=false` and `near_success_orientation_blocked`.
