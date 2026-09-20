# Current experiment status

## TCP-delta commands and tracking: current decision

**Train on the clean expert's commanded TCP-frame delta pose**, relative to the
TCP pose in the same observation. This delta is a desired *controller pose
reference*, not the displacement the TCP must achieve before the next image.
For corrective recordings, use `teacher_target_pose` for policy supervision and
`executed_target_pose` for action-conditioned dynamics; they can differ. Do not
replace the command label with the difference between two measured TCP poses.
The latter includes controller lag and contact and is useful for dynamics or
path evaluation, not as a drop-in `MotionUpdate` target. See
[`CollectCorrectiveCheatCode.py`](../aic_example_policies/aic_example_policies/ros/CollectCorrectiveCheatCode.py)
and [dataset eligibility](DATASETS.md).

An audit of **23,252 consecutive 50 ms observation pairs** in the 74 verified
one-NIC, no-SC aligned/corrective SFP episodes found these median translation
magnitudes: **9.83 mm** from the observed TCP to the *executed command target*,
**0.64 mm** actual TCP displacement during the next 50 ms, and **9.35 mm**
remaining from the next measured TCP pose to that same target. These are
separate medians, so they need not subtract exactly. In the last three recorded
seconds, the corresponding medians were 5.35 / 0.04 / 5.36 mm. The clean
teacher delta was also 9.83 mm median across all 32,183 native observations.
This is a target-tracking gap, **not** a 9.35 mm expert-label or model-prediction
error. The consecutive-pair audit used the saved `states.npy`,
`executed_physical.npy`, `teacher_physical.npy`, and each episode's native times
and command indices under the [Dreamer pilot data](../outputs/experiments/2026-09-18_dreamer60_pilot/artifacts/data), retaining only 50 ms intervals with
consecutive command indices. Position-delta magnitude is unchanged when a TCP
delta is composed into an absolute `base_link` target.

These episodes all have official full-insertion Tier 3 scores. They represent a
restricted aligned scene without SC distractors, so they do not establish
general performance in cluttered settings. The wider historical CheatCode SFP
collection has **140/140 verified full insertions**; its action labels are much
cleaner than the successful agent/VLM recordings with missing Cartesian labels.
The CheatCode policy itself issues a smoothly changing target about every
50 ms, including slow insertion and a settling phase. Its success despite the
gap above argues **against** waiting for every commanded pose to be reached or
shrinking the learned delta to the next observed displacement.

The existing [AIC controller](aic_controller.md) already runs at **500 Hz**,
interpolates position and orientation references, and applies Cartesian
impedance control. Its tracking-error reset is a coarse stuck-target safeguard
(`min_translation_error: 0.2 m`, timeout 2 s in
[`aic_ros2_controllers.yaml`](../aic_bringup/config/aic_ros2_controllers.yaml));
it is not a millimetre-level waypoint gate. The synchronized policy
[`Observation`](policy.md) arrives at up to **20 Hz**. In `insert_cable()`, use a
fresh observation as the inference event, publish the bounded Cartesian
`MODE_POSITION` target at the 20 Hz cadence, and let the controller run its own
fast loop. A chunked model may infer once per four commands, but every delta
must retain its trained observation-reference convention. For an
observation-relative delta, compose it with that observation's TCP pose and
publish the resulting absolute target in `base_link`; sending the delta in
`gripper/tcp` lets the controller apply it from the *later* TCP pose at receipt.
The runtime already has this observation-relative transport path in
[`RunACTTorchScript.py`](../aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py).

**Do not add general tracking-aware action-chunk pacing by default.** A second
fast loop would duplicate the controller's interpolation, and a reach-before-
advance rule would distort these successful expert commands, especially in
contact. Keep the nominal command cadence and log measured TCP, reference
pose, force, command, and simulated time. Consider a guarded
alignment-to-insertion transition only if the learned policy shows excess
off-path motion or harmful contact relative to experts, using observable
signals rather than privileged geometry. Compare the same frozen policy with
and without that guard on matched simulator scenes before adopting it. The
expert tracking-gap statistics alone do not demonstrate that extra pacing
will improve insertion.

## September 18 work

At 18:14 UTC the action contract was corrected: the verified expert label is
a **TCP-frame delta relative to the recorded observation**, rather than an
absolute base-link target for the model to predict. The completed all-data ACT
and fresh ACT60 results below used absolute predictions and remain historical
baselines. The new strict 60/14 ACT run completed 6,000 updates at 18:55 UTC
on the 32,183 native observation-command pairs, predicting one TCP delta per
observation. Its fixed held-out rule selected update 6,000 (2.96 mm first
command and 2.63 mm final-three-second translation error); four fresh paired
development scenes ran from 18:56 to 19:09 UTC: all four valid, **0/4 full
insertions**, mean official score **27.73**. Three ended 5 cm from the port;
one drifted to 33 cm. These imitation errors are not live insertion scores.
The corrected world dynamics completed 3,191 updates and stopped early after
its fixed held-out prediction gates failed; the selected update-1,000 model's
one-step TCP error was 10.33 mm versus 1.85 mm for persistence (14 episodes).
Here, one step means **200 ms and four actually executed 20 Hz commands**.
Starting from a recorded observation, the dynamics model predicts the TCP's
*measured position after those commands*. The 10.33 mm is the mean distance
between that predicted future position and the recorded future position in
`base_link`; it is not the error between a controller command target and the
robot. The 1.85 mm persistence baseline instead predicts that the future TCP
will remain at its initial measured position, then compares that unchanged
position with the same recorded future position. It works well over this short
interval because the TCP usually moves little, even when the commanded target
is farther away. See the [held-out dynamics evaluator](../outputs/experiments/2026-09-18_dreamer60_pilot/world_tcp_delta_final_runtime_archive/sources/dreamer_source/dreamer4/aic/evaluate_world.py).
A fresh supervised policy using corrected delta labels ran from 18:46 to
19:20 UTC on GPUs 2–3, stopping at update 4,509 after a documented held-out
plateau; fixed selection chose saved update 4,000 (2.16 mm combined
first-command error). The first live world startup found a camera-size
mismatch before any command. After matching the collector's resize to
288×256, trained raw-camera inference measured **32.76 ms p95** over 1,000
calls and the four-command callback loop **33.79 ms p95** in isolation.
The same-GPU live diagnostic measured **464.67 ms p95** across 168 published
commands, above the 300 ms requirement. Separating renderer and policy GPUs
alone did not fix this. A private inference worker preserved byte-identical
commands on verified native samples and passed a full live-scene diagnostic.
The fresh world development set was **4/4 valid, 1/4 full insertions**, mean
official score **52.14**, with pooled **81.00 ms p95** live decision latency.
The frozen paired final assessment completed all **20/20 eligible scenes per
policy**: ACT **0/20 full insertions**, one official partial, mean total
**22.69**; world **0/20 full insertions**, two official partials, mean total
**32.32**. The world policy's 8,908 live decisions measured **77.01 ms p95**,
under the 300 ms requirement. A failed ACT startup before trial 17's scored
rollout was preserved, then that same scene completed as an unchanged-model
retry. The previous absolute-action dynamics and BC runs are superseded.
The [follow-up](experiments/2026-09-18-world-followup.md) found the world
policy closer to the opening in more scenes but with persistent lateral error
and a large requested/measured TCP gap. Across all strict held-out 200 ms
dynamics windows, future measured TCP error was **12.06 mm** versus **2.29 mm**
for persistence; near the actual opening it was **13.94 mm** versus **0.57 mm**.
The 289-episode visual audit and matched tokenizer comparison improved SC
gross reconstruction with expanded data, but fine connector/port features
remained blurred. Reward and imagination training remain disabled. See the
[delta correction record](experiments/2026-09-18-tcp-delta-correction.md) and
[contract audit](../outputs/experiments/2026-09-18_dreamer60_pilot/act60_delta_contract_audit.json).

The [bounded supervised initialization comparison](experiments/2026-09-18-world-supervised-init-ablation.md)
trained the same six-view control architecture for 2,500 BC updates in each
arm and evaluated four fresh paired scenes disjoint from all 289 verified
episodes and the final set. Held-out first-command error was 2.285 mm with
selected world weights versus 2.578 mm from a fresh world trunk. Live mean
official score was 32.85 versus 26.24, with **zero full insertions in either
arm** and one partial for the selected-world arm. Both met the 300 ms limit:
pooled p95 command latency was 76.26/79.69 ms. The fresh arm's world weights
were a separate random draw rather than the exact pretraining ancestor, so
one seed cannot establish a causal pretraining advantage. The reward and
imagination gates remain closed.

The [September 19 full verified-data run](experiments/2026-09-19-full-world-training.md)
is complete. A fresh six-view tokenizer optimized all 250 canonical training
episodes, held out the 39 scene-disjoint validation episodes, and stopped at
78,000 updates under the declared validation plateau rule; update 75,000 was
selected. On 156 held-out frames it reduced whole-image/contact-field MSE by
91.25%/93.29% and contact edge L1 by 53.09% relative to the earlier bounded
reference. Corrected dynamics also converged by rule, but future TCP error was
11.46/19.17/20.12 mm at 200/400/600 ms versus 2.07/4.42/7.24 mm for persistence.
Every dynamics gate failed, so reward and imagination remain disabled.

The supervised full-data world policy selected update 1,000 and then completed
all **20/20 eligible sealed final scenes: 2 full insertions, 9 official partials,
mean total 45.14**. It beat the corrected60 ACT score on 13/20 paired scenes;
ACT remained 0 full / 1 partial with mean 22.69. Across 8,895 live decisions,
the new policy measured **80.77 ms p95**, 98.29 ms p99, and 146.74 ms maximum,
with no 300 ms misses. This is the strongest learned result in that paired set,
but 2/20 is not reliable and the complete-pipeline comparison does not isolate
which training change caused it. Next, run the full-data same-architecture
supervised comparison from the exact preserved random ancestor versus selected
world weights on new development scenes and multiple seeds. Keep the sealed
final scenes out of selection.

For size context, ACT has **16,354,566 inference parameters** (65,418,264
FP32 bytes), while the selected world policy has **22,569,344 acting
parameters** (90,277,376 FP32 bytes) and 27,130,942 parameters in the complete
training model. Thus the acting world policy is 1.38 times ACT's size. On their
matched corrected 60/14 data and frozen 20 scenes, ACT achieved 0 full / 1
partial insertion with a 22.69 mean score; the world policy achieved 0 full / 2
partial with a 32.32 mean. Neither met the reliability goal.

The user approved a new **6h50m ACT window, 11:51:28–18:41:28 UTC**, on physical
GPUs **0–1**. The canonical `expert_verified` collection now contains **289**
eligible episodes: 268 SFP across all five NIC counts and 21 newly collected
SC, with a grouped **250 train / 39 validation** split. The time-input
comparison used immutable v2 (260 episodes; 229/31). A matched visual-grid
comparison started on all of v3 at **15:18:10 UTC**, including its 29 additions.
The 253 successful agent recordings remain pending action-label repair;
212 non-insertions and 63 unresolved historical episodes are excluded.
Task conditioning and DDP passed GPU checks. The first SFP stage completed
6,000 updates in 53m33s at its requested cap. Mixed SFP/SC training started
13:16:39 UTC with batch 128 per GPU and balanced task/card-count sampling,
then stopped deliberately at 6,001 updates after 48m33s. This freed time for
a matched comparison of retaining versus masking elapsed time; it was not
stopped by the overall deadline or claimed to have fully converged.
SC collection accepted 21 of 36 attempts across two batches, all full insertions with
verified labels and terminal images, within a restricted scene distribution.
Historical SC command labels cannot be faithfully repaired. The initial
2,000-update ACT checkpoint completed five fresh development scenes with
**0/5 insertions**, mean score 31.71. A measured entrance stall was 7.91 mm
off axis. Step-4,000 and step-6,000 checks on two development scenes also had
zero insertions. The first mixed-task checkpoint completed **0/5 SFP and 0/4
SC insertions**; SFP NIC1 regressed while other SFP counts improved approach
scores. The time-input pair started at 14:06:35 UTC and completed 2,000 updates per
branch from the same mixed-task step-6,000 parent and immutable dataset.
Elapsed time is strongly associated with task identity in that dataset;
the first nine-scene evaluations gave **1/9 insertions for retained time**
(SFP 0/5, SC 1/4; mean 41.48) versus **0/9 for masked time** (mean 27.22).
The clean SC port-1 success scored 86.71. Those runs used a wall-time cap;
two masked SFP cases were still moving at the end. At 90 simulated seconds,
retained time completed **0/9**, mean **35.57**; its earlier SC success did not
repeat. Masked time had zero insertions and one shortened SC trial with contact
and force penalties. A separate full-duration retry also failed. Its supplementary
eight-valid-plus-retry comparison scored **23.22**, but does not erase the
original adverse outcome. Subsequent review found grossly displaced pre-command
robot starts in two masked SC trials, including the shortened one; the composite
is therefore not a clean policy comparison. The engine readiness source now
requires finite named arm joints near the configured home pose; it passed an
isolated rootless build/test but has not been installed into the pinned
evaluation image. Current trials restart the simulator for each scene and audit
the pre-command arm joints and first camera frames independently.
Retained time remains the parent for matched stride-32/stride-16 continuations.
Both completed 3,000 updates on v3, with held-out command errors **5.60 mm**
and **5.32 mm**; their fresh-simulator development trials completed with
physical start checks. No reliable learned insertion is established.
Their completed nine-scene fresh-simulator development results were
**0/9 insertions** each, mean score **33.27** for stride 32 and **24.44**
for stride 16. The earlier parent was also **0/9**, mean **23.03**.
The prespecified development rule selected stride 32; its frozen 32-scene
final assessment ran on GPUs 0 and 1 from 17:03 to 18:03 UTC. All **32/32**
trials passed the fresh-simulator initial-state and duration checks and were
officially scored. The policy achieved **0/32 full insertions** (SFP **0/20**,
SC **0/12**), mean official total **25.70** (SFP 21.04, SC 33.48). Two SC
trials were scored as partial insertions; nine trials incurred a prohibited
contact penalty. All 32 one-frame-per-second videos and start/end contact
sheets are archived with the selected model. The target was not met; no
offline or online SERL was trained in this window. See the
[ACT record](experiments/2026-09-18-act-all-verified.md) and
[frozen final report](../outputs/experiments/2026-09-18_act_all_verified_6h50/selected_act_final_single/final_single_results.md).

The parallel [Dreamer-v4 pilot](experiments/2026-09-18-dreamer-proposal.md)
was approved and started at 15:52:56 UTC, with a separate 22:42:56 UTC deadline.
Its strict first comparison uses the original 60 aligned SFP training episodes
and 14 held-out episodes. A fresh ImageNet ACT baseline completed 6,000 updates
on GPU 4; its four paired development scenes scored **0/4 insertions**.
Dreamer's first tokenizer stage stopped at 5,981 updates at its 50-minute cap;
two bounded detail refinements completed 2,500 updates each on GPUs 2–3, but
fine contact details remained blurred. A selected six-view tokenizer was frozen
for a supervised diagnostic. Its action-conditioned dynamics stage stopped
deliberately after 6,344 updates at 18:11:50 UTC; it beat persistence at
four/eight steps on average but failed the one-step and near-contact gates.
The held-out selected checkpoint was step 1,500. The superseded supervised BC
started at 18:12:54 UTC and stopped at 18:14:32 UTC when the action contract
was corrected. An untrained Dreamer inference path measured
28.49 ms p95 over 1,000 decisions including image
preprocessing and command conversion. Early tokenizer reconstructions blur
connector/port details. An audit of all 74 expert bags found the official
insertion event after the final saved observation in every episode, so the
strict pilot's insertion-state reward/imagination gate fails. The corrected
TCP-delta pilot's current training and live results are summarized at the top
of this page.

## Completed September 17–18 experiment

Updated: **2026-09-18**, branch `feat/hybrid-train`, base `7534090` plus
working-tree repairs and the direct visual actor. Evidence now includes CPU
regressions, the earlier 40-update BC smoke, **live Gazebo policy/control trials,
Isaac reset/terminal probes, a dataset audit, and new supervised ACT training**.
The earlier live-validation pass used at most two GPUs and did no training.
The eight-hour ACT experiment completed training and final evaluation using
physical GPUs 0–3, within its **2026-09-18 05:12:28 UTC** deadline.
[ACT experiment results](experiments/2026-09-17-act-verified-8h.md) ·
[Earlier results and videos](experiments/2026-09-17-live-validation.md) ·
[Initial audit](experiments/2026-09-17-reentry-audit.md) ·
[Accounting repairs](experiments/2026-09-17-evaluation-curriculum-fixes.md) ·
[Actor implementation](experiments/2026-09-17-direct-visual-policy.md).

## Where we stand

There is **no verified reliable general learned insertion policy in the evidence reviewed**.
The repository has usable training and evaluation machinery, ACT checkpoints,
many failed or partial RL experiments, and privileged local insertion diagnostics.
These have different evidential value; see the [experiment ledger](EXPERIMENTS.md).

The verified-data ACT experiment finished with **1/20 full insertions** on
fresh SFP NIC-1 / card-0 / port-1 / rail-0 scenes (mean total **39.16**).
All 20 evaluations completed normally with ground truth disabled. The **18/20
reliability target was not met**. The selected model had 2/10 development
insertions; its checkpoint, normalizer, runtime, lineage and exact scene set are
preserved in `outputs/experiments/2026-09-17_act_verified_8h/selected_act_final/`.
See the [report and videos](experiments/2026-09-17-act-verified-8h.md).

Historical RGB/BGR and wall/simulation-clock differences were identified; new
collections record aligned RGB observations and teacher commands. Quaternion,
command transport and warm-start normalization contracts were checked/repaired.
At the end of that earlier run, its newest cache held **87 recent episodes
(70 train / 17 validation)**;
the selected continuation used 60 train / 14 recent validation episodes from the
74-recording predecessor. There were 47 bounded ACT jobs, including small fitting
diagnostics and continuations. **No offline or online SERL updates ran.**

Training-command errors fell below 1 mm while recent held-out errors remained
around 4–5 mm. On one final failure, the tip was at the port entrance but
12.6 mm off axis, while the policy continued requesting descent. A separately
labeled privileged expert control on that same scene scored 94.68
(Tier 3 75); it is never counted as learned-policy success.

The latest located long run started June 14 and ended June 19. Its recovered
events show **200 evaluation cycles, zero successful evaluations, and zero
promotions**, ending at level 0. The recorded `episodes_used=2000` is a wrapper
budget counter: the old wrapper added 10 after every timed training segment, without
counting completed simulator episodes. Do not report this as 2,000 measured
episodes. The July handoff note remains useful historical context.

The ACT 175k CPU/CUDA exports, associated pretrained checkpoint/normalizer, and
historically named “clean” dataset are present locally. The new audit found
action-label and split-quality concerns in that dataset. Their presence establishes availability.
The inspected May 13 Gazebo evaluation has `policy_ready=false` and no score.
The May submission bundle's referenced model/evaluation directories are missing
from this checkout; its reported scores are historical, not reverified.

## Architecture direction

The current requested sequence is **ACT → offline SERL → online SERL**. The
completed experiment trained and evaluated ACT only, starting with verified
CheatCode demonstrations, then aligned expert and corrective collections.
Episode holdouts and a frozen 20-scene assessment were used. No offline or
online SERL updates have been run in this pass.

The [June 17 tutorial notes](https://github.com/yoonjung0705/tutorials/blob/master/python/libraries/tutorial_aic.md#617-2026-hybrid-train-branch-model-training-especially-for-isaac)
call for dropping the residual adapter, giving the actor direct visual features,
considering DINOv2, and reconsidering whether ACT warm starting is needed. They
also report an expert rollout withdrawing when initialized already inserted.
The two CheatCode controllers were found to request a 20 cm approach pose even
from such starts; they now preserve depth when an aligned final-descent gate is
met. The change has CPU coverage and still needs live verification.

The source still contains several distinct actors:

| Path | What the code does |
| --- | --- |
| **New `actor_mode=direct_visual`** | Shared image/state actor for offline learning, Isaac loading/training, and Gazebo inference/adaptation. Predicts the full command from visual features; no ACT action proposal or residual. Default for the offline CLI. |
| Direct actor backbones | Small conv, ResNet18, ImageNet ResNet18, and DINOv2 ViT-S/14. Optional ACT initialization copies visual weights only. Backbone parameters train by default. |
| Offline `ACTAdapterSERLActor` | Predicts a residual from encoded state plus ACT's proposed action chunk; the residual head does not receive image features directly. |
| Offline `actor_mode=act_direct` | Uses `ACTChunkActor`; this is a different implementation from Isaac's identically named mode. |
| Isaac `IsaacACTAdapterActor`, `act_direct` | Predicts the executed action without adding ACT's action, but still feeds state plus ACT's proposed actions to the trainable head. This retains the visual information bottleneck described in the notes. |
| Vision critics | The offline encoder supports small convolutional networks, ResNet18 variants, and ConvNeXt Tiny variants. DINOv2 is not implemented in this encoder. A visual critic does not give the actor direct visual features. |

The legacy actor modes remain for old checkpoints and recipes. Use the
[direct visual workflow](DIRECT_VISUAL_POLICY.md) for new architecture work.
The new actor completed 40 BC updates on saved demonstrations; a pretrained
DINOv2 forward/backward check also passed. These are implementation/learning
checks. The subsequent direct actor and ACT Gazebo trials both completed with
no insertion. Fix reset/control and dataset problems before comparing encoders.

## Repairs completed on September 17

| Area | Current behavior |
| --- | --- |
| Gazebo score interpretation | Correct insertion requires each trial's official Tier 3 score of 75. Tier 1 remains model validity. The parser exposes per-trial outcomes and counts; transfer validation requires complete scoring and all observed trials to succeed. |
| ACT runtime evaluation | Explicit command mode is required. Empty checkpoint selections and incomplete runs fail. Matching complete evaluations may be reused; failed, legacy, or changed-setting evaluations get fresh attempt directories. Completion requires readiness, clean engine exit, and all configured trials scored. |
| Episode accounting | Isaac writes one outcome per terminated/truncated environment episode, using pre-reset identity and terminal success flags. The wrapper counts these completed episodes. A checkpoint left by a failed process does not permit continuation. |
| Promotion and demotion | The wrapper enforces the configured success rate and consecutive-failure threshold. Evaluation must complete the fixed per-level episode set without updates, exploration, guide, or guard overrides. Empty, partial, or legacy sampled logs cannot authorize promotion. |
| Evaluation settings | SFP module consistency is required; defaults are 0.5 mm axial/lateral error, 0.03 rad orientation, and 1.0/1.5 mm module axial/lateral error. Modified collision geometry is an explicit diagnostic option. Settings are saved with the result. |
| Stops and schema checks | `--updates 0` no longer ends evaluation after its first step. The no-promotion limit stops the wrapper with exit 3. Actor state width mismatch raises an error instead of silently padding/truncating input. |
| Artifacts | Stateful/axial launchers default to mounted `outputs/experiments/`, preserve cycle directories, and reject accidental reuse. Missing checkpoints fail explicitly. Other historical scripts still require path inspection. |

The [repair record](experiments/2026-09-17-evaluation-curriculum-fixes.md) lists
tests and compatibility changes. These repairs do not reclassify historical
results or establish that a policy inserts successfully. Old metrics lack the
terminal evidence needed to reconstruct reliable episode success rates.

## Latest live findings

Subsequent [source inventory](DATASETS.md) found the broader 668-episode local
collection and recovered both CheatCode sources previously absent at their old
manifest paths. All 140 accepted CheatCode trials have historical official
Tier 3 success records. The subsequent audit verified raw state/action equality,
sampled final images, and checked the 130-episode cache's video timestamps.
Across all sources, 393 episodes have verified score/array lineage, 212 are
scored non-insertions, and 63 mappings remain unresolved. See [data eligibility](DATASETS.md#verified-insertion-episodes).

The ACT result above supersedes the early pilot status. The table below records
the preceding live-validation pass; it is not the final ACT assessment.

| Check | Result |
| --- | --- |
| ACT 175k / direct visual actor | One official SFP trial each, both complete, neither inserted. Total scores 36.58 / 23.28; these are unequal training budgets and execution settings, not an architecture ranking. Tier 2 sampling anomalies also limit aggregate-score comparisons. |
| Gazebo control | Final bridge probe completed 260 commands with three-camera recordings. Initial 20 zero commands drifted 7.13 mm over 2.35 simulation seconds. Signed translation effects remain confounded by drift. |
| Isaac resets | Requested −40/−2/+43 mm aligned starts drifted and never met strict insertion criteria in a five-second zero-action probe. It completed zero episodes. |
| Isaac terminal accounting | Separate one-second probe completed exactly three episodes, recorded three timeouts, and stopped at the requested count. Zero successes. |
| Demonstrations | 51.74% zero actions; 189,885 zero-action frames have recorded TCP speed above 1 mm/s. 6.11% of all labels exceed the direct actor's translation limits. Current holdout is entirely SFP card 0 / port 1. |
| Runtime | Rootless Unix-socket bridge, image-sized IPC messages, local model imports, visible policy exceptions, and one-second recording now work in live checks. Isaac cameras require the documented container-local workaround on this old driver. |

See the [full report and videos](experiments/2026-09-17-live-validation.md) for
settings, measured geometry, failure records, and interpretation limits.

## Still unresolved

- Stable zero-action control, translation direction at measured simulation
  rates, and physically valid resets. The expert depth-preservation fix still
  needs a valid already-inserted reset and a live expert rollout.
- Demonstration command validity/time alignment, verified insertion labels,
  source-trial grouping, and balanced held-out scenes. Do not treat every zero
  command as corrupt or every historical scalar score as insertion success.
- Earlier policy trials had missing scorer world-frame connectivity. The new
  evaluator provides the fixed world transform while keeping object ground
  truth disabled for learned policies; old scores remain historical evidence.
- No verified reliable learned policy or controlled backbone performance comparison.
  Legacy Isaac `act_direct` still means the ACT-action-context head.
- Some historical model dependencies and container-local artifacts remain
  unbacked-up. See the [artifact map](../outputs_README.md).

## Next experiments, in order

1. **Resolve learned-policy alignment/contact failures.** Collect expert labels
   at the actual stalled/misaligned states, with synchronized observations and
   executed commands. Keep failed rollouts as failures. Measure lateral error,
   approach and official full insertion separately.
2. **Test visual precision under a matched budget.** Compare connector crops or
   multiscale features, then a backbone change, with the same full-action head,
   corrected data, held-out scenes and measured control rate. More updates alone
   did not close the current generalization gap.
3. **Keep Isaac reset/control validation separate.** The Gazebo expert's success
   does not establish valid Isaac resets or stable zero-action behavior. Pass
   that gate before transferring the policy or scaling online work.
4. **Resume ACT → offline SERL → online SERL in stages.** Use the direct visual
   full-command actor and trustworthy failure/success labels. Require a small
   fixed-scene evaluation gate before a large online budget. Use new held-out
   scenes; the completed final set is now a known test set.

Preserve actor-only evaluation for the June experiment lineage. Any privileged
guide or action override belongs to a separately labeled diagnostic experiment.
Report metrics for the same timestep/episode together: independent best depth,
lateral error, and orientation values cannot establish insertion.
