# Current experiment status

## Active September 18 work

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
and **5.32 mm**; their fresh-simulator development trials are running with
physical start checks. No reliable learned
insertion is established.
See [active ACT record](experiments/2026-09-18-act-all-verified.md).

The parallel [Dreamer-v4 pilot](experiments/2026-09-18-dreamer-proposal.md)
was approved and started at 15:52:56 UTC, with a separate 22:42:56 UTC deadline.
Its strict first comparison uses the original 60 aligned SFP training episodes
and 14 held-out episodes. Dreamer tokenizer training is active on GPUs 2–3;
a fresh ImageNet ACT baseline is training on GPU 4. An untrained Dreamer
inference path measured 28.49 ms p95 over 1,000 decisions including image
preprocessing and command conversion. Early tokenizer reconstructions blur
connector/port details; neither model has a new live success result yet.

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
