# Task-conditioned ACT on all eligible experts

Status: active; time-input comparison completed and matched stride continuations
started on all v3 episodes at 15:18:10 UTC. Initial SFP step-2,000
development result **0/5 insertions**; later two-scene checks remain **0/2**.

User confirmed a 6h50m total window covering preparation, training, evaluation,
and documentation. Start **2026-09-18 11:51:28 UTC**; hard deadline
**18:41:28 UTC**. ACT uses only physical GPUs **0 and 1**. A separately requested
Dreamer review is read-only pending approval; its potential GPU allocation is
2–4, making at most five GPUs across both efforts after that approval.

Artifacts: `outputs/experiments/2026-09-18_act_all_verified_6h50/`.

## Dataset and scope

See [DATASETS.md](../DATASETS.md#canonical-expert-collection-2026-09-18).
The first materialization contains all 251 eligible SFP episodes across NIC
counts 1–5; 223 train / 28 validation, with identical scene/task configurations
grouped. All 253 successful historical agent recordings remain pending action
repair. Their labels cannot currently support trustworthy full-trajectory BC.
The original collection was renamed and preserved. Nine newly verified SC
episodes brought v2 to **260 episodes**, split **229 train / 31 validation**.
The latest v3 adds 29 verified episodes and contains **289 (268 SFP + 21 SC)**,
split **250 train / 39 validation**. All earlier split assignments remain
unchanged. The time-input comparison used v2; the stride continuations use v3.

## Implementation and intended training

- Reuse the canonical 10D task vector. Input: 32 robot values + 10 task bits +
  elapsed time = 43 values. Task bits use identity normalization at training,
  export, and inference. Invalid metadata and overlapping episode splits fail.
- Fresh ImageNet ResNet18 initialization; no old ACT policy warm start for the
  initial fit. Chunk 8, execute 4, full absolute TCP targets, X-positive
  quaternion convention, controller-error inputs masked, dropout 0, terminal
  sampling probability 0.5 over the last eight seconds, color jitter 0.03.
- Two-process DDP with independent rank batches, BF16, rank-zero validation and
  checkpoint writing, and coordinated signal/deadline handling. Batch sizes
  are measured per GPU. Benchmark useful global throughput and memory before
  selecting a sustained run; fitting in memory is not sufficient evidence of
  efficient GPU use.
- Include all eligible training episodes; validation remains separate. Report
  errors and official simulator insertion outcomes by task and NIC count.
- Stop optimization in time for frozen fresh-scene evaluation, video review,
  reporting, and owned-resource cleanup before the hard deadline.

## Initial checks

Trainer/data regressions: 10 passed, including actual ACT updates on two CPU
distributed ranks and propagation of a stop raised only on the second rank.
Runtime/export contract regressions: 30 passed, including 43D task/time slicing
and all ten SFP/two SC target identities. The first two-GPU batch-64-per-rank
check completed 30 updates with finite loss and a saved checkpoint; it is a
training-path check, not a useful policy result. Comparative throughput checks
completed. With three CPU camera workers, batch 128 per GPU achieved 539.7
frames/s and 2.11 updates/s, using 11.23 GB per GPU; warmed batch 384 achieved
581.3 frames/s but only 0.76 updates/s and used 33.03 GB per GPU. Chosen batch:
**128 per GPU / 256 global**. Measured GPU utilization was approximately
88%/91%; these short measurements exclude checkpoint and validation pauses
and do not establish policy quality.

The sustained initial SFP stage started **12:21:07 UTC** from fresh ImageNet
weights, using all 223 training episodes and 28 held-out episodes. It requests
6,000 updates, saves every 2,000, validates every 1,000, and uses learning rates
3e-5 (ACT) / 1e-5 (visual backbone), BF16 and three camera workers per rank.
It completed all 6,000 requested updates in **3,212.68 seconds** (53m33s),
processing 1,536,000 sampled image anchors. It ended at its planned stage cap,
not a saturation criterion or the overall time limit. Commands and logs:
`act_all_sfp_stage1_launch.json` and
`act_all_sfp_stage1.log`; throughput evidence: `benchmark_report.md`.

The new 43D model has **16,354,566 parameters**, all trainable, including
11,166,912 backbone parameters. FP32 parameter storage is 65,418,264 bytes;
the representative safetensors file is 65,526,376 bytes. The extra ten task
inputs add 5,120 parameters to the previous 33D model. The three cameras share
one ImageNet ResNet18. DINOv2 is not used in this ACT run. Exact measurements
are in `act_model_size.json`; training activation/optimizer memory is separate
from parameter storage.

Replaying the exact deterministic distributed sampler confirms all 223
training episodes are sampled by update 5. By update 1,000 it has visited
73.9% of training image anchors and 99.88% of command labels through the
eight-action chunks; by update 5,000, image-anchor coverage reaches 99.54%.
This is sampling coverage, not a convergence measurement.

## Initial SC expert collection

Collection finished with **9 accepted / 20 attempted**; eleven failed attempts
remain excluded and five planned scenes were not attempted. All nine reached
official Tier 3 **75**, totals **84.86–91.48**, with no prohibited contacts or
force penalties. Terminal camera views were inspected. Their 5,672 observed
frames become 8,303 cache rows under causal holding at 20 Hz. Admission also
requires verified image/action lineage and negligible target roundtrip error.

These SC scenes have fewer fixtures and small board/rail variation; they do
not establish coverage of hard fixture layouts. Port 0 covers NIC counts 1–3;
port 1 covers only NIC1 and initially had no held-out demonstration. Six SC episodes are
training data and three are validation data. The append workflow preserves all
251 existing SFP records and their splits. Source details and limitations are
in [DATASETS.md](../DATASETS.md#canonical-expert-collection-2026-09-18).

## Mixed-task continuation

Stage 2 started **13:16:39 UTC** from stage-1 checkpoint 006000, using all
229 training episodes and 31 validation episodes. The new immutable cache is
`cache_all_verified_sfp_sc_v2`, stored on NVMe with a link in the experiment.
Normalization is rebased to the new training statistics while preserving
physical predictions at initialization. Task bits retain identity normalization.

Sampling gives equal weight to eight task/NIC-count groups: SFP NIC1–5 and SC
NIC1–3. Consequently SC receives 37.5% of samples despite its small episode
count; per-task validation and simulator checks must detect overfitting. The
stage requested 8,000 updates, with a fresh optimizer and cosine LR multiplier
from 1.0 to 0.1; base learning rates and batch size match stage 1. It has a
100-minute stage cap and a 17:45 UTC hard training cutoff. This stage was
subsequently stopped deliberately after 6,001 updates (see below).
Commands: `act_all_sfp_sc_stage2_launch.json`; log:
`act_all_sfp_sc_stage2.log`.

At stage-2 step 2,000, pooled validation command error was **5.33 mm**, versus
**2.11 mm** on the training probe. At step 4,000 validation was **5.43 mm**;
At step 6,000 validation remained **5.43 mm**, while the training probe fell
to **1.74 mm**. Lower training loss did not resolve the generalization gap.

Step 2,000 completed all nine development trials with ground truth disabled:

| Setting | Official total | Full insertion |
| --- | ---: | --- |
| SFP, NIC1 / SC0 | 1.00 | No |
| SFP, NIC2 / SC1 | 36.87 | No |
| SFP, NIC3 / SC1 | 36.06 | No |
| SFP, NIC4 / SC1 | 35.73 | No |
| SFP, NIC5 / SC1 | 35.78 | No |
| SC port0, NIC1 / SC1 | 35.51 | No |
| SC port0, NIC2 / SC1 | 36.33 | No |
| SC port0, NIC3 / SC2 | 35.00 | No |
| SC port1, NIC1 / SC1 | 36.65 | No |

The NIC1 SFP policy moved away from the port; its final distance was 20 cm.
Model/normalizer hashes, 43D metadata, and task normalization were consistent.
Small controller tracking error and a peak tared wrist force of 8.48 N support
bad commanded motion rather than a contact stall. Its board/grasp parameters
fall within the marginal ranges of both historical and recent NIC1 training
scenes; that does not prove joint-distribution coverage. Other SFP counts
approached the port more consistently than their stage-1 step-2,000 baseline.
The four SC trials reached the correct receptacle but remained 1–2 cm short.
No learned insertion has been established. Artifact audit:
`act_all_sfp_sc_stage2/stage2_nic1_artifact_motion_audit.json`.

The four-trial SC batch also verified task resets, unique rollout directories,
per-trial timestamps, and complete official scoring. Its audit is
`act_all_sfp_sc_stage2/evaluation_step002000_sc_covered/002000/batch_audit.json`.
Final batching configurations preserve the original 32 scene/task definitions
exactly in parsed content and have not yet been evaluated.

## Initial ACT measurements

Stage-1 held-out mean first-command position error was 6.87 mm at update 1,000,
5.96 mm at 2,000, 5.52 mm at 3,000, and 5.04 mm at 6,000. The corresponding
update-6,000 training probe was 2.29 mm; a generalization gap remains. These
errors measure imitation of recorded commands,
not plug-to-port error or insertion success.

Step 2,000 completed all five fresh SFP development scenes with ground truth
disabled:

| NIC cards | SC distractors | Official total | Full insertion |
| --- | --- | --- | --- |
| 1 | 0 | 36.13 | No |
| 2 | 1 | 28.96 | No |
| 3 | 1 | 25.23 | No |
| 4 | 1 | 32.58 | No |
| 5 | 1 | 35.64 | No |

Mean total: **31.71**. Each recorded 64–76 simulation seconds. NIC4 remained
active near the wall-clock limit; the others largely stalled. None ended
before the 40-second phase range. All are development results; final scenes
remain untouched. Raw reports, videos, and contact sheets are under
`act_all_sfp_stage1/` (`step002000_review.md`,
`development_step002000_report.json`, and `videos/`).

Matched NIC1/NIC5 development checks scored 31.09/36.03 at step 4,000 and
31.31/35.59 at step 6,000; both checkpoints had **0/2 insertions**. These
two-scene comparisons do not establish saturation, but show that the lower
validation error has not yet produced successful insertion. Step 6,000 recorded
71.0/65.1 simulation seconds. Reports: `step004000_review.md`,
`step006000_review.md`, and `stage1_checkpoint_comparison.json`.

There are 59 unique development/final configurations, disjoint from admitted
experts. The primary frozen final assessment is 20 SFP scenes plus 12 SC scenes
within the newly collected SC scope. Broader SC layouts are separate challenge
probes. Final scenes remain untouched.

## Follow-up hypotheses and data checks

A fresh NIC3/SC1 corrective pilot used stage-1 step 6,000 with 25% student
selection per cycle and maximum deviations of 3 cm / 0.08 rad from the teacher.
It failed admission: total **36.885**, Tier 3 **25**, final distance 4 cm.
The recording itself passed alignment checks: 765 frames, 50 student frames,
all 2,295 images readable, separate teacher/executed targets, and maximum
applied deviation 19.4 mm / 0.0291 rad. It is preserved as a failure and is
not in the expert folder.

The unchanged teacher then succeeded on the identical scene with zero student
intervention: total **92.791**, Tier 3 **75**, 31.27 simulated seconds, no
prohibited contacts or force penalty. Terminal images were inspected for both
attempts. This single pair establishes scene solvability by that teacher and
motivates checking recovery coverage; it does not estimate either policy's
reliability. The successful teacher diagnostic was subsequently included in
the v3 append. Artifacts: `sfp_corrective_pilot/` and
`sfp_corrective_pilot_nominal_pair/`.

A bounded nominal multi-card collection completed **16/16 verified insertions**.
Together with the paired teacher pilot this supplied 17 accepted episodes,
scores 92.155–94.368, with all 22,737 recorded images checked. A second SC
collection admitted **12/16** attempts: three port-1/NIC1/SC1, three
port-0/NIC1/SC1, four port-0/NIC2/SC1, and two port-0/NIC3/SC2. Accepted scores
were 84.83–92.55; all 23,031 recorded images and all accepted terminal views
passed review. Four incomplete insertions remain excluded. Its containers
stopped after the audit; evidence is under `sc_collection_gpu0_followup/`.

At approximately 14:21 UTC, v3 appended these **29** accepted episodes, giving
**289 total (268 SFP + 21 SC), 250 train / 39 validation**, and 169,710 cached
frames. Every old record, array prefix, and split assignment was preserved;
the old/new cache remains immutable for existing readers. No training/validation
scene overlap or overlap with the 59 reserved evaluation scenes was found.
The current time-input pair continues on v2. Evidence: `v3_combined_preflight.json`,
`v3_append_result.json`, and `v3_publication_audit.json`. New appends require
explicit finite zero insertion-force penalty as well as the other gates, and
stratify target identity so new SC port-1 scenes can supply a held-out example.

The exact mixed-task sampler exposes a strong time/task association:
**98.58% of draws at elapsed time >=40 s are SC**, and **94.31% of draws at
30–40 s are SC**. The >=40 s bin represents 10.34% of sampled training frames.
This makes a time shortcut plausible; it does not prove the cause of the SFP
drift. Evidence: `v2_elapsed_time_support.json` / `.md`.

On seven fixed SFP rollout observations, changing raw elapsed time from 0 to
40 s moved the first command by a median 26.01 mm in stage 1 and 30.33 mm in
stage 2, mostly downward. The stage-2 Y change was only about -4.5 to +5.8 mm,
so this probe alone did not reproduce the large lateral drift. Task swapping
also changed predictions. These are input sensitivities, not counterfactual
accuracy or evidence that a retrained time mask works. All 70 CPU predictions
were finite; no actions were executed. `elapsed_time_counterfactual.json`
contains the observations and outputs. The probe isolated OpenCV preprocessing
in a child process after identifying a native JPEG/TIFF import collision;
the live TorchScript runtime and GPU training were unaffected.

The mixed stage stopped cleanly at **6,001 updates** after publishing checkpoint
6,000, in **2,912.75 seconds** (48m33s). This was an intentional signal to
prioritize the controlled time-input comparison, before its originally
requested 8,000 updates and well before the total budget deadline. Both new
branches start from checkpoint 6,000 and the immutable v2 cache, use
2,000 updates with global batch 128 and learning rates 1e-5 / 3e-6, and differ
only by the time-input mask. No normalization rebasing is needed: action and
non-time statistics must exactly match the parent; the masked time coordinate
has mean 0 and standard deviation 1. The launcher checks these contracts and
marks incomplete pairs as unmatched. The pair started **14:06:35 UTC**;
artifacts are under `paired_time_stage2_6k_v2/`. The stride-grid comparison
remains a later experiment, keeping timing and geometry interventions separate.

Both branches completed **2,000/2,000 updates**, in 558.54 seconds (control)
and 522.99 seconds (masked). Final validation command errors were **5.431 mm**
and **5.311 mm**, respectively; training probes were 1.681/1.930 mm. This small
offline difference is not an insertion result. All 113 post-run contract
checks passed. Exact sampling replay matched both saved RNG endpoints and
confirmed that all 229 training episodes were sampled by update 68.
Evidence: `paired_time_contract_audit.json` / `.md` and the
[training curves](../../outputs/experiments/2026-09-18_act_all_verified_6h50/training_progress.png).

The first control evaluation attempt stopped before simulation because the
experiment helper's checkpoint glob matched no real checkpoint directory.
The helper now exposes a symlink to the real checkpoint; CPU discovery checks
and the subsequent live retry passed. Original failed artifacts and helper
versions are retained. Both branches completed the corrected nine-scene protocol:

| Time input | Full insertions | SFP mean / insertions | SC mean / insertions | Overall mean |
| --- | ---: | ---: | ---: | ---: |
| Retained | 1/9 | 33.43 / 0/5 | 51.54 / 1/4 | 41.48 |
| Masked | 0/9 | 20.48 / 0/5 | 35.64 / 0/4 | 27.22 |

The retained-time SC port-1/NIC1 success scored **86.708**, Tier 3 **75**,
with zero prohibited contact and force penalty; terminal frames show seating.
However, this protocol's 90-second policy limit is wall time, while commands
are paced by simulation time. Actual simulated durations differed. Masked
NIC2/NIC5 moved 54.9/13.4 mm over their last ten simulation seconds and ended
away from the target; the corresponding control failures were nearly static.
Both branches were therefore rerun at **90 simulated seconds**, with a
180-second wall watchdog, before choosing the parent. The old evaluations,
duration audits, videos and source hashes remain preserved; no model weights
changed for this rerun. Final scenes remain unused.

### Equal simulation-duration time comparison

| Observation set | Full insertions | Mean total | Duration completeness |
| --- | ---: | ---: | --- |
| Retained time, original nine | 0/9 | 35.569 | All nine reached 90 simulated seconds |
| Masked time, original nine | 0/9 | 15.225 | Final SC port-1 trial stopped at 50.65 simulated seconds |
| Masked time, eight valid originals + separate final-scene retry | 0/9 | 23.219 | All nine selected observations reached 90 simulated seconds |

The retained-time run had zero contact/force penalties and terminal images
consistent with no insertion. Its previous SC success did not repeat. The
masked original final trial scored **−35**, including contact **−24** and force
**−12**, and hit the 180-second wall watchdog. The cause of its slowdown is not
established; contact/physics could contribute. This is an adverse observation,
not just missing data or a demonstrated infrastructure fault.

The separately preserved retry scored **36.954**, Tier 3 **25**, with zero
contact/force penalties, and reached 90 simulated seconds in 119.10 wall seconds.
The supplementary composite supports an equal-duration comparison but does not
replace the original experiment record. All ten masked attempts remain recorded:
zero insertions, mean 17.398, one contact/force-penalized attempt, with one scene
repeated. The retained-time branch has higher mean Tier 3 and total score in the
duration-matched comparison and was selected as the continuation parent.
Decision and exact evidence hashes: `time_parent_selection.json`.

**Initial-condition confound discovered after that selection:** two original
masked SC trials (port 0/NIC2 and port 1/NIC1) received first observations with
TCP positions around `[-0.379, 0.304, 0.173]` and `[-0.433, 0.260, 0.146]` m,
instead of the usual start near `[-0.372, 0.194, 0.320]` m. These frames are
captured before the policy's first command, not one second after it starts.
Their outcomes and penalties are retained, but cannot be assigned solely to
the masked policy. The supplementary composite still includes the anomalous
NIC2 start, so it does **not** establish a clean full-nine comparison. The
retained-time parent remains the engineering baseline for the already started
stride pair while the reset mechanism is investigated; subsequent evaluation
must validate physical starting conditions. The original decision is immutable;
see `time_parent_selection_addendum_initial_conditions.json`.

Closed-loop executions can diverge before their different stopping budgets:
old/new retained-time SC port-1 initial TCP positions differed only 0.00022 mm,
but differed 10.51 mm at one simulated second and 39.96 mm at five seconds.
Do not attribute the changed outcome solely to additional simulation time.
Evidence: `paired_time_stage2_6k_v2/control_sc_port1_old_vs_sim90_early_trajectory.json`.

### Matched stride continuations on v3

Started **15:18:10 UTC** from retained-time checkpoint 002000. Both arms use all
**250 training / 39 validation episodes**, 3,000 updates, batch 64 per GPU
(128 global), seed 17, identical task/count and terminal sampling, new AdamW,
head LR 1e-5 / backbone LR 3e-6, and cosine multiplier ending at 0.1. Training
statistics are rebased from the same parent in both arms. The sole model change
is final backbone stride 32 versus 16; both retain elapsed time and the same
16,354,566 parameters. Each arm has a 45-minute cap and the common 17:45 cutoff.

The readiness audit passed 82 CPU checks, including physical-unit equivalence
of all changed normalization projections, inherited masks, unchanged task bits,
all prior split assignments, and source pins. This is a matched continuation
comparison, not fresh training of both architectures. Comparison with earlier
v2 models also changes the dataset and update count. Both completed checkpoints
will use the same nine development scenes at 90 simulated seconds, after both
training branches finish. Both branches completed their 3,000 requested updates
without a deadline, signal, or cap stop. The stride-32 arm took **721.09 wall
seconds**; stride 16 took **1,210.15 seconds**. At update 3,000, first-command
validation error was **5.602 mm / 5.319 mm**, with last-three-second error
**5.935 mm / 5.707 mm**, respectively. Training probes were lower, so this is
not evidence of insertion or complete convergence. The post-run contract audit
passed **120/120** checks after a bookkeeping fix to replay 3,000 rather than
2,000 sampling updates; the initial failed audit was preserved. Both branches
sampled all 250 training episodes by update 68, but reached only 62.47% of
unique training image anchors across 384,000 draws. Commands/logs:
`paired_stride_v3_launch.json` and `paired_stride_v3_from_control_time_2k/`;
audits: `stride_v3_readiness_audit.md` and `paired_stride_contract_audit.md`.

Both update-3,000 models began a nine-scene development check on **15:54 UTC**,
with each original scene run in a fresh simulator. Every trial must pass a
pre-command initial state audit: all six named arm joints within **0.05 rad**
of configured home, finite named gripper positions recorded, and a complete
90-second simulation-duration audit. The 0.05 rad bound was fixed from six
fresh starts (maximum observed loaded error 0.03151 rad); the two displaced
masked starts differed by 0.426/0.528 rad. Gripper width has no inferred target.
The recorder now adds joint names. The nine-scene launch and audit evidence will
appear under each arm's `development_step003000_singles_sim90/` directory.

The engine's source readiness check was also repaired: stationary joints away
from home and timeout now fail instead of marking the simulator ready. A pure
C++ predicate passed **15 checks**; an isolated rootless build of the current
interfaces and engine completed, and its CTest passed. The active evaluation
image still contains the older binary. For today's remaining evaluations the
operational protection is the fresh simulator per trial and the independent
start-state auditor. The source fix has not been live-tested as an installed
engine. Evidence: `engine_readiness_fix/verification.json`.

The visual precision audit found a 288×256 ACT input and an 8×9 ResNet18
feature grid per camera. At one observed entrance stall the estimated scale
was 0.85–1.11 input pixels/mm. A stride-16 option (16×18 feature grid, unchanged
weights and parameter count) now has a persistent checkpoint/export contract.
Its 32 focused CPU checks passed, including rejection of mixed-stride checkpoint
averaging. Short contended GPU probes completed ten finite updates at batch 32
and 64; the latter used 17.21 GB allocated memory. This establishes feasibility
for a matched global-batch-128 comparison, not a policy improvement. Insertion
benefit remains unmeasured. A matched FP32 TorchScript timing check used 20
warmups and 100 alternating synchronized samples per model, under concurrent
DDP: stride 32 p50/p95/p99 = **17.79/25.23/25.96 ms**, stride 16 =
**20.23/26.35/26.74 ms**. Both were below 200 ms for model computation;
preprocessing, transfers, ROS, and actuation are excluded. Exact configs,
model/source hashes and samples: `stride_model_latency_fp32.json` / `.md`.
The later two-GPU production-style probe completed 100 updates on v3 at batch
64 per rank while both development simulations ran: **0.510 s/update**, 250.77
global examples/s, and 17.27 GB peak allocated memory per rank. Its projected
3,000-update wall time, including validation and saves, is **27.11 minutes**;
this is a throughput projection, not evidence of convergence. The saved
probe model is not a selection candidate. Evidence:
`stride16_ddp_v3_b64_throughput100_result.json`.
Upsampling the existing cache would not recover image detail lost during
recording. An elapsed-time-masked comparison remains a separate hypothesis.

## Stage-1 visual and contact diagnostics

On an episode-balanced 448-frame diagnostic, stage-1 step 2,000 averaged 6.07 mm error.
ImageNet-mean blank images raise that to 42.47 mm; matched image swaps raise
the matched subset to 18.89 mm, a paired increase of 13.02 mm. This establishes
visual dependence, not correct localization. Port swaps change output by
3.25 mm on average; there are no ground-truth counterfactual targets, so this
is sensitivity, not task-selection accuracy. Warm model-only FP32 TorchScript
latency under concurrent DDP was p95 **22.62 ms**, p99 **23.76 ms**; it excludes
camera handling, preprocessing, transfers, and ROS.

Post-hoc official-bag geometry on the NIC1 failure shows the tip stopped at
the entrance, **7.91 mm off axis**, with 1.97 degrees angular error and
45.80 mm of seating depth remaining. It stayed there for 58.4 simulation
seconds with approximately 12.7 N wrist force. The final command requested
another 49.75 mm downward and its projected tip still missed the axis by
5.90 mm. This supports a contact stall from misalignment; the exact contact
bodies are not directly identified. The source bag, measurements and plot
are indexed by `step002000_dev_nic1_geometry_audit.json` / `.md` / `.png`.

## Important interpretation limits

Historical camera arrays were BGR and historical recording timestamps followed
wall time; later aligned recordings use RGB and simulation timestamps. The
loader reconciles channel order, while the timestamp difference remains
explicit. A score-verified expert outcome does not itself validate all action
labels. No fabricated action labels or unverified SC episodes are admitted.

The complete legacy timing audit (`legacy_timing_audit.json` / `.md`) found
dataset-duration / official-simulation-duration ratios of **1.35–1.69**, median
**1.46**, for the 140 historical episodes; the corresponding recent median
is 0.984. Legacy arrays and videos contain no per-frame simulation clock.
There are 14,563 adjacent historical pairs with identical 32D state but changed
commands, including 380 translation changes greater than 1 mm. Six of twelve
spaced checks also had identical pixels in all three cameras. These observations
establish ambiguous timing, not a recoverable exact retiming map. A controlled
elapsed-time-masked continuation is being evaluated; the initial run and cache
are preserved unchanged.
