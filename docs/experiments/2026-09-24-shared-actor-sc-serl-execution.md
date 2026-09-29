# Shared SFP/SC actor, SC demonstrations, and gated recovery RL

Date: 2026-09-24. Status: **execution in progress**. This supersedes the
training order in the earlier [hierarchical continuation](2026-09-24-hierarchical-recovery-continuation.md),
while retaining its failure evidence and safety limits. The selected old RPDP
checkpoint is Isaac SFP-to-NIC near-port BC followed by a small SAC test; it
was not trained on SC-to-SC or full transport. Its 1/8 to 3/8 video rerun is
not proof that RL learned a lateral direction: action sampling was off,
recovery events were not logged, and run outcomes varied. See the
[experiment record](2026-09-22-serl-mixture-recovery.md).

## Fixed design

- **One deployable visual actor** handles SFP-to-NIC and SC-to-SC, full
  transport, alignment, insertion, and post-retreat retry. Task family,
  selected target, visual card context, observation-only relative pose, robot/force
  history, and compact recovery history condition it. No separate transport or
  recovery actor; no residual/delta-action head on a frozen policy.
- The actor predicts complete **recorded TCP target poses in the selected
  port-opening frame**. A
  deterministic SE(3) converter issues TCP-body-frame commands. Privileged
  geometry can generate teacher labels and diagnose rollouts, but cannot enter
  autonomous actor observations, select evaluation crops, or choose retry
  directions. Match the label/action timebase, target opening, and tool/plug
  transforms across both tasks.

The TCP target contract replaces the initially proposed connector-target
contract for the **shared** SFP/SC actor. Scored SFP bags show that the port
frame can be reconstructed from each trial YAML to within about 0.005 mm in
a checked scene, whereas the physical cable grasp varies by reset and most
historical SFP datasets have no retained per-episode plug TF. Predicting the
recorded TCP target in the port frame preserves exact teacher commands for
both families without inventing a fixed SFP TCP→plug label. The visual
plug-to-opening pose estimator remains an actor input. This is still a
complete target pose, not a delta correction to another policy. Retain the
SC connector-target labels as an ablation and for geometric diagnostics.
The 10D task vector has two family bits, two selected-port bits, five
**target NIC-card index** bits, and one card-valid bit. SC-to-SC has no NIC
target, so its last six bits are correctly zero even when NIC cards are
present as obstacles. Their **count is not explicitly encoded**; the three
RGB views must show them. This matters for the sparse four/five-card SC data
and should be tested before attributing failures to recovery behavior.
- A deterministic supervisor owns blockage detection, measured SE(3) retreat,
  force clearance, and route-checkpoint escalation. Its behavior and state are
  identical during training and deployment. The **same actor** chooses normal
  motion and a lateral/route retry after control returns.
- During training, sample coherent full-trajectory alternatives and learn
  their probabilities and action values from real simulator rewards. For
  deployment/evaluation, freeze the deterministic or stochastic selection
  rule on development scenes and report repeated-seed variance. Record
  proposed, sampled, projected, and actually executed actions separately.
- Only the currently reproduced Isaac incident classes are training targets:
  plug/port, gripper/card, and plug/card blockage. Secondary cable/card
  contact during a gripper jam is **not** an isolated cable-snag label. The
  high-force diagnostic scripts are not demonstrations.
- Rootless Docker, no interruption of active jobs, at most four GPUs at once,
  complete inference p95 <300 ms. Keep four reserved final IK configurations
  sealed until the development policy and thresholds are frozen.

## Ordered steps and pass criteria

1. **Freeze inventory and split.** Hash exact Gazebo/Isaac scenes, bags,
   videos, scored outcomes, model checkpoints, simulator images, task encoding,
   and current code diff. Group by episode/reset configuration and parent
   scene. Record invalid/no-insertion trajectories separately. Pass: a
   machine-readable, reproducible inventory and sealed final manifest.
2. **Validate Isaac SC simulation and images.** Compare the corrected grasp,
   20 cable links, port colliders, contact pairs, reset/terminal timing, and
   visual appearance against matched scored Gazebo SC episodes. The local RTX
   guard override now produces three cameras, but moving distal cable looks
   sharply kinked/looped; locate and repair any material mismatch. Pass:
   moving three-camera visual rollouts, orange cable, plausible distal cable
   shape and motion, and normal-collision insertion/contact checks on fresh
   0–5-card development scenes. If distal cable remains unfaithful, do not
   train or claim cable-specific Isaac recovery.
3. **Validate SC targets and perception.** Reconstruct recorded teacher
   commands as fixed-timed connector waypoints in the selected port-opening
   frame; audit the `world`/`aic_world` alias, opening offsets, SC grasp, and
   missing command/image intervals. Train/test an observation-only SC plug and
   port estimator, grouped by episode and scene. Pass: correct action
   round-trip and held-out lateral/orientation estimates accurate enough for
   the 0.5 mm insertion corridor; no true geometry at actor evaluation.
4. **Fill successful SC Gazebo demonstrations.** The 21 canonical verified SC
   episodes cover one to three NIC cards; collect scored full-insertion
   episodes for missing card counts, both SC target ports, and varied starts.
   Use bounded CheatCode/backtrack/retry search when naïve teacher motion
   fails. Preserve force, contact, commands, images, and failed attempts.
   Privileged search is allowed only for teacher data. Pass: complete,
   replayable successes over the development task matrix, with held-out
   scene groups and no failed trajectory admitted to BC.
5. **Train a single shared BC actor.** Balance verified SFP and SC episodes,
   preserve the one-hot task/target encoding, include full-start transport,
   and train complete port-relative TCP target trajectories. Initialize the
   primary probabilistic mode from the supervised policy. Pass: autonomous
   full-depth scored insertion on new development starts in both families,
   including multi-card SC, with safe force and <300 ms p95 inference. If it
   fails, improve labels/data/perception before RL.
6. **Validate the supervisor.** Trigger on persistent force plus commanded
   motion and measured stall, not force alone. Retrace measured position and
   orientation to a low-force checkpoint; escalate to an earlier route
   checkpoint after repeated same-corridor blockage. Pass: deterministic unit
   and live incident tests, terminal/reset accounting, no false retreat on
   valid insertion, bounded force/retries, and actor-ownership logs.
7. **Frozen-actor recovery ablation.** Same BC actor, development scenes,
   and budgets with no backoff, local backoff, and hierarchy. Pass: measure
   incident clearance and scored insertion, peak/integrated force, retries,
   path/time, and p95 latency; continue hierarchy only if useful without
   force regression.
8. **Optional offline critic gate.** Build causal replay from successes,
   failures, and safe recovery, retaining terminal observation and executed
   action. Warm critics with the actor frozen; compare against direct online
   initialization on episode-grouped held-out outcomes. If critic ranking or
   calibration fails, skip offline RL.
9. **Online SERL on short Isaac incidents.** Start from shared BC, mix
   immutable prior and online replay, regularize the whole action mixture,
   and mask supervisor-owned decisions from actor loss. Match no/local/
   hierarchical arms by starts, updates, and evaluation. Pass: improved
   held-out scored insertion and safe contact, diverse useful retry choices,
   no categorical mode collapse or SFP forgetting.
10. **Scale only after the incident gate.** Cover all supported card counts,
    target ports, starts, and both task families with complete normal episodes
    as well as failure starts. Episode count is determined from learning
    curves, not fixed in advance. Pass: stable gains across new episode groups.
11. **Isaac-to-Gazebo transfer.** Check camera, pose, force, timing, and action
    parity; use scored Gazebo rollouts to diagnose any gap. Apply targeted
    supervised correction and repair the sequential Gazebo RL bridge if
    online adaptation is necessary. Pass: same actor/supervisor autonomously
    inserts in Gazebo, including recovered failures, without privileged input.
12. **Freeze development choices.** Select checkpoint, recovery thresholds,
    stochastic/deterministic deployment rule, retries, and time limits using
    only development scenes. Pass: reproducible frozen bundle with inference
    p95 <300 ms and exact dependency/asset hashes.
13. **Official and reserved evaluation.** Run released production settings
    first and then the sealed final split once, without tuning on those
    scenes. Report complete scored insertions by task, card count, and port;
    axial/lateral/orientation error, force, contacts, latency, and failures.
14. **Archive and maintain.** Preserve both successful and failed models,
    scene manifests, scripts, metrics, plots, videos, raw replay pointers,
    and decisions. Update STATUS, EXPERIMENTS, the artifact map, and this log
    after each gate.

## Execution log

| Step | State | Evidence and decision |
| --- | --- | --- |
| 1 | Passed | [Frozen machine inventory](2026-09-24-shared-actor-sc-inventory.json): 289 verified episodes (250 train/39 validation), 268 SFP/21 SC, 287 scene hashes and zero cross-split scene hashes. SC only covers NIC 1–3; port 1 appears only at NIC 1. Four reserved final YAMLs were byte-hashed without evaluating them. Historical BC/RL checkpoints and the manifest are hashed. Dirty tracked/untracked code snapshot is under `/var/tmp/aic_shared_sc_20260924/inventory/`. |
| 2 | Passed for the bounded mechanics gate | Repaired the five-card SC scene and restored three-camera rendering. Spurious robot self-collision caused 20.8 kN internal finger contacts and distal kinks; with it off, the moving cable has no >90° turn and 37.1 mm minimum adjacent-link distance. Controlled 0/3/5-card full-start scenes each reached 0.074 mm from the seated target. External gripper/card contact (364 N) and 3 mm misaligned plug/port blockage (47 N) still occur. [Machine results](2026-09-24-isaac-sc-self-collision-gate.json) and [visual/shape evidence](2026-09-24-isaac-sc-fidelity-fix.md#cable-self-collision-diagnosis-and-corrected-moving-view). This does not prove exact Gazebo cable dynamics or isolated snag. |
| 3 | Fresh frozen offline check passed pooled lateral gate; one-scene and live gates open | The original port tracker latched onto a false position, giving **45.5 mm** combined lateral p95 on the second suite. A corrected observation-only tracker was frozen before a third independent eight-scene capture. On 228 near-port frames from six third-suite episodes, port lateral p95 was **0.181 mm** and fused physical plug-to-opening lateral p95 was **0.443 mm**, with 227/228 fused frames initialized. One scene reached **1.004 mm per-episode lateral p95**, so pooled success does not establish every-scene corridor accuracy. Scored-bag port TF used **only after inference** gives orientation yaw p95 **0.672°**; one scene reached **0.981°** and merits training-scene diagnosis before calling orientation solved. Complete offline native-JPEG decode, crops, four CNNs, geometry, and rail yaw measured **147.36 ms p95**, excluding ROS, actor, and command conversion. Raw MCAPs were losslessly archived and verified. The [supervision audit](2026-09-24-sc-supervision-audit.md) preserves the failed alternatives and exact artifact paths. Four reserved final IK settings remain sealed. |
| 4 | Task-cell coverage and bounded BC admission passed; diversity still limited | The 21 canonical scored SC experts and **11** clean supplemental full-insertion candidates now cover all **12** `(0–5 NIC cards, SC port 0/1)` cells, though some have just one success. The two-card/port-1 gap closed when a teacher-only 5 mm near-port bias cap raised a matched partial from Tier 3 **42.3 to 75**. The same cap caused a five-card/port-0 matched success to fall from 75 to **16.0**, so it is not a universal rule. The [matrix](2026-09-24-sc-near-bias-and-success-matrix.json) preserves both outcomes. Episode-specific physical-grasp relabeling produced **6,220** causal held-target chunks for the 11 candidates, with target SE(3) round-trip error at floating-point precision. A [separate admission audit](2026-09-24-sc-bc-admission.json) checked every successful episode, scene split, command/frame stream, and force/stall pattern: 11/11 and all 6,220 chunks passed. Failed matched runs are excluded. High-card generalization still needs new held-out starts. |
| 5 | First shared BC failed autonomous gate; corrected supervised comparison running | Exact full TCP-target labels cover 268 SFP and 32 successful SC episodes. The first 13.60M-parameter RGB/state/task actor trained for 6,500 updates and selected update 4,500. Held-out first-target translation median/p95 was **2.48/11.81 mm SFP**, **2.63/5.55 mm SC** on 320 randomly sampled held-out rows per family. One no-privilege SC full-start rollout failed with final plug **0.51 m** from opening, peak **785.95 N**, and an off-limit wrist contact. On that same teacher scene, the initial body command was **16.01 mm** from the near-zero teacher target; pose estimate was 6.04 mm and 8.06° wrong. The runtime's old TCP-frame delta command also had possible controller-reference mismatch. The next supervised arm adds differentiable command consistency, oversamples the first 50 frames, and sends observation-referenced **absolute base-frame setpoints** after converting the full port-frame target; a 40 N diagnostic stop is enabled. This changes control transport, not the actor's target representation. No RL or recovery promotion. |
| 6–14 | Not opened | Supervised full-start insertion and safe-force gate failed; do not use RL to compensate for an unvalidated BC/runtime. |

### First shared BC and causal failure diagnostic

The one-GPU rootless training command, full log, selected `best.pt`,
`history.json`, and `summary.json` are under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_full_20260924/`.
The command used `scripts/train_shared_port_tcp_bc.py`, all three label
JSONLs in the artifact map, the current 289-episode verified image cache,
`--updates 8000 --batch-size 32 --workers 4 --val-batches 20 --val-every 500
--patience 2000`. It stopped at 6,500 updates, selecting update 4,500. Source
hashes and the 10D family/port/card vector are stored in `best.pt`.
True TCP-in-port geometry is an auxiliary *training label*: the actor's pose
head receives only RGB, ordinary robot state, and task encoding, and the
action head consumes its own predicted pose. The deployable output is four
full TCP targets in the selected port-opening frame. The deterministic
converter passed a nonzero SFP recorded-action round trip within
`6.8e-9` in action units; it does not consult scored TF.

The first unprivileged rollout used `trial_990501` from a new development
suite, `RunSharedPortTCP`, the selected checkpoint, 90 s runtime,
`delta_pose`/`gripper/tcp` command transport, and no teacher, guard, or
privileged input. Scored Tier 3 was zero, total **-35**, final plug distance
**0.51 m**; peak force was **785.95 N**, and the wrist touched an enclosure
wall. Exact `sc_development_trial.yaml`, the generated evaluation command and
logs, scored YAML, MCAP, and once-per-second three-view JPEGs are under
`eval_shared_sc_dev/best/attempt_0001/`. This unsafe failure bars more live
rollouts with that runtime/checkpoint combination. The predicted current
port-frame pose and teacher target were compared on the *matching scored
teacher scene* in `sc_teacher_scene_diagnostic.json`: the first nearly-zero
teacher action became a predicted body translation `[2.31, 1.03, -15.81]`
mm; pose position/orientation error was **6.04 mm / 8.06°**. Later sampled
teacher-scene command errors ranged about **2.2–11.8 mm**. Scored port TF was
read only for this posthoc diagnosis, never during actor inference.

A grouped offline audit sampled 25 rows per episode and axial phase across
all 39 held-out groups (2,925 rows). It measures the final TCP-body command
produced by the **predicted pose and predicted full target together**, using
truth only for metric calculation. For SFP transport/approach/near-port,
command-error median/p95 was **3.40/10.36**, **1.77/6.86**, **2.61/8.82 mm**
across 32 episodes. For SC it was **2.09/8.70**, **1.73/5.94**,
**1.57/4.98 mm** across seven episodes. The batched model-only forward p95
was 22.50 ms, excluding image decode, native SC perception, ROS, and command
conversion. Exact metrics are
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_full_20260924/grouped_offline_metrics.json`.
These near-port errors are far above 0.5 mm and explain why offline target
medians cannot be used as an insertion gate.

The next supervised arm keeps the same 13.60M model, initialization, data,
task encoding, and port-frame complete-target output. It warm-starts the
selected checkpoint, adds a differentiable loss on the **resulting TCP-body
command**, and oversamples the first 50 frames of each episode. The runtime
will compose the predicted port target with the *observed* TCP into an
absolute `base_link` setpoint, preventing ambiguity between observed and
controller-relative delta poses, and stops the diagnostic if measured force
exceeds 40 N. This is not a residual action head. It must pass grouped
offline phase errors and a low-force live gate before any recovery/RL work.

That command-consistency arm completed 4,000 updates on rootless GPU 3 and
selected update 2,500. It improved held-out first-command median/p95 to
**1.65/7.10 mm SFP** and **1.70/4.04 mm SC** on the fixed 320-row-per-family
sample. A separate 2,925-row phase audit found SC near-port command
median/p95 **1.46/3.84 mm** (versus **1.57/4.98 mm** before), while SFP
near-port was **1.84/9.16 mm** (versus **2.61/8.82 mm**). On the matched
teacher scene the first command error fell from 16.01 to **10.66 mm**, still
unacceptable for an initial hold. Exact checkpoint, code snapshot, grouped
metrics, and teacher-scene diagnostic are under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_command_full_20260924/`.
The 40 N guarded absolute-setpoint runtime has **not** been live tested;
near-port and initial-action offline gates failed. These numbers support
continuing supervised perception/phase work, not starting recovery RL.

The teacher's initial frames can demand almost no movement before transport;
current RGB and TCP state alone do not encode how long the episode has been
running. A third supervised variant adds **elapsed simulation time from the
first observed camera stamp** as an ordinary causal input to the same actor.
It warm-starts the command-consistency arm with a zero-initialized time
branch, preserving its initial outputs, and continues command loss and
initial-frame oversampling. This is not a phase oracle or privileged scene
input. Its one-GPU rootless run is under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_time_full_20260924/`.
It remains subject to the same grouped near-port and first-command gates.

### Elapsed time, frozen perception, and pose upper bound

The elapsed-time arm completed 2,000 updates and early-stopped, selecting
update 1,000. On the matched scored teacher scene, its first commanded
translation missed the teacher by **5.29 mm**, down from 16.01 and 10.66 mm
in the preceding arms. The grouped 39-episode audit still found SC near-port
body-command median/p95 **1.158/3.338 mm**, and SFP **1.836/8.966 mm**.
The axis-specific re-score of that same frozen checkpoint gives SC near-port
**0.366/1.204 mm lateral** and **1.134/3.016 mm axial** command errors;
SFP near-port is **1.034/3.047 mm lateral**. These are commanded-displacement
errors relative to the recorded teacher, not autonomous insertion offsets.
The 0.5 mm corridor applies to lateral alignment, so the lateral figures are
the appropriate precision diagnostic. All metrics use complete held-out
episode groups; the phase uses scored geometry only to classify rows after
inference.

The fourth SC perception confirmation used eight new generator trials
`990601`–`990608`. Source code, model hashes, fixed thresholds and calibration
were written to `sc_yaw_boundary_refreeze_eval_20260924/frozen_bundle.json`
**before** any of these scenes were captured. Four scenes reached near-port
(154 frames); the observation-only RGB estimator achieved combined physical
plug-to-opening lateral **0.156/0.294 mm median/p95**, axial
**0.201/0.309 mm**, and board-line yaw **0.048/0.084°**. Position and yaw
histories initialized on every near-port frame. The largest per-scene lateral
p95 was 0.314 mm; the largest yaw p95 was 0.156°. The rail candidate was
accepted on only 14.3% of near-port frames in that latter scene, so the
causal history held its previous estimate during occlusion. This remains an
offline test on teacher views; only four of eight scenes reached the critical
phase. No scored TF or projected label selected autonomous crops. All eight
raw MCAPs were losslessly compressed to NAS, decompressed for SHA256 check,
then removed locally; the restore paths and hashes are in
`2026-09-24-sc-yaw-boundary-refreeze-mcap-archive.json`.

To separate **TCP localization** from action mapping, a bounded **optimistic
upper-bound diagnostic** trained from the elapsed-time checkpoint for 3,000
updates, selecting update 2,000. Only on SC frames within 30 mm axially, it
substituted the true current TCP pose plus 0.3 mm synthetic translation noise
and small orientation noise as the action head's pose input. This is **not a
deployable policy**; the runtime explicitly rejects its checkpoint and the
offline scorer requires `--oracle-sc-pose-diagnostic`. It still reached only
SC near-port body-command **1.167/3.270 mm**, lateral **0.571/1.141 mm**,
axial **0.883/3.133 mm** median/p95 on the same 7 held-out SC episodes.
Compared with the elapsed-time arm, more accurate **TCP pose** did not remove
the action-mapping gap. This is distinct from the SC high-resolution
estimator's **physical plug tip relative to opening**: variable plug-to-TCP
grasp geometry means that a true TCP pose does not reveal the plug tip. A
separate nondeployable physical-tip-pose input test is running before deciding
whether to attach that estimator to the shared actor. The inference mask
uses truth only in the TCP diagnostic;
its result cannot be presented as autonomous evidence.

Exact selected checkpoints, source snapshots, training histories, summaries,
and axis scores are under `/var/tmp/chmin_aic_20260918_act/`:
`shared_port_tcp_bc_time_full_20260924/`,
`shared_port_tcp_bc_sc_oracle_full_20260924/`, and
`sc_yaw_boundary_refreeze_eval_20260924/`. The fourth suite's image inference
order was `train_sc_native_locator_v2.py` for port and tip crops, then frozen
`predict_sc_native_keypoints.py`, `predict_sc_orientation_landmarks.py`, and
`predict_sc_native_tip_only.py`, followed by the pairwise tracker, relative
fusion, board-line yaw (`--reject-search-boundary`), and physical-tip scorers.
The exact paths and outputs are in the frozen bundle and JSON reports there.

The next bounded supervised arm starts from the deployable elapsed-time
checkpoint, oversamples rows within 30 mm, and adds a direct near-port
**lateral command and complete-target loss**. This still predicts full
port-frame targets; it is not a residual/delta policy. Its one-GPU run is
`shared_port_tcp_bc_lateral_full_20260924/`. Compare its frozen grouped
lateral, axial, first-command, and task-family errors before any live rollout
or recovery RL. If lateral p95 remains above the corridor, diagnose target
ambiguity and collect more varied successful SC episodes before further
actor capacity changes.

That nonprivileged lateral arm early-stopped at 1,500 updates and selected
update 500. On the same seven held-out SC episodes, 0–30 mm lateral
command error changed from **0.366/1.204** to **0.595/0.916 mm median/p95**;
the tail improved while the median worsened. SFP 0–30 mm lateral p95 changed
from 3.047 to **2.736 mm**. For the five SC episodes that actually reached
**0–5 mm TCP axial distance**, lateral median/p95 was **0.586/0.752 mm**;
the elapsed-time baseline was 0.369/0.820 mm. TCP axial distance is a
grouping proxy here, not the measured plug-tip insertion depth. It narrows
the critical phase but cannot alone certify successful seating. The fixed
episode-grouped metrics are in each arm's
`grouped_offline_metrics_fine_axis.json`. This is insufficient to promote
the actor.

The first oracle substituted **robot TCP pose**. It does not reveal the
physical plug-tip offset, which changes with SC grasp and cable mechanics.
A separate bounded tip-input arm adds a 17,364-parameter tip context branch
to the same 13.60M actor and supplies **true physical SC plug-tip pose plus
0.3 mm synthetic noise** only within 30 mm for this training-only upper
bound. The runtime rejects both oracle variants; no true tip pose is passed
to an autonomous actor. A joint TCP-plus-tip oracle arm follows if the
tip-only result leaves material ambiguity. Both remain distinct from a
future RGB-estimated-tip input and must be judged on the same held-out
episodes. Artifacts are `shared_port_tcp_bc_sc_tip_oracle_full_20260924/`
and `shared_port_tcp_bc_joint_oracle_full_20260924/`.

Both bounded runs completed. The tip-only arm early-stopped at 1,500
updates, selecting update 500. On the fixed five SC held-out episodes with
0–5 mm TCP axial distance, its lateral command median/p95 was
**0.289/0.714 mm** versus **0.369/0.820 mm** for the nonprivileged
elapsed-time actor. The jointly conditioned arm selected update 750 and
worsened to **1.351/1.868 mm** on those same 65 frames. This arm failed;
adding two privileged inputs and changing optimization also makes it an
unfair conclusion about the deployable RGB estimator. The tip-only result
indicates a small possible benefit, but neither oracle arm meets the
0.5 mm tail gate. Selection by an all-phase median may miss the best
near-port checkpoint, so the next policy comparison must use a predefined
near-port metric and identical training budgets. No oracle checkpoint can
run in `RunSharedPortTCP`.

Because SC has only **25 successful training episodes** (seven held out),
a fresh eight-scene **training-only** Gazebo teacher batch was generated with
the official `qualification_eval_like` SC profile, seed `2026092424`, and
trial IDs `991001`–`991008`. Its exact generator source, scene YAML hashes,
card counts, CheatCode collection settings, all attempts, and frames are
under `sc_bc_expansion_batch1_20260924/`. The rootless GPU-0 collection
uses `scripts/prepare_sc_bc_expansion_batch.py` and the saved `run.sh`;
only scored full insertion with clean action/image causality will be added
to BC. It neither opens reserved final IK scenes nor alters the seven SC
held-out groups. Card counts in this random batch are one to three; targeted
four/five-card success collection remains necessary for those sparse cells.

An episode-level [teacher target plot](../../outputs/experiments/2026-09-24_shared_sc_bc_diagnostics/teacher_near_target_scatter.png)
and [exact values](2026-09-24-sc-teacher-target-diversity.json) show why
accurate **TCP localization alone** is insufficient. For the 32 successful
SC episodes, median near-port current TCP points cluster near x ≈ 4–5 mm
in the selected opening frame, while recorded first TCP target medians span
roughly x = -6 to +4 mm and y = -5 to +5 mm. The plot contains one arrow
per episode, from median current TCP to median recorded target, colored by
selected port; squares mark supplemental teachers and black outlines mark
held-out episodes. This is an analysis of teacher labels, not an actor result.
It establishes distinct recorded actions at similar TCP locations; it does
**not** establish whether physical plug pose, visible cable route, teacher
phase, or stochastic teacher choice is the main cause. Causal history and
more varied successful scenes remain candidate fixes.

### Fresh SC teacher expansion and matched route retries

The first eight-scene qualification-like training batch finished on rootless
GPU 0. Only **2/8** CheatCode attempts fully inserted: `trial_991002`
(three NIC cards, SC port 0) and `trial_991006` (one card, port 1). The
other six scored 38.36, 0, 20.61, 17.97, 12.56, and 23.29; they remain
outside BC. The two successes passed the independent admission audit and
provide **1,579** causal four-target chunks, all with complete RGB, aligned
commands, stable physical grasp, and TCP-target round-trip error at floating
point precision. The exact scores and cell mapping are in
[batch score JSON](2026-09-24-sc-bc-expansion-batch1-scored.json), the
[admission audit](2026-09-24-sc-bc-expansion-batch1-admission.json), and the
[label summary](2026-09-24-sc-bc-expansion-batch1-waypoint-labels.json).
All eight MCAPs were losslessly compressed and SHA256 verified on NAS before
local removal; [archive manifest](2026-09-24-sc-bc-expansion-batch1-mcap-archive.json).
The [three-camera endpoint sheet](../../outputs/experiments/2026-09-24_shared_sc_bc_diagnostics/batch1_terminal_sheet.jpg)
shows each trial's final narrow views but does not expose the whole cable
route.

An observation-side [tracking audit](2026-09-24-sc-bc-expansion-batch1-tracking.json)
used the collector's actual executed base-frame setpoints, measured TCP,
and wrist wrench. Four failed trials (`991001`, `991003`, `991007`,
`991008`) moved at most **0.4 mm over their final five seconds**, while
their final TCP-to-setpoint gaps were **50, 32, 36, and 76 mm**. The
observed peak wrist force across the eight was roughly 21–35 N.
This supports a route/IK/contact tracking stall in those incidents; it
does not identify a cable/card contact pair or prove that every failure
is a cable snag. The scored terminal plug-to-port gaps for the no-insertion
cases were about 30–70 mm.

A matched teacher-only **8 cm pre-route lift** retried parents `991001`,
`991003`, and `991008` with unchanged scene geometry and no student action.
Tier 3 became **19.11, 16.15, 16.60**, versus parent **38.36, 0, 23.29**;
all three remain failures. One retry had **0.843 mm physical tip/grasp
drift**, so it would also fail successful-demo admission. Exact attempts,
source lineage, scores, and bag restore paths are in
[retry score JSON](2026-09-24-sc-bc-expansion-lift-retry1-scored.json) and
[archive manifest](2026-09-24-sc-bc-expansion-lift-retry1-mcap-archive.json).
A separate teacher-only **5 mm near-port bias cap** then retried parents
`991001` and `991008`, whose original scored endpoint was closest to
insertion. It reached **20.92 and 23.58** versus parent **38.36 and
23.29**, with **0/2** full insertion. Both failures remain outside BC.
See [near-cap score JSON](2026-09-24-sc-bc-expansion-nearcap-retry1-scored.json)
and the verified [bag archive](2026-09-24-sc-bc-expansion-nearcap-retry1-mcap-archive.json).
Neither bounded teacher modification generalizes from its earlier isolated
success; the stalled route still needs a different teacher search or
simulator/controller diagnosis.

The next supervised comparison warm-started the same elapsed-time actor,
adds the two newly admitted SC episodes as an **extra** label file without
altering the original 268 SFP/32 SC sources, and selects checkpoints by a
fixed held-out near-lateral p95 metric. The selection sample explicitly
includes SC 0–5 mm TCP axial frames and SFP 0–30 mm frames, with grouped
posthoc scoring on unchanged 39 held-out episodes. The batch's six failed
trajectories are not BC labels. Run and source snapshot are under
`shared_port_tcp_bc_expanded2_full_20260924/`. It completed 2,750 updates
and selected update 1,750. On the fixed grouped held-out set, the five SC
episodes reaching 0–5 mm TCP axial distance had lateral command
**0.198/0.337 mm median/p95** over 65 frames, compared with
**0.369/0.820 mm** for the elapsed-time arm. The SC 5–10 mm band was
**0.232/0.368 mm**; SC 10–30 mm **0.204/0.533 mm**. SFP's 10–30 mm
band remained **0.843/2.650 mm**. This selected checkpoint passed the
*offline SC lateral command* component gate on the observed seating rows,
but failed the shared SFP and autonomous insertion gates. On the matched
SC scored teacher scene its first command still missed the near-zero
teacher by **7.17 mm**. A force-stopped, 5 mm-per-command absolute-base
Gazebo diagnostic tested the corrected runtime; it was not a promotion
rollout.

That diagnostic completed on the previously used **development** SC
`trial_990501` with no teacher, true port/plug pose, guide, or exploration
at inference. It used `RunSharedPortTCP`, the expanded checkpoint,
`absolute_pose` in `base_link`, a 5 mm per-component translation clamp,
0.05 rad rotation clamp, and 40 N diagnostic force stop. It **failed**:
Tier 3 = 0 / task incomplete. The observation stream crossed 40 N at
37.20 s, peaked at **43.50 N**, and the policy stopped at a measured
42.0 N. In the matching scored teacher scene, the initial measured TCPs
differed by only **0.50 mm**, yet the [path plot](../../outputs/experiments/2026-09-24_shared_sc_bc_diagnostics/guarded_sc_vs_teacher_tcp_path.png)
shows the autonomous actor moved along a different transport route. The
teacher moved primarily toward positive base-y while holding near base-x
≈ −0.40 m; the actor moved toward positive base-x and negative base-y,
ending near a white enclosure structure visible in the
[first/middle/last camera sheet](../../outputs/experiments/2026-09-24_shared_sc_bc_diagnostics/guarded_sc_first_mid_last.jpg).
Its measured TCP moved **0.416 m** from the start, so this is not a mere
failure to move. At peak force the TCP-to-issued-setpoint gap was **6.94
mm**. The image suggests enclosure proximity; the exact contact pair has
not been verified from the bag, so do not call it a cable snag. Exact
scored output, logs, 29 three-camera snapshots, and 20 Hz observation-side
TCP/force trace are under
`shared_port_tcp_bc_expanded2_full_20260924/eval_guarded_sc_dev/best/attempt_0001/`.
The MCAP was decompressed and SHA256 checked in the
[NAS archive](2026-09-24-shared-bc-expanded2-guarded-sc-mcap-archive.json)
before local removal. This is decisive **step-5 failure despite the
near-port offline SC component pass**; recovery RL remains closed.

The likely control-learning issue is compounding off-expert transport
error: on the matching *teacher* states, predicted directions after frame
50 were broadly similar to recorded commands, while the autonomous
closed-loop rollout left that state distribution and kept chasing a wrong
route. This is an inference from the paired trajectories, not a proven
single root cause. Next, collect bounded teacher corrections from actual
student-induced off-policy states (DAgger-style) with the existing
privileged collector's clipped student window, label with the clean
teacher target, train the same one actor, and repeat the identical
guarded development start. Every correction must record proposed,
executed, teacher target, force, and scored outcome. If the teacher cannot
recover those states, improve its route planner before admitting data.

The first shared-actor correction smoke used the *already successful*
`trial_991002` training scene as `trial_994001`. Student proposals were
interleaved with teacher control in 2.5 s windows, at 0.5 selection
probability per four-second cycle, clipped to **10 mm** position and
**0.03 rad** orientation from the teacher setpoint. The 1,167 recorded
frames include 200 student-active frames, and the observed maximum target
disagreement was exactly the 10 mm clip. The official result was only
**Tier 3 = 40.41, about 10 mm partial insertion**, versus the unperturbed
teacher's Tier 3 = 75. The force in the recorded observation stream peaked
at 21.27 N; this is not evidence of a particular contact pair. It failed
the full-success admission gate, so **zero frames from this run enter BC**.
The source images, per-tick proposed/executed/teacher targets, score, and
logs remain under
`/var/tmp/chmin_aic_20260918_act/sc_shared_dagger_smoke_20260924/`.
The raw MCAP is in the [losslessly verified NAS archive](2026-09-24-sc-shared-dagger-smoke-mcap-archive.json).
The next matched smoke keeps the scene and student checkpoint fixed,
reduces the clips to 3 mm/0.01 rad and cycle selection probability to
0.25, and permits 140 s runtime. It recorded 100 active frames in a
1,159-frame episode and reached **Tier 3 = 46.05, about 10 mm partial
insertion**. This also fails full-success admission; its raw bag is in the
[verified NAS archive](2026-09-24-sc-shared-dagger-smallclip-mcap-archive.json).
An otherwise identical **zero-student control** also ended at Tier 3
**46.05** and about 10 mm partial insertion. Repeating it with the
*original* `trial_991002` identifier again gave partial insertion, so
the trial rename alone does not explain the mismatch with the earlier
eight-trial batch's Tier 3 = 75. The zero-student clone's raw bag is in the
[verified NAS archive](2026-09-24-sc-shared-dagger-teacher-control-mcap-archive.json);
the original-ID replay is archived separately. A two-trial control with
the same `trial_991001` then `trial_991002` order as the original batch
**did reproduce full insertion** in its second trial (Tier 3 = 75, task
duration 44.61 s). The first trial scored 23.87, so it remains excluded.
The second trial's initial TCP differed from the original success by only
about 0.17 mm, compared with about 1.9 mm in the isolated replay. This
strongly implicates reset/predecessor state in the replay discrepancy;
it does not identify the exact simulator mechanism. The two-trial control
bag is in the [verified NAS archive](2026-09-24-sc-shared-dagger-sequence-control-mcap-archive.json).
The collector
code changed between the original batch and these tests, though the new
student branch is inactive in controls; the exact source hashes are in
each evaluation summary. **Do not attribute the isolated partial
insertions to student perturbation.** A matched two-trial run with 3 mm /
0.01 rad student clips and 0.25 selection probability is now running.
These are teacher-assisted data-collection probes, never autonomous
actor evaluation.

That two-trial student run scored **22.11** on the first scene and **75**
on the second, but random cycle selection produced **100 student-active
frames in the failed first scene and zero in the successful second**.
It therefore contributes **no corrective BC rows** despite the second
scene's success. Its exact draw/outcome and raw bags are preserved in the
[verified NAS archive](2026-09-24-sc-shared-dagger-sequence-student-mcap-archive.json).
To obtain an identifiable correction experiment, the collector now accepts
an explicit one-based episode allowlist. The next ordered pair leaves the
warm-up episode teacher-only and activates bounded 3 mm/0.01 rad student
windows only in the second episode; that run is in progress. This is a
collection intervention, not a deployable supervisor or privileged actor.

The second-only run **succeeded in both scenes** (official Tier 3 = 75
each). The first scene was teacher-only and is excluded from the corrective
manifest. The second scene has **400 student-active frames**; each executed
target stayed within 3 mm and 0.01 rad of the teacher target. Its
physical TCP-to-tip transform drifted at most **0.0194 mm** over sampled
scored TF, and the image-synchronous observation force peaked at **22.44
N**. The official force channel reported a separate, brief **56.68 N**
maximum for 0.04 s above 20 N, below the one-second scoring penalty; do
not conflate the two sampling streams. The local admission manifest is
`/var/tmp/chmin_aic_20260918_act/sc_shared_dagger_secondonly_candidates_20260924/manifest.json`;
the [tracked label audit](2026-09-24-sc-shared-dagger-secondonly-labels.json)
records **910 causal four-target chunks**, 317 with a student-active anchor,
and TCP target round-trip error **1.24e-16 m**. The source run is
`/var/tmp/chmin_aic_20260918_act/sc_shared_dagger_secondonly_student_20260924/`,
with exact `run.sh`, per-tick proposals/teacher/executed targets, images,
scored YAML, and [losslessly verified MCAP archive](2026-09-24-sc-shared-dagger-secondonly-mcap-archive.json).
Corrective replays get unique episode IDs even when reusing an expert trial
name; this prevents elapsed-time normalization and sampling from silently
merging the correction with the original expert episode. The two-arm BC
comparison starts from `shared_port_tcp_bc_expanded2_full_20260924/best.pt`
and uses the same 2,750-update budget, seed, architecture, validation
groups, and rootless GPU3 container. Its control has the same 2,750-update
continuation without the corrective episode. Its treatment adds this one
episode at a fivefold **episode sampling multiplier** to test whether
off-expert state supervision affects the guarded development rollout.
This does not establish scene diversity; it is a one-scene intervention.

The matched 2,750-update comparison has completed. Both models have
**13,615,160 parameters** and use the same 39 episode-grouped validation
scenes. The no-correction control selected update 1,750; the fivefold
corrective-sampling arm selected update 2,750. On held-out SC frames at
0–5 mm TCP axial distance, lateral command median/p95 was
**0.243/0.380 mm control** versus **0.446/0.629 mm corrective**
(65 frames, five scenes). SFP at 10–30 mm was **0.646/2.411 mm** versus
**0.698/2.116 mm** (800 frames, 32 scenes). These are teacher-command
errors, not insertion success. On the identical guarded, autonomous
`trial_990501`, **both scored Tier 3 = 0** and the scorer identified
`ur5e::wrist_2_link` against an enclosure collider. The control total was
−23, with no long force penalty. The corrective arm total was **−35**:
force above 20 N persisted for 8.26 s, adding a −12 penalty. Both raw
bags have verified [control](2026-09-24-shared-bc-corrective-control-guarded-sc-mcap-archive.json)
and [corrective](2026-09-24-shared-bc-corrective-treatment-guarded-sc-mcap-archive.json)
NAS archives. **One corrective scene with fivefold sampling did not help
transport and worsened near-port SC command error.** The supervised gate
and RL gate stay closed.

The next bounded diagnosis lowers the corrective episode weight to one,
with the same checkpoint/update budget, while collecting bounded student
corrections across five additional teacher-successful SC scenes. The
previous `trial_990501–990508` perception development suite is now
explicitly retired from **actor** evaluation because its images may enter
supervised training. A fresh eight-start qualification-like SC development
suite (seed 2026092431, trials 995001–995008) is frozen under
`/var/tmp/chmin_aic_20260918_act/shared_bc_corrective_fresh_dev_20260924/`;
its scenes have not been used for actor training or checkpoint selection.
The reserved four final IK configurations remain sealed.

The one-scene **unit-weight** correction ablation also completed the same
2,750 updates and 39-scene held-out evaluation. It selected update 2,750.
SC 0–5 mm lateral command median/p95 improved to **0.182/0.309 mm**;
SC 10–30 mm p95 was **0.448 mm**. SFP 10–30 mm p95 was **2.158 mm**.
The fivefold boost appears to have contributed to the near-port
regression, but these offline metrics still do not establish insertion.
On scored teacher images from the former actor-development `trial_990501`,
the first command of the unit-weight model missed the nearly-zero teacher
target by **14.74 mm**, compared with **8.90 mm** for the no-correction
control. Across its first five sampled teacher frames, median command
error was 10.26 mm. Good near-port validation error can therefore coexist
with a bad initial full-start command. A new supervised
`start_plus_near` ablation keeps the same complete-target architecture
and labels, adds a first-2.5-s command loss, doubles early-episode
sampling, and selects checkpoints using both early command and near-port
lateral p95. It is training on rootless GPU3; this is not RL or a
residual-action architecture.

The start-aware run completed 2,750 updates and selected update 2,500.
On its fixed episode-grouped validation selection, early-command p95 was
**5.92 mm SFP** and **5.99 mm SC**, each over about 90 early rows. On the
former SC teacher scene its first command error fell to **5.30 mm**, down
from 8.90 mm for the no-correction continuation and 14.74 mm for the
unit-weight correction arm. It retains the same 13.615M-parameter
complete-target actor. The separate grouped held-out phase audit gives
SC 0–5 mm lateral command **0.245/0.366 mm median/p95**, versus
0.182/0.309 mm for the unit-weight correction arm; SFP 10–30 mm lateral
p95 is **2.307 mm**. This is a transport/near-port tradeoff in offline
command metrics, not evidence of autonomous insertion. Its exact command
and teacher-frame diagnostic are under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_startaware_20260924/`.

The eight-scene, original-order SC correction collection has completed.
`trial_990502`, `trial_990503`, and `trial_990507` scored full insertion
with bounded student action windows; these three were admitted after
scene identity, causal image/command records, exact 3 mm/0.01 rad target
clips, official force/contact penalties, and dynamic physical-grasp drift
checks. They cover two and four NIC-card obstacles with selected SC port 1.
`trial_990501`, `990504`, `990505`, `990506`, and `990508` did not fully
insert and are excluded regardless of their diagnostic value. The three
accepted episodes contain **1,138** causal four-target chunks, **467** with
student-active anchors, and physical-grasp drift at most **0.0057 mm**;
round-trip TCP target error was **1.59e-16 m**. The [label summary](2026-09-24-sc-shared-dagger-multiscene-labels.json)
and local `sc_shared_dagger_multiscene_candidates_20260924/manifest.json`
preserve per-scene decisions. All eight raw MCAPs were losslessly verified
in the [NAS archive](2026-09-24-sc-shared-dagger-multiscene-mcap-archive.json)
before local removal. The source run with images, action streams, scores,
and exact command is
`/var/tmp/chmin_aic_20260918_act/sc_shared_dagger_multiscene_20260924/`.
The next start-aware BC run adds these three episodes at ordinary episode
weight, with the same initial checkpoint, seed, 2,750-update budget, and
39-scene validation as the one-correction start-aware arm. Both checkpoints
are evaluated autonomously on a newly frozen four-start subset:
`trial_995001`/1 card/SC port 0, `995002`/2 cards/port 0,
`995007`/3 cards/port 1, and `995008`/4 cards/port 1. These starts were
generated and selected before their policy scores were seen.
Four separate SFP-to-NIC development starts (seed 2026092432,
`trial_996001–996004`, one to three present NIC cards) are frozen in
`/var/tmp/chmin_aic_20260918_act/shared_bc_corrective_fresh_sfp_dev_20260924/`.
They will check that the shared actor does not trade SC correction for
SFP transport/insertion. Neither fresh suite is a reserved final scene.

The first autonomous four-start run of the one-correction start-aware
checkpoint scored **Tier 3 = 0/4**. Every trial had a scored
`ur5e::wrist_2_link`–enclosure collision, across both selected SC ports
and one to four NIC-card obstacles. This pattern triggered a runtime-input
audit before adding more transport demonstrations. Task family/target-port
bits matched the training vectors. Using true scored current port pose
only in a posthoc conversion audit, 29 SC teacher targets round-tripped
from port frame to base-frame TCP targets within **2.25e-5 mm** and
**1.77e-6°**. Thus the mathematical converter and task encoding are
consistent on saved labels; actor prediction and input distribution remain
the likely failure points. The direct-resize run's four MCAPs are in the
[verified NAS archive](2026-09-24-shared-bc-startaware-fresh-sc-four-direct-preprocess-mcap-archive.json).

One concrete mismatch was found: training reads a saved 288×256 JPEG,
then uses PIL bilinear resize to 224×224. Live inference had resized the
native camera directly to 224×224 with OpenCV area interpolation. Across
64 paired frames of a scored SC teacher scene, pixel MAE was only about
1.2–1.4/255 per camera, but the same checkpoint's first target moved by
**0.902/3.248 mm median/p95**; an early sample moved 6.54 mm. Passing
the saved native JPEG through a 288×256 area resize, quality-95 JPEG
round trip, and PIL bilinear 224 resize reduced target disagreement to
**0.075/0.256 mm**. A two-stage path without the JPEG round trip was
0.185/0.577 mm. The [per-frame audit](2026-09-24-shared-bc-camera-preprocessing-parity.json)
and `scripts/audit_shared_bc_camera_preprocessing.py` preserve the
comparison. The audit uses saved native JPEGs, not exact original ROS
bytes, so the figures bound evidence from the recorded stream rather
than proving exact live pixel identity. The runtime now reproduces the
training pipeline and logs camera/state/model inference p50/p95/p99.
The same checkpoint is being rerun on the same fresh four SC scenes with
only this preprocessing change. This is a runtime correction; it does
not use simulator geometry, a guide, or a new model.

The matched-preprocessing rerun is complete: **Tier 3 = 0/4 again**, with
the same scored wrist/enclosure contact in each scene. This confirms the
input mismatch was real but **did not cause the main transport failure by
itself**. The corrected live camera/state/model inference p95 values were
**82.76, 98.33, 95.23, and 95.87 ms** (p99 at most 121.46 ms), below
the 300 ms limit. The [four verified raw bags](2026-09-25-shared-bc-startaware-fresh-sc-four-matched-preprocess-mcap-archive.json)
and per-scene policy latency logs are under the matched-preprocess
evaluation directory. The multiscene trained checkpoint is now running
on the same four frozen scenes with this corrected runtime. Its grouped
offline SC early-command p95 improved from **5.99 to 4.55 mm**, but
near-port SC 0–5 mm lateral command p95 regressed from **0.366 to
0.881 mm**; the autonomous comparison will decide whether the extra
corrections help transport.

A route-label audit found a material training distribution split:
among the 25 earlier **SC training** episodes checked, 20 took the
direct route and five used an 8 cm pre-route lift. By frame 100 the
direct-route TCP had moved about 170–245 mm laterally in the selected
port frame, while the lifted route had moved less than about 3 mm
laterally and about 70 mm along the vertical port axis. At frame 20,
the median direct teacher next target relative to current TCP was
**(−31.2, +8.7, +12.3) mm**, versus **(−0.9, −1.4, −9.4) mm** on the
lifted route. These are genuinely different trajectories, so a policy
must maintain a coherent route choice. However, the trained actor on
*expert states* predicted **(−28.5, +8.5, +8.1) mm** for direct scenes
and **(+3.5, −2.4, −9.0) mm** for lifted scenes at that frame. It can
distinguish those two modes once it remains on an expert path; simple
mode averaging at frame 20 is **not established as the cause** of the
autonomous wrist collisions. The [per-episode route and prediction
audit](2026-09-25-sc-transport-route-mode-audit.json) and
`scripts/audit_sc_transport_route_modes.py` preserve the evidence.
Current 3 mm teacher-centered corrective windows may be too narrow to
cover the actor's observed 5–15 mm initial command errors and resulting
off-route states. A larger bounded correction ablation on a repeatable
teacher-success scene is the next data test, with full scoring and
force/contact admission before BC.

**September 25 runtime-action audit.** The multiscene start-aware checkpoint
also scored **0/4 Tier 3 insertions** on the same four fresh SC development
scenes with matched image preprocessing. Two scenes had no off-limit contact,
but the plug ended roughly 0.60 m from the port; the other two had scored
arm/card or wrist contacts. Its [four raw MCAPs](2026-09-25-shared-bc-multiscene-fresh-sc-four-mcap-archive.json)
are verified and archived. Extra 3 mm corrections did not solve full-start
transport. This comparison uses the same runtime, scenes, and 5 mm command
cap as the matched-preprocessing baseline.

The runtime converts each predicted complete port-frame TCP target into a
current-TCP-frame delta, then clips **each translation coordinate to 5 mm**
before sending the base-frame setpoint. The teacher labels are *setpoints*,
not 5 mm increments. In 25 scored SC training episodes (14,528 command
anchors), the teacher target minus measured TCP has a maximum port-frame
coordinate gap of **10.99/36.66 mm median/p95**, and **90.5%** of anchors
exceed 5 mm. During frames 20–40 of transport, the corresponding gap is
**36.10/48.98 mm**, and **100%** exceed 5 mm. The port-frame gap is a proxy
for the runner's body-frame component clamp, not an exact count of clipped
runtime commands; the frames rotate. The [reproducible label audit](2026-09-25-shared-sc-target-cap-audit.json)
and `scripts/audit_shared_sc_target_cap.py` capture the numbers. A 20 mm-cap
autonomous ablation is now running on the same four development scenes,
checkpoint, matched images, and 40 N force stop. This tests whether delayed
tracking contributes to the route failure; it does not assume larger
commands are safe or sufficient. Keep the old 5 mm results frozen.

The 20 mm-cap ablation finished **0/4** with scored off-limit arm/wrist
contacts in all four scenes and a sustained-force penalty in one. Two final
plug/port distances were **1.09 and 1.06 m**; the robot often moved farther
along the wrong route. Its [four verified MCAPs](2026-09-25-shared-bc-startaware-fresh-sc-four-cap20mm-mcap-archive.json)
are archived. Thus the cap mismatch is real, but raising it alone made this
development set worse and is not a candidate runtime fix.

A new scored teacher-only reference on **the same first two development
scenes**, with no student or perturbation, fully inserted on `trial_995001`
(Tier 3=75, no off-limit contact or excessive force). `trial_995002` reached
only about 20 mm from the port (Tier 3=24.83) and had a long force penalty;
it is not a BC success. Its [two verified MCAPs](2026-09-25-shared-bc-fresh-teacher-reference-mcap-archive.json)
and recorded frames/targets are under `shared_bc_fresh_teacher_reference_20260925`.
On the successful first scene, the teacher measured TCP had moved about
**+184 mm in base y after 5 s**, while the 5 mm-capped actor had moved
**−20.5 mm** and the 20 mm-capped actor **−99.3 mm**. By 30 s the teacher
was at **+215 mm**, versus **−263/−362 mm** for those two actor runs.
These measured paths are posthoc replay diagnostics, not actor inputs.

On that exact teacher scene, the actor's first image/state produced a
setpoint **8.8 mm** from the observed TCP while the teacher's first setpoint
was effectively a hold. At 1 s on the teacher trajectory, the actor's
target still differed from the teacher by **20.6 mm**; at 2, 3, and 5 s
teacher states those errors were **9.6, 2.4, and 2.1 mm**. Reusing the
unchanged first image and state but advancing the elapsed-time input to
3 s did **not** create the correct transport: its predicted delta remained
about `(3, −6, +9) mm` in base coordinates. A simple startup pause would
therefore not correct this checkpoint. Across the existing SC label
episodes, the first teacher setpoint is nearly stationary (train median
0.006 mm), but this checkpoint predicts a **3.35 mm median, 7.63 mm p95**
first displacement on those same training starts. The
[first-frame audit](2026-09-25-shared-sc-first-frame-audit.json) separates
25 train and seven grouped-validation episodes. This indicates incomplete
learning of the startup action as well as closed-loop distribution shift;
it does not establish that the model follows safely after a privileged
teacher handoff.

The previous corrective collector admitted student actions only from
3 s onward, missing this observed initial failure. Its intervention start,
end, cycle, and duty window are now configurable with the old 3–35 s
defaults preserved. A bounded 0–35 s student-on-teacher diagnostic with
10 mm positional disagreement limit on the first development scene reached
**partial** insertion at roughly 10 mm (Tier 3=46.05), with no scored
off-limit contact or excessive force. It had student action active at time
zero, but the teacher controlled alternating windows; the outcome is
neither autonomous nor a BC success. Its [verified bag](2026-09-25-shared-bc-early-student-probe-mcap-archive.json)
and all proposed/teacher/executed records remain available. This episode
is excluded from training. A tighter 3 mm early window is the next
diagnostic. In parallel, a same-architecture supervised continuation on
the unchanged scored training set adds a stronger first-0.5-s command loss
and chooses checkpoints using grouped-validation first-command and near-port
errors. It is a controlled response to the measured startup error, not an
RL update.

The tighter **3 mm / 0.01 rad early intervention** on the same development
scene fully inserted (**Tier 3=75**) with no scored off-limit contact or
excessive force, while alternating teacher and student windows. Its
[verified MCAP](2026-09-25-shared-bc-early-student-3mm-mcap-archive.json)
and proposed/teacher/executed records are preserved. It remains excluded
from BC because the scene is an actor development start. This shows a
bounded early correction can yield a clean *teacher-assisted* trajectory;
it does not prove an autonomous actor can recover or choose a route.

The targeted initial-command continuation completed **2,000 updates** on
the same 264 training episodes and 39 grouped-validation episodes; selected
update 250. The actor architecture and 13.615M parameters are unchanged.
On the fresh teacher reference's first image, its predicted target error
fell from **8.8 to 5.6 mm**. On teacher states at 0.5 and 1 s it now missed
the next teacher target by **3.0 and 3.4 mm**, versus the parent model's
20.6 mm at 1 s. These are teacher-state checks. Checkpoint, replay command,
history, and summary: `/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_initial_gate_20260925/`.

Its first unprivileged rollout on the matched scored scene still scored
**Tier 3=0**. There was no scored off-limit contact, but the live force
watchdog stopped at **44.3 N** (score channel reported 30.75 N briefly).
Complete camera/state/model inference p95 was **133.59 ms**, below 300 ms.
The measured base-y TCP displacement after 5 s was **−34.2 mm**, still
opposite the successful teacher's **+183.9 mm**; after 30 s it was
−281.3 mm. The [verified raw bag](2026-09-25-shared-bc-initial-gate-trial995001-mcap-archive.json),
20 Hz controller trace, score, logs, and image snapshots are retained in
`shared_port_tcp_bc_initial_gate_20260925/eval_fresh_sc_trial995001/`.
Thus improved teacher-state targets did not close the on-policy startup gap.
An eight-scene existing **training** suite is now being replayed with bounded
3 mm/0.01 rad student interventions enabled from time zero on selected
episodes. Only independently scored full insertions with causal data will
enter BC. This is the first collection specifically exposing startup
deviation states. The four fresh actor development starts and four sealed
final IK configurations are excluded.

One matched low-learning-rate supervised ablation finished 2,000 updates
from the same start-aware parent using the same scored 264 training episodes,
with a 20× initial-command term, 12× startup sampling, and learning rate
`2e-5`; all other action/actor contracts remain complete-target. The selected
update 250 has grouped-validation **SC initial-command p95 2.78 mm** and
**SC near-port lateral-command p95 0.289 mm** on the fixed validation sample;
the preceding `initial_gate` selection had **6.08/0.702 mm** on that same
sample. SFP initial-command p95 is **4.06 mm** and near-port p95 **2.10 mm**.
Its `best.pt`, `run.sh`, history, and summary are under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_initial_low_lr_20260925/`.
These are offline teacher-state numbers only; autonomous scoring awaits the
active correction collection.

A scored-TF **posthoc** audit on the fresh teacher's first image found a
coupled-head tradeoff: the parent actor's TCP-in-opening pose-head error
was **3.21 mm**, while the two action-loss continuations were **14.0 mm**
(`initial_gate`) and **8.16 mm** (lower learning rate). The label geometry
was used only after inference to measure this error, never as an actor
input. A model can lower its *complete command* error on teacher frames
by changing its predicted target and predicted port/TCP pose together;
those components need not remain separately accurate in a closed loop.
The [exact frame, scored TF, checkpoint paths, and errors](2026-09-25-shared-sc-fresh-start-pose-audit.json)
are preserved. This correlation does not prove pose drift alone caused the
wrong autonomous route. A controlled continuation now freezes the visual
and pose encoder from the parent while fitting only the complete-target
action head with the stronger startup loss. Frozen BatchNorm statistics
are held in evaluation mode during that training. This tests whether
the new action target can improve without degrading the pose estimate.

That frozen-encoder continuation finished 2,000 updates and selected
update 1,250. **All 139 frozen visual/context/pose tensors matched the
parent bit for bit**. Grouped SC initial-command p95 was **4.34 mm**,
start-command p95 **4.86 mm**, and near-port lateral-command p95
**0.702 mm**. SFP initial and near-port p95 were **4.60 and 2.60 mm**.
The result preserves pose-head behavior but is weaker offline than the
lower-learning-rate unfrozen actor; its first target on the fresh teacher
frame still moved **−4.41 mm in base y**, opposite the later teacher
transport. It remains an unpromoted comparison under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_frozen_pose_start_20260925/`.

The first early-correction training-suite replay was **invalidated by scratch
exhaustion** near `trial_990507`: its final score YAML was not written,
the seventh bag metadata is damaged, and the eighth trial never started.
Engine logs retain first-six per-trial totals, including >86 on trials
990502/990503, but a total alone is insufficient for the official Tier 3,
force, and contact admission gate. No episode from that incomplete run has
entered BC. Its seven raw MCAP byte streams are [losslessly archived](2026-09-25-sc-shared-early-dagger-incomplete-mcap-archive.json)
alongside images, proposals, teacher/executed commands, and failure logs.
Two inactive old bags were copied to NAS with SHA256 verification and
replaced by same-path symlinks to create safe scratch margin; the
[relocation record](2026-09-25-sc-old-bag-space-relocation.json) preserves
their exact paths. A fresh scored replay will keep the original reset order
through the seven relevant scenes and require a complete final score file
before candidate extraction.

The seven-scene retry completed with a valid official score file. Bounded
student-influenced `trial_990501`, `990503`, and `990507` each scored
**Tier 3=75**, with no scored off-limit contact or excessive-force penalty.
`990502`, `990504`, `990505`, and `990506` were partial/no-insertion and
are excluded. The three successes passed the exact 3 mm/0.01 rad
teacher-disagreement bound, complete causal RGB/command records, 80 N
force ceiling, and episode-specific physical-grasp drift ceiling; maximum
drift among them was **0.0183 mm**. They contain **1,725** causal
four-target training chunks and **1,063** student-active recorded frames;
TCP target round trip was **1.78e-16 m**. The [label summary](2026-09-25-sc-shared-early-dagger-retry-labels.json),
candidate manifest, and [seven verified MCAPs](2026-09-25-sc-shared-early-dagger-retry-mcap-archive.json)
preserve admitted and excluded episodes. The successful scenes belong to
the actor training side, with no overlap with the grouped validation
scene hashes. Labels now carry the *actual episode start simulation time*
so a chunk whose first valid anchor is frame 1 or 5 retains its true
startup time; prior correction sources started at frame 0 and are unchanged.
One GPU3 matched continuation is running with and without these new
corrections from the same low-learning-rate parent, seed, update budget,
and grouped validation. A separate low-learning-rate checkpoint is being
tested autonomously on the fresh development start with no teacher or
geometry input. Neither teacher-assisted data collection nor offline BC
scores count as autonomous insertion.

That low-learning-rate parent scored **Tier 3=0**, total −23 on the matched
fresh autonomous SC scene and incurred an off-limit wrist/enclosure contact.
Its [raw scored MCAP](2026-09-25-shared-bc-initial-low-lr-trial995001-mcap-archive.json)
is archived. Lower grouped offline initial-command and near-port errors
therefore did not transfer to this closed-loop route. The controlled
frozen-pose-encoder checkpoint also scored **Tier 3=0**, total −23,
with a scored wrist/enclosure contact on that same autonomous scene.
Its [verified bag](2026-09-25-shared-bc-frozen-pose-trial995001-mcap-archive.json)
is archived. Keeping the pose head fixed did not independently repair
the full-start route. The matched early-correction control has completed
1,500 updates; the treatment is training, and both will be scored using
the same fixed development scene and runtime settings.

The matched 1,500-update control and early-correction treatment both
completed from the same lower-learning-rate parent, seed, validation
sample, and learning rate `1e-5`. Control selected update 1,500 on 264
training episodes; treatment selected update 1,250 on 267 episodes.
Grouped SC initial-command p95 was **2.47 mm control / 3.47 mm treatment**,
while SC near-port lateral p95 was **0.516 / 0.407 mm**. SFP initial-command
p95 was **3.92 / 2.88 mm**. The treatment thus does not win every offline
component. The autonomous control on the fixed fresh SC start scored
**Tier 3=0**, total −23, with a wrist/enclosure contact; its
[MCAP archive](2026-09-25-shared-bc-early-dagger-control-trial995001-mcap-archive.json)
is verified. The treatment's same-scene rollout is active. Exact matched
commands, histories, checkpoints, and scores are under
`shared_port_tcp_bc_early_dagger_{control,treatment}_20260925/`.

The treatment also scored **Tier 3=0**, total −23, and the same scored
wrist/enclosure contact on that fixed autonomous development start. Its
[verified bag](2026-09-25-shared-bc-early-dagger-treatment-trial995001-mcap-archive.json)
preserves the failure. Three 3 mm teacher-bounded successes improved some
offline components but did **not** establish useful autonomous startup
correction. The next causal probe lets the latest observation-only actor
control only the **first 1 s**, clips its proposed absolute target within
8 mm/0.02 rad of the teacher, then returns fully to teacher control on a
training-side scene with a scored teacher-only full insertion. A successful
probe would show the teacher can recover from a larger *initial* deviation
and may yield useful supervision there; it would not be an autonomous
insertion. Failure would limit this DAgger route and push us to a different
full-start representation or explicit transport structure.

The larger startup probe has now completed. On training-side
`trial_990501`, an 8 mm/0.02 rad teacher-bounded student pulse at nominal
expert time 0–1.0 s actually lasted **5.6 simulation seconds and 21 recorded
commands** because the collector advances its nominal teacher clock by
0.05 s per command while simulation advances more rapidly. The teacher then
controlled the rest; official Tier 3 was **16.88, no insertion**, with the
plug about 80 mm from the port. A 6 mm/0.02 rad pulse at nominal 0–0.5 s
lasted **3.2 simulation seconds and 11 commands**; the teacher recovered to
partial insertion (Tier 3 **46.05**, about 10 mm remaining), but not full
insertion. Neither had scored off-limit contact or sustained force penalty.
These runs are **not admitted as successful BC episodes**. They bound this
teacher-assisted collection route on one scene; variation across replay and
the changed pulse size/duration prevent assigning a unique causal effect to
clip alone. Exact commands, frames, score, and run scripts are under
`/var/tmp/chmin_aic_20260918_act/sc_early_pulse_{8mm_training_probe,6mm_halfsec_training_probe}_20260925/`;
raw bags are [8 mm](2026-09-25-sc-early-pulse-8mm-mcap-archive.json) and
[6 mm](2026-09-25-sc-early-pulse-6mm-halfsec-mcap-archive.json) SHA-verified
NAS archives. The earlier 3 mm alternating teacher-assisted success did not
improve autonomous insertion after matched BC training, so further data
collection needs a new causal strategy rather than treating these partial
recoveries as positive labels. This remains step 5, not an RL result.

A separate **observation-only diagnostic startup hold** kept the same
trained actor and fresh development scene `trial_995001`, but commanded the
measured TCP to hold for the first 3 simulation seconds before returning to
actor control. This still scored **Tier 3=0, total −23**, with the same
wrist/enclosure collision class. The run and exact 3-second flag are in
`shared_port_tcp_bc_early_dagger_treatment_20260925/run_fresh_hold3.sh` and
`eval_fresh_sc_trial995001_hold3/`; the [bag](2026-09-25-shared-bc-startup-hold3-trial995001-mcap-archive.json)
is preserved. This rules out the simple diagnosis that only the very first
movement caused the failed route. The hold is a diagnostic intervention and
is not a promoted runtime rule. The more general SC transport representation,
route choice, and closed-loop recovery remain unresolved.

A further bounded diagnostic continued the same shared actor for 1,500
updates while changing the SC sample fraction from 50% to 80%; the other
labels and objective stayed the same. Its selected update was 1,000. Grouped
held-out SC initial-command p95 improved from **3.47 to 2.46 mm**, near-port
lateral p95 remained **0.407/0.417 mm**, and SFP initial-command p95 was
**2.88/2.71 mm**. The same fresh autonomous `trial_995001` still scored
**Tier 3=0, total −23**, with wrist/enclosure contact. The checkpoint,
replay command, and validation history are under
`shared_port_tcp_bc_sc80_diagnostic_20260925/`; the
[scored bag](2026-09-25-shared-bc-sc80-trial995001-mcap-archive.json) is
archived. A [direction audit](2026-09-25-shared-sc80-transport-direction-audit.json)
of 242 held-out SC *teacher-state* frames from seven complete episodes at
3–15 s found median predicted/teacher port-frame displacement cosine
**0.998**, zero opposite-direction rows, and 1.18/2.65 mm command-vector
error median/p95. This rules out a simple train-time direction-sign bug on
those held-out teacher states. It does **not** show that the policy stays on
course on the fresh autonomous scene; scene generalization and compounding
state error are both plausible. Increasing the SC sampling fraction alone
has not passed the BC gate. A new four-scene training-only 3 mm bounded
student collection (`sc_shared_early_dagger_newbatch1_20260925/`) has
completed: **3/4** scenes reached scored Tier 3=75 and one no-insertion scene
was excluded. The three full successes had no scored off-limit contact or
sustained force penalty, at most 0.020 mm physical-grasp drift, all three
cameras, strictly causal command/image order, and 1,726 numerical-round-trip
four-target labels. Their manifest and labels are under
`sc_shared_early_dagger_newbatch1_candidates_20260925/`; the
[label audit](2026-09-25-sc-shared-early-dagger-newbatch1-labels.json) and
[SHA-verified raw bags](2026-09-25-sc-shared-early-dagger-newbatch1-mcap-archive.json)
are recorded. Two more fresh four-scene training batches are in progress or
prepared before the next matched BC fit.

The second new four-scene batch (`sc_shared_early_dagger_newbatch2_20260925/`)
scored **4/4 Tier 3=75**, with no scored off-limit contacts or force penalty.
All four were admitted after the same RGB/timing/teacher-clip/physical-grasp
checks; maximum measured grasp drift was **0.028 mm**. Its candidate manifest
and 2,043 four-target chunks are under
`sc_shared_early_dagger_newbatch2_candidates_20260925/`; see the
[label audit](2026-09-25-sc-shared-early-dagger-newbatch2-labels.json) and
[verified bag archive](2026-09-25-sc-shared-early-dagger-newbatch2-mcap-archive.json).
The first two new batches add seven successful training-side correction
episodes. A third fresh four-scene batch is running before a matched
same-checkpoint, same-update SC-weighted BC comparison. These are
teacher-assisted collection scores, never autonomous policy insertions.

The third new batch (`sc_shared_early_dagger_newbatch3_20260925/`)
scored **2/4** full insertions. The other two (Tier 3 **19.95** and **24.10**;
one had a force penalty) are excluded. The two successful episodes passed
all camera, causal teacher-clip, contact/force, and grasp checks; maximum
grasp drift was **0.012 mm** and their
[label audit](2026-09-25-sc-shared-early-dagger-newbatch3-labels.json)
contains 1,090 four-target chunks. All four
[bags](2026-09-25-sc-shared-early-dagger-newbatch3-mcap-archive.json)
are verified and archived. Across 12 fresh training-only scenes, **9 full
successes and 4,859 causal chunks** were added; three non-full episodes
were excluded. A matched 1,500-update BC comparison from the same SC-weighted
parent is now running, with identical seed, replay, validation, and optimizer
settings except these nine added successes. The control and treatment commands
are `shared_port_tcp_bc_newsc_{control,treatment}_20260925/run.sh`.

The matched control and nine-episode treatment have now both finished
1,500 updates and selected update **250**. Control/treatment grouped SC
initial-command p95 was **2.764/2.821 mm** and near-port lateral p95
**0.281/0.561 mm**; SFP initial-command p95 was **3.485/2.988 mm**.
On the same fresh autonomous SC `trial_995001`, both scored **Tier 3=0**
with wrist/enclosure contact. Control total was **−35**, including sustained
force penalty; treatment total was **−23**, without that sustained penalty.
Their [control](2026-09-25-shared-bc-newsc-control-trial995001-mcap-archive.json)
and [treatment](2026-09-25-shared-bc-newsc-treatment-trial995001-mcap-archive.json)
bags are SHA-verified. The added 3 mm bounded correction episodes changed
some force behavior but did **not** produce autonomous transport/insertion.
The next supervised investigation must target generalization under the
actor's own full-start state distribution or a route representation; repeating
small-error BC collection alone is not justified by this match. Step 5 and
the RL gate remain failed.

A **clock-alignment audit** found another concrete inconsistency in step 5.
The SC collector's `nominal_expert_time` advances 0.05 s per command, while
camera simulation time can advance several times faster; e.g. SC command 51
was at **2.55 s nominal** but **12.45 s simulation elapsed**. Historical
SFP labels already use 0.05 s per frame. The earlier time-aware actor was
trained on SFP nominal time and SC simulation time, then fed simulation time
at deployment. The new `--time-coordinate nominal_decision` reader takes
actual SC `nominal_expert_time` from its per-episode recorded frame (23,399
SC rows checked) and SFP frame/20; the runtime supplies issued-command/20,
with checkpoint metadata selecting the clock. Old checkpoints preserve their
simulation-time path. This changes only the causal time input, not geometry,
labels, or episode splits.

A matched 1,500-update nominal-clock continuation from the same SC-weighted
parent selected update 1,000. Against the simulation-clock control, grouped
SC initial-command p95 improved **2.764→2.156 mm**, SC near lateral p95
changed **0.281→0.352 mm**, and SFP initial p95 improved **3.485→2.604 mm**.
The same fresh autonomous `trial_995001` still scored **Tier 3=0, total −23**
with a wrist/enclosure-wall contact. Its exact run is under
`shared_port_tcp_bc_nominal_clock_20260925/` and its
[bag](2026-09-25-shared-bc-nominal-clock-trial995001-mcap-archive.json)
is preserved. This fixes a real clock contract but is not sufficient for
transport. A matched nominal-clock run with the nine new successful SC
episodes selected update **1** (the continuation worsened its combined
validation criterion); its autonomous same-scene run also scored **Tier 3=0, total −23** with
wrist/enclosure contact; its [bag](2026-09-25-shared-bc-nominal-newsc-trial995001-mcap-archive.json)
is preserved. Neither clock variant promotes BC or opens RL. A scored
privileged diagnostic now tests an unbounded observation-only actor takeover
after five **nominal** seconds of teacher transport, with the actor limited
to its usual 5 mm body-translation cap. The teacher prefix is clearly
separated from actor-controlled frames; this is not autonomous evidence.

The early teacher-prefix takeover finished. The teacher controlled the
first **100 commands / 5.0 nominal seconds** (6.9 simulation seconds in this
replay). The observation-only nominal-clock actor then directly controlled
**671 frames**, with a 5 mm body-action cap. An exact frame audit found
**0 mm** difference between recorded student proposals and executed target
translations during actor-owned frames, confirming that the teacher target
was only logged after takeover. Official Tier 3 was **0.47** (no insertion),
with final plug distance about **0.19 m**, wrist/card off-limit contact, and
a sustained-force penalty. The
[verified bag](2026-09-25-shared-bc-takeover-after5nominal-mcap-archive.json)
and images are under `shared_bc_takeover_after5nominal_20260925/`.
This is a privileged teacher-initialized test, not autonomous success; the
handoff occurred during early transport, so it does not yet localize the
failure to transport versus alignment. A second handoff after 20 nominal
seconds is running on the same development scene and actor.

The **20-second nominal teacher-prefix handoff** scored Tier 3 **75** with
no off-limit contact or force penalty. The actor directly owned **371**
commands from frame 399 onward; recorded executed targets matched its raw
proposals exactly. The `/scoring/insertion_event` wall timestamp was **87.3 s
after handoff**, so the event occurred during actor ownership. However,
scored physical-tip geometry at handoff was already about **13.8 mm inside
the opening**, with lateral error about **0.08 mm**. The tip reached roughly
14.9 mm axial depth later under the actor. The teacher therefore did almost
all transport, alignment, and insertion before the handoff. This result shows
that the actor can finish the last roughly 1 mm from a highly favorable
teacher state; it does **not** show autonomous approach or a broad insertion
controller. The [geometry audit](2026-09-25-shared-bc-takeover-after20nominal-geometry-label-audit.json),
[bag](2026-09-25-shared-bc-takeover-after20nominal-mcap-archive.json),
frame-level student ownership, and run script are preserved. A 15-second
nominal handoff is now running to test approach before axial insertion.

The **15-second nominal handoff** failed: official Tier 3 **5.15**, no
insertion, with a scored wrist/NIC-card contact. At actor takeover frame 300,
the physical plug tip was **20.8 mm before** the port opening and **0.62 mm
laterally off**. After 100 actor-owned commands, lateral offset was about
**56 mm**; terminal lateral offset was about **157 mm**. All **871**
actor-owned executed target translations matched student proposals exactly,
so teacher clipping did not cause this divergence. The
[geometry audit](2026-09-25-shared-bc-takeover-after15nominal-geometry-label-audit.json)
and [bag](2026-09-25-shared-bc-takeover-after15nominal-mcap-archive.json)
are preserved. An [offline checkpoint replay](2026-09-25-shared-bc-takeover-after15nominal-pose-action-audit.json)
on these images found TCP-in-opening pose-head error **5.4 mm** at frame 302, **32.5 mm** by frame 320, and
**87.3 mm** by frame 400; complete-command error likewise grew from
**2.0 mm** at frame 302 to **39.9 mm** by frame 320. This is direct evidence
of a perception/action runaway on the actor's own states, not a successful
near-port controller. A repeat retaining original-resolution cameras every
fourth frame is running to test whether the separate frozen high-resolution
SC estimator remains accurate on these off-policy views.

The original-resolution repeat retained **293** native camera triplets and
again failed after the 15-second handoff, with a scored off-limit contact.
Its [raw bag](2026-09-25-shared-bc-takeover-after15nominal-native-mcap-archive.json)
is SHA-verified on NAS while the original-resolution images remain local.
The frozen observation-only RGB locator and refiners were run without
retraining. In valid camera views during frames 300–319 (just after
handoff), the global locator's opening-pixel error was **0.9/2.2 px
median/p95**; the causal fused physical plug-to-opening lateral error was
**0.14/0.14 mm** across five retained frames. In frames 320–399, as the
actor diverged, locator opening error rose to **17.2/41.1 px** and fused
relative lateral error to **0.16/8.69 mm** median/p95. After frame 400,
the opening was visible in far fewer views and the crop often missed it;
the aggregate near-port p95 of 298 mm is dominated by this late drift,
not by the handoff. The [native label audit](2026-09-25-shared-bc-takeover-native-pose-label-audit.json),
[frozen port trajectory](2026-09-25-shared-bc-takeover-native-port-tracker.json),
and [relative pose report](2026-09-25-shared-bc-takeover-native-relative-pose.json)
retain per-frame values. These use ground-truth geometry only for scoring.
The physical TCP-to-tip offset changed by up to **0.49 mm** over this failed
rollout, so late physical-tip label errors may include grasp slip; the
handoff comparison precedes the large off-route motion.

A **nondeployable true-TCP-pose substitution** in the existing frozen action
head reduced command-vector error at actor frame 320 from **39.85 to
8.22 mm**, but worsened some other frames (e.g. frame 297 from 0.83 to
8.32 mm). The [exact frame audit](2026-09-25-shared-bc-takeover-after15nominal-pose-action-audit.json)
shows that the action head and pose head are coupled; simply swapping a
better pose into the unadapted head is not a reliable controller. A bounded
supervised oracle-pose **training-only** upper bound is running before
implementing an observation-only external-pose input.

The nominal-clock **training-only oracle TCP-pose continuation** selected
update 250 and worsened grouped SC near-port lateral p95 from **0.352 to
0.999 mm**. On the failed takeover images, exact true-pose input reduced
command-vector error at frame 320 from about **40 to 6.4 mm**, but error was
worse at frame 297 (**0.8 to 9.2 mm**) and frame 349 (**21.3 to 42.7 mm**).
The [upper-bound audit](2026-09-25-shared-bc-takeover-oracle-pose-trained-audit.json)
shows accurate pose alone does not reliably fix the current action mapping.
This checkpoint is explicitly nondeployable and no RL or final evaluation is
opened. The next bounded data ablation targets actor-owned off-policy states,
which were absent from clean expert BC and only weakly represented by the
3 mm teacher-clipped episodes. Four previously successful **training-side**
SC scenes are now being rerun with teacher control through 15 nominal seconds
and then direct observation-only actor takeover. From these runs, only short
actor-owned, low-force, near-port rows with recorded teacher correction
targets will enter a **separate local-correction experimental source**.
A failed whole trajectory is never reclassified as a verified expert
success. Selection may use scored geometry for training only; autonomous
actor inputs and crop selection remain observation-only. The exact selectors
and thresholds are in `scripts/build_sc_safe_local_corrections.py`.

### September 25 direct-takeover and local-correction outcome

Four additional training-side SC starts used the same teacher for the first
15 nominal decision seconds and then passed **unmodified actor proposals**
to the simulator (student cap 5 mm; teacher cap 20 mm). All four failed full
insertion and scored off-limit contact plus sustained-force penalty. The raw
bags are [losslessly archived](2026-09-25-sc-unbounded-takeover-training-mcap-archive.json).
They are not expert successes. Scored static port transforms, RGB, measured
TCP, force and exact proposal/execution were retained. The dynamic physical
grasp reconstruction failed on a missing sub-10 ms TF at one late frame, so
the local selector uses a fixed tip proxy and conservative early-handoff
gates; see the [label audit](2026-09-25-sc-unbounded-takeover-local-label-audit.json).

The first selector rejected every row because normal pre-handoff force was
already about 20 N; it was corrected to compare force against each episode's
preceding one-nominal-second baseline while also enforcing an absolute 35 N
limit, <8 N force rise, first two nominal seconds after takeover, and <25 mm
measured TCP displacement from handoff. One scene with a 64 N baseline was
excluded. This left **25 safe local chunks in three episodes**, with teacher
versus actor target discrepancy **39.21 mm median, 66.21 mm p95**. The exact
[selection report](2026-09-25-sc-safe-local-corrections.json) and
`/var/tmp/chmin_aic_20260918_act/sc_unbounded_takeover_training_20260925/safe_local_corrections.jsonl`
identify each row. Whole failed episodes remain outside the expert manifest.

Matched 500-update continuations from the nominal-clock parent used the same
seed, episode split, validation, and replay except these 25 rows:
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_local_correction_{control,local}_20260925/`.
Both selected update 500. Grouped held-out SC initial-command p95 was
**2.966 mm control / 2.825 mm local**, but near-port lateral p95 worsened
**0.304 / 0.360 mm**. Both direct autonomous runs on new SC `trial_995001`
scored **Tier 3=0**. The control scored total −23 with wrist/enclosure
contact; the local arm scored no insertion with a 34.85 N transient force
below the penalty duration. Their bags are [control](2026-09-25-shared-bc-local-correction-control-trial995001-mcap-archive.json)
and [local](2026-09-25-shared-bc-local-correction-local-trial995001-mcap-archive.json).
The separate [control](2026-09-25-shared-bc-local-correction-control-takeover-audit.json)
and [local](2026-09-25-shared-bc-local-correction-local-takeover-audit.json)
failed-handoff replay audits show little command improvement; e.g. frame 320
error **39.9→37.7 mm**, frame 400 **135→136 mm**. This 25-row correction is
far too small relative to 151k training rows to claim failure coverage. A
bounded 15× local-row sampling ablation is being run to test that specific
sampling explanation without adding failed episodes to the expert set.

That 15× ablation completed 500 updates from the same nominal-clock parent.
Its grouped held-out SC initial-command p95 was **2.806 mm**, but near-port
lateral p95 worsened to **0.492 mm**. The failed-handoff replay improved frame
320 command error **37.7→26.6 mm** compared with unit sampling, while frame
400 remained **136.7 mm**. On the identical fresh autonomous SC start it
still scored **Tier 3=0, total −23**, with wrist/enclosure contact and a
38.61 N transient force. See [offline frame audit](2026-09-25-shared-bc-local-correction-boost15-takeover-audit.json),
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_local_correction_boost15_20260925/`,
and [verified scored bag](2026-09-25-shared-bc-local-correction-boost15-trial995001-mcap-archive.json).
Sampling helps one local command, not the later distribution shift or safe
full-start transport. The next training-side probe executes a direct actor
takeover only from 15 to 17 nominal seconds, then restores the privileged
teacher. Its full episode must score successful before any whole-trajectory
BC admission; actor-owned rows remain distinctly marked. This is a
teacher-rescue data probe, not autonomous success.

The four-scene rescue probe completed with official Tier 3 scores
**21.78, 75, 75, 75**. Trial 996101 also incurred a sustained-force
penalty and is excluded. The other three had no scored off-limit contact or
sustained-force penalty. Each used **41 consecutive direct actor commands**
from nominal 15 to 17 s, then teacher control; actor-target discrepancy
reached 69, 88, and 38 mm on those three scenes. This is materially farther
off the teacher path than the earlier 3 mm clipped examples. The exact scene
configuration, four `episode_*/frames.jsonl` files, and scored YAML are under
`/var/tmp/chmin_aic_20260918_act/sc_bounded_takeover_rescue2s_20260925/`;
the [four MCAPs are SHA-verified on NAS](2026-09-25-sc-rescue2s-mcap-archive.json).
The bag-to-episode wall-clock and TF joins passed for all four. On the three
successes, sampled scored physical TCP-to-tip drift stayed **0.0073, 0.0459,
0.0203 mm**. The successful-scene manifest at
`/var/tmp/chmin_aic_20260918_act/sc_bounded_takeover_rescue2s_candidates_20260925/manifest.json`
requires exact actor proposal=executed during takeover and excludes the
failed trial. A causal-hold builder generated **2,046** four-target TCP
chunks, with negligible target round-trip error (see [label audit](2026-09-25-sc-rescue2s-labels.json)).
They are marked teacher-rescued corrective supervision, **not clean expert
or autonomous insertions**. A matched 1,500-update BC control/treatment from
the nominal-clock parent used identical seed, validation, fivefold
corrective-episode sampling, and update count; the only replay change was
these three scored rescue episodes. Control selected update 500, rescue
update 250. Held-out SC initial-command p95 improved **2.714→2.505 mm**,
but near-port lateral p95 worsened **0.470→0.568 mm**. A replay of the
earlier failed actor handoff showed essentially no large-error improvement:
frame 320 command error **40.0→39.2 mm**, frame 400 **135.5→136.4 mm**
([control](2026-09-25-shared-bc-rescue2s-control-takeover-audit.json),
[rescue](2026-09-25-shared-bc-rescue2s-rescue-takeover-audit.json)).
Autonomously on fresh SC `trial_995001`, **both Tier 3=0**; control finished
total **+1** without scored off-limit contact, while rescue finished **−23**
after wrist/enclosure contact. The [control](2026-09-25-shared-bc-rescue2s-control-trial995001-mcap-archive.json)
and [rescue](2026-09-25-shared-bc-rescue2s-rescue-trial995001-mcap-archive.json)
bags are archived. These three teacher-rescued trajectories did not meet
the autonomous BC gate. An active-frame 15× sampling ablation is running;
the separate early-takeover rescue probe shifts the actor's two-second
window to the episode start, where autonomous transport first diverges.
No failed whole episode is admitted as expert data.

The **early direct-takeover** version moved the same 41 actor-owned commands
to nominal time 0–2 s, then restored the teacher on the same four training
scenes. Official Tier 3 was **75, 46.05, 24.17, 42.57**; only trial 996101
fully inserted. None scored off-limit contact or sustained-force penalty.
Its physical TCP-to-tip transform changed only **0.0004 mm** at sampled
times. The exact scored TF/episode joins passed. The one successful scene
yielded **522** causal chunks, with exact actor proposal/execution checked;
three failed scenes remain excluded from the corrective expert candidate
set. The source and candidate manifests are under
`/var/tmp/chmin_aic_20260918_act/sc_early_takeover_rescue2s_20260925/`
and `sc_early_takeover_rescue2s_candidates_20260925/`, respectively; see
the [label audit](2026-09-25-sc-early-rescue2s-labels.json) and
[verified bag archive](2026-09-25-sc-early-rescue2s-mcap-archive.json).
This is evidence that two nominal seconds of direct start motion is often
too much for the teacher to recover on this small scene set. A 0.5-second
direct-start window on the same training scenes is being scored before an
early-state BC comparison.

The **0.5 nominal-second direct-start** version scored Tier 3
**20.97, 75, 12.11, 23.49**. Only trial 996102 fully inserted, with no
scored off-limit contact or sustained-force penalty. Its scored physical
TCP-to-tip drift was **0.0043 mm** at sampled times. The admission audit
accepted only that trial, yielding **515** causal chunks; the [label
report](2026-09-25-sc-early-rescue0p5s-labels.json) and [verified four-bag
archive](2026-09-25-sc-early-rescue0p5s-mcap-archive.json) preserve all
outcomes. Across the 2.0 and 0.5 s early variants there are only **two
full-success teacher-rescued episodes, 1,037 chunks**, and 52 actor-owned
frames. A 1,500-update BC continuation from the nominal-clock parent used
these two sources and 20× actor-owned-row sampling. Grouped validation
selected **update 1** (selection score 3.647 mm); every later checkpoint
was worse, e.g. SC initial-command p95 **4.286 mm** at update 1 and
**5.129 mm** at update 1,500. The matched no-addition control selected
update 500 with SC initial p95 **2.714 mm**. The early-data treatment fails
the offline checkpoint gate; it is not promoted to autonomous evaluation.
This does not mean early correction is useless: two scene-diverse successes
and 52 actor-owned rows are too few for a robust full-start controller, and
aggressive repetition may overfit them.

The three near-port teacher-rescued episodes were also tested with **15×
sampling of the 96 actor-owned anchor chunks**. Held-out SC initial-command
p95 improved to **2.413 mm**, near lateral p95 to **0.436 mm**, but the
fresh autonomous SC trial still scored **Tier 3=0, total −23**, with
wrist/enclosure contact. Its [failed-handoff replay](2026-09-25-shared-bc-rescue2s-activeboost15-takeover-audit.json)
retained 136 mm command error at frame 400; the [scored bag](2026-09-25-shared-bc-rescue2s-activeboost15-trial995001-mcap-archive.json)
is archived. Additional sampling of these few examples did not solve the
closed-loop route.

The next upper-bound ablation trains the **same action head with true SC
TCP-to-port pose supplied in every phase**, with 0.3 mm training noise.
This is strictly a privileged training/diagnostic input, never deployable.
It asks whether correct geometry during early transport can make the current
complete-target action architecture learn the teacher route. If even this
upper bound fails on held-out teacher states and scored off-policy images,
the next implementation should change action learning and scene diversity
before integrating the high-resolution estimator into the actor. The
checkpoint and commands are under
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_sc_allpose_oracle_20260925/`.

The all-phase true-pose upper bound completed 1,500 supervised updates and
selected update 1,500. **Despite receiving ground-truth SC TCP-to-opening
pose throughout transport**, grouped held-out SC initial-command p95 was
**4.817 mm**, compared with **2.156 mm** for the observation-only
nominal-clock parent; near-port lateral p95 was **0.920 vs 0.352 mm**.
This is a diagnostic architecture/data comparison, not deployable evidence.
On the frozen failed-handoff development images, the true-pose-conditioned
command error was **110.1 mm at frame 0**, **6.3 mm at frame 320** and
**43.7 mm at frame 349**; the nominal actor's corresponding ordinary
errors were **9.7, 39.9, 20.4 mm**. The exact
[oracle](2026-09-25-shared-bc-takeover-oracle-allphase-trained-audit.json)
and [nominal](2026-09-25-shared-bc-takeover-nominal-allphase-control-audit.json)
frame audits, `run.sh`, checkpoint and grouped validation history preserve
this failure. Correct geometry can fix an individual off-policy command,
but by itself it does not teach this head a safe route on an unseen scene.
The two early teacher-rescued successes are also too sparse and unstable to
cover full-start distribution shift. The next supervised step is to expand
scored SC route/scene diversity in bounded batches, retaining only full
successes as expert demonstrations, before another actor comparison.

Before spending more simulator time, two existing scored candidate groups
were independently re-audited. The prior **16-scene card-count matrix pilot**
has six full successes: four zero-card controls, one three-card/port-1 and
one five-card/port-1. Its other ten trials stay excluded. Their physical
grasp calibration, three-camera frames, causal commands, scene hashes,
target round-trip and force/stall checks passed, yielding **2,806** causal
chunks ([label audit](2026-09-25-sc-bc-matrix-pilot-physical-labels.json),
[admission](2026-09-25-sc-bc-matrix-pilot-admission.json)). The zero-card
settings are deliberate training controls outside the qualification-like
generator's support, not claims about production coverage. A separate
four-card lift success and two five-card/port-0 successes from prior route
probes also passed the same checks, yielding **2,228** chunks
([labels](2026-09-25-sc-lift-fivecard-physical-labels.json),
[admission](2026-09-25-sc-lift-fivecard-admission.json)). The roots
`sc_bc_matrix_pilot_candidates_20260925/` and
`sc_lift_fivecard_candidates_20260925/` retain manifests and exact source
hashes. **Provenance correction:** all nine episode IDs and their exact
2,806+2,228 frame counts were **already present** in the original
`sc_supplemental_candidates_v2_20260924/port_tcp_waypoint_chunks.jsonl`
used by the shared actor. The previous admission audit checked only against
the May verified manifest, so it did not detect overlap with later SC
supplemental sources. A nominal-clock 1,500-update `control9`/`diverse18`
comparison was run before this discovery; the treatment duplicated these
nine scenes and is **invalid as a diversity experiment**. It must not be
used to infer generalization. `train_shared_port_tcp_bc.py` now rejects a
duplicate `(family, episode, frame)` across all source files. The matched
control with nine earlier teacher-bounded successes selected update 1 and
failed the offline gate, consistent with the previous nominal-clock
continuation. A new eight-scene, one-GPU qualification-like clean-teacher
batch was collected next and admitted only after scoring and causal audit.

The fresh eight-scene clean-teacher batch scored **5/8** full insertions.
The three failures were retried on the *same training scenes* with an 8 cm
pre-route lift; **two more** fully inserted, while the four-card/port-1 scene
still failed. The first five provided **2,463** unique causal chunks and
the lift retry added **1,222**, for **seven new successful SC episodes and
3,685 chunks**. All seven have scored physical-grasp calibration, complete
three-camera RGB, causal command/image/TF join, negligible TCP target
round-trip error, no overlap with every SC source already used by BC, and
passed force/stall admission. The full source runs, exact engine configs,
image frames, and [first-batch](2026-09-25-sc-diversity-batch1-mcap-archive.json)
and [lift-retry](2026-09-25-sc-diversity-batch1-lift-mcap-archive.json)
SHA-verified bags remain available. The exact [first-batch
admission](2026-09-25-sc-diversity-batch1-admission.json) and [lift-retry
admission](2026-09-25-sc-diversity-batch1-lift-admission.json) list each
episode and label SHA. Failed starts were excluded.

A matched 1,500-update nominal-clock continuation compared the original
267 training episodes with +5 new clean episodes; the +7 arm then added
the two scored lift successes under identical seed, validation, weights,
and update budget. Control/+5/+7 selected updates **1500/250/1250**.
Held-out SC initial-command p95 was **3.124/2.876/2.169 mm**;
near-port lateral p95 was **0.312/0.466/0.446 mm**. The +7 arm improved
teacher-state early commands, but an independent failed-handoff replay
still had about **40 mm** command error at frame 320 and **134 mm** at
frame 400, nearly unchanged from control
([control](2026-09-25-shared-bc-freshdiversity-control-takeover-audit.json),
[five](2026-09-25-shared-bc-freshdiversity-new5-takeover-audit.json),
[seven](2026-09-25-shared-bc-freshdiversity-new7-takeover-audit.json)).
All three actors scored **Tier 3=0, total −23** on the same new autonomous
SC `trial_995001` with wrist/enclosure contact; their scored bags are
[control](2026-09-25-shared-bc-freshdiversity-control-trial995001-mcap-archive.json),
[five](2026-09-25-shared-bc-freshdiversity-new5-trial995001-mcap-archive.json)
and [seven](2026-09-25-shared-bc-freshdiversity-new7-trial995001-mcap-archive.json).
No promotion or RL follows. These data improve supervised agreement but
do not teach the current actor to recover from its own large route error.

Two bounded supervised follow-ups are running from this result. First, the
same +7 data and architecture are training **from ImageNet initialization**
for up to 10,000 updates with grouped validation and early stopping; this
checks whether the previous 1,500-update warm continuation was trapped by
its old solution. The exact command is
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_scratch_new7_20260925/run.sh`.
Second, the +7 shared actor is taking **direct observation-only control**
for the first two nominal seconds on five new *training-side* clean-teacher
success scenes, with a 2 mm per-command cap, after which the privileged
teacher resumes. The exact scene subset and command are under
`/var/tmp/chmin_aic_20260918_act/sc_diversity_actor_early2mm_base_20260925/`.
Only scored full recoveries with exact proposal/execution and low-force
causal joins can become corrective BC examples. The intervention scenes
remain distinct from the new development start and sealed final split.

The from-scratch +7 fit stopped at update **7,000** after its selected
update **5,000** failed to improve for 2,000 updates. Its grouped held-out
SC initial-command p95 was **2.118 mm**, close to the warm +7 result
**2.169 mm**, but near-port lateral p95 was much worse (**1.114 vs
0.446 mm**) and the combined selection score was **4.425 vs 3.015 mm**.
This is an offline failure for a longer ImageNet-initialized fit; it is
not promoted to autonomous scoring. The exact 10,000-update cap,
patience, logs, selected checkpoint, and `summary.json` are in
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_scratch_new7_20260925/`.
Insufficient optimization time in the 1,500-update continuation alone is
not supported as the primary explanation for the closed-loop failure.

The five-scene direct early-control probe with the +7 actor and a **2 mm
per-command cap** scored official Tier 3 **18.49, 46.05, 75, 75, 0**.
Trials 997003 and 997005 fully inserted after teacher rescue; the three
failures remain excluded. The two successes had sampled physical
TCP-to-tip drift **0.0121/0.0006 mm**, 82 actor-owned frames, exact
actor-proposal/executed-pose agreement, and **1,387** causal target chunks.
Their [label audit](2026-09-25-sc-diversity-actor-early2mm-labels.json),
candidate manifest under
`/var/tmp/chmin_aic_20260918_act/sc_diversity_actor_early2mm_candidates_20260925/`,
and [all five scored bags](2026-09-25-sc-diversity-actor-early2mm-mcap-archive.json)
preserve success and failure evidence. This is teacher-rescued corrective
data on training scenes already represented by clean teacher episodes,
not five new scene groups or autonomous actor insertions. A matched
1,500-update continuation from the +7 checkpoint is running with and
without these two episodes. Both use the same seed, grouped validation,
sampling, and fivefold actor-owned-row weight; only the corrective source
differs. The autonomous gate remains closed pending a fresh-scene score.

That matched comparison completed. Control/corrected selected updates
**750/1000**; grouped held-out SC initial-command p95 was **2.791/2.525 mm**,
near-port lateral p95 **0.319/0.312 mm**, and combined selection score
**3.189/2.838 mm**. On the separate failed-handoff replay, frame 400
command error improved only **134.1→127.3 mm**, still far outside a safe
route ([control](2026-09-25-shared-bc-early2mm-control-takeover-audit.json),
[corrected](2026-09-25-shared-bc-early2mm-corrected-takeover-audit.json)).
Both actors scored **Tier 3=0, total −23** on the same fresh autonomous SC
`trial_995001`. Control contacted the wrist/enclosure; corrected contacted
the forearm/enclosure wall. The raw bags are verified in the
[control](2026-09-25-shared-bc-early2mm-control-trial995001-mcap-archive.json)
and [corrected](2026-09-25-shared-bc-early2mm-corrected-trial995001-mcap-archive.json)
manifests. The 82 actor-owned training frames improved held-out
teacher-state predictions but not safe full-start insertion. A direct
early autonomous command/pose trace is the next diagnosis before adding
more corrective data or changing the actor architecture.

The autonomous runner previously issued all four predicted chunk targets
before the next RGB observation, whereas the actor-takeover collector always
reobserved after the first. An explicit `--shared-replan-every-command`
ablation used the same nominal-clock checkpoint and the same fresh SC
`trial_995001`; it also scored **Tier 3=0, total −23**, with the same
wrist/enclosure contact. Complete image/state/model inference latency was
**68.42/107.27/152.87 ms p50/p95/p99** over 286 calls, within the 300 ms
limit. Its [scored bag](2026-09-25-shared-bc-replan1-trial995001-mcap-archive.json)
is archived. This execution cadence mismatch is real but does not explain
the current autonomous failure by itself.

The distinct **nondeployable true physical-tip input** continuation used the
same nominal-clock 267-episode source, 0.3 mm injected tip noise for SC rows
within 30 mm of the opening, and 1,500 updates. It selected update 500;
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_nominal_sc_tip_oracle_20260925/`
contains `run.sh`, `train.log`, `summary.json`, `history.json`, and
`best.pt`. At the failed direct-takeover frames, the frozen checkpoint was
replayed with **true physical tip only for this offline upper-bound audit**,
using the same <30 mm gate as training. At frame 320, command error stayed
**39.4 mm** despite true tip input; frame 349 was **20.2 mm**. The
[frame audit](2026-09-25-shared-bc-takeover-oracle-tip-trained-audit.json)
records mask and errors. This cannot establish the deployable high-res pose
route, and it gives no reason to promote geometry to the actor yet. The
off-policy frozen estimator's own [relative-pose report](2026-09-25-shared-bc-takeover-native-relative-pose.json)
shows handoff lateral error ~0.14 mm, early actor median ~0.16 mm but p95
8.69 mm, and late actor median 8.42 mm after target loss. At handoff the
visual estimate was already adequate; closed-loop action distribution shift
is the more immediate gate. All RL branches remain disabled.

Step 4 preflight: the official qualification-like SC scene generator always
instantiates **at least one NIC card** because it unconditionally samples a
`target_nic` even for SC-to-SC tasks. A 1,000-scene SC-only generator sample
contained no zero-card case, while four- and five-card cases were rare.
`/var/tmp/chmin_aic_20260918_act/sc_bc_matrix_pilot_20260924/` contains a
prepared, **not yet run**, 16-scene scored CheatCode pilot: two starts for
each missing `(NIC-card count, selected SC port)` cell `(0,0)`, `(0,1)`,
`(2,1)`, `(3,1)`, `(4,0)`, `(4,1)`, `(5,0)`, `(5,1)`. The zero-card scenes
remove all NIC entities from otherwise qualification-like one-card scenes;
the source manifest records this explicit intervention and both generator
seeds. This is a data-collection probe, not a successful-expert or RL dataset
until scores, frames, and action causality are checked.

The 16-scene pilot has now completed and scored. CheatCode fully inserted
in **6/16**: four zero-card controls (both SC ports), one three-card/port-1,
and one five-card/port-1. It failed both two-card/port-1, all four-card,
both five-card/port-0, and the other three-card/port-1 and five-card/port-1
starts. All ten failures remain outside BC. Nine of those failures scored
no insertion with the final plug 30–80 mm from the port; one five-card/
port-0 trial reached only about 10 mm partial insertion. Off-limit contact
scoring found none, though several trials briefly exceeded 20 N force.
The terminal executed TCP remained roughly 38–96 mm from the teacher's
target on the failures. This is a large route/tracking stall, **not** proof
of a specific cable/card contact pair from Gazebo alone. The
[scored capture audit](2026-09-24-sc-bc-matrix-pilot-capture.json), source
manifest, full images, and per-tick teacher/executed records are under
`/var/tmp/chmin_aic_20260918_act/sc_bc_matrix_pilot_20260924/`.
The full-success candidates still need scored-TF/action and frame audit
before admission to an expert manifest. Bounded route retry or backtracking
for the failed high-card cells remains open; no shared BC/RL training began.

An [8/12 cm pre-route-lift probe](2026-09-24-sc-lift-route-probe.json)
used the same scored scenes and the privileged collector's target logic.
On four NIC cards with SC port 1, an 8 cm lift before transport raised
Tier 3 from **17.4 to 75**; the successful repeat has exact teacher/issued
targets, complete RGB files, and physical-tip TF sampled at the beginning,
middle, and end. On the five-card/port-0 scene, stock/8 cm/12 cm variants
scored **17.7/16.4/16.9**. The 12 cm repeat did not resolve that route.
These runs are stored under the `sc_lift_route_probe_20260924` and
`sc_lift12_route_probe_20260924` roots in `/var/tmp/chmin_aic_20260918_act/`.
They do not yet establish that the lift generalizes or that an autonomous
actor can choose a safe route. The successful four-card repeat remains a
candidate until its causal chunks and physical plug labels are finalized.
The same 8 cm teacher lift also turned one four-card/port-0 matched scene
from Tier 3 **23.2 to 75**; its matched two-card/port-1 scene remained a
no-insertion failure. A separate [three-start five-card/port-0
search](2026-09-24-sc-fivecard-newstarts-lift.json), generated from the
qualification-like profile after all earlier scenes, produced **two full
insertions and one no-insertion**. This supplies candidate successes for
that previously empty task cell; their bag/image/action audits are pending.
Three new two-card/port-1 starts are being tested with the same lift. None
of these privileged route probes demonstrates autonomous recovery or a
general route policy.
An attempted five-card target swap from the successful port-1 scene was
**invalid**: that scene has no port-0 mount, so the teacher returned false
after 99 frames and Tier 3 was zero. The artifacts remain under
`sc_port_swap_lift_probe_20260924`, but this is an unavailable-target
configuration, not evidence of another route failure and not training data.

Storage: keep bulky runs under local `/var/tmp/` or ignored `outputs/`, and
small machine summaries and this record in Git. The workspace filesystem has
only about 4 GB free as this run begins; do not duplicate large bags or
checkpoints there. Rootless Docker setup is documented at
`/data1/chmin/code/ws_aic/ROOTLESS_DOCKER_GUIDE.MD` on this host.

## September 25: full-start failure traced to both perception and action

The latest corrected shared BC checkpoint was allowed to control a fresh SC
development scene from the first command, with privileged CheatCode targets
recorded **only for posthoc diagnosis**. It scored Tier 3 = 0, total = −35,
with wrist/enclosure contact and sustained-force penalty; the terminal plug
was about 0.78 m from the opening. Its scored bag is SHA-verified in the
[archive manifest](2026-09-25-shared-corrected-fullstart-teacher-trace-mcap-archive.json).
The exact run command is in
`/var/tmp/chmin_aic_20260918_act/shared_corrected_fullstart_teacher_trace_20260925/run.sh`.
No diagnostic label entered actor observations, crop selection, or commands.

The [per-frame action audit](2026-09-25-shared-corrected-fullstart-pose-action-audit.json)
uses the checkpoint's **nominal decision clock**, scored geometry only after
the run, and an image/state/task-only actor. On frame 0, the actor's own
auxiliary TCP-to-port pose is wrong by 28.21 mm and 8.45°, and its first body
translation differs from the teacher by 5.81 mm. By frames 20/25/35, pose
errors are 62/95/167 mm and first-command translation errors are 44/84/141
mm. The error grows during actor-driven motion; grouped teacher-state BC
metrics did not predict this closed-loop behavior.

The frozen native RGB port locator is also **not a full-route estimator**.
On a separate, previously retained fresh-scene native-image trace, it misses
the visible port opening in the initial side views by roughly 350–390 px,
then reaches subpixel median error near the port. The exact phase/visibility
counts and p95 are in the [phase audit](2026-09-25-sc-global-opening-locator-phase-diagnostic.json).
Its training source had only 324/4,758 frames with the plug more than 200 mm
from the opening, and the previous reported high accuracy was dominated by
nearer views. This explains why the high-resolution crop system cannot yet
be assumed to guide transport.

Perception alone does **not** repair the current frozen action head. Replacing
only its predicted port pose with exact scored pose in an offline,
nondeployable calculation changed frame-0 first-command error from 5.81 to
24.36 mm; on later drifted frames it helps but leaves large errors (for
example frame 35: 141.1 to 26.8 mm). The action and pose heads have learned
coupled biases, so this is an information ablation, not a deployable oracle
result or evidence that true geometry should enter the actor. A separate
far-view RGB locator training ablation completed on GPU 3, using only
training-scene simulator projections as labels, scene-grouped validation,
10× sampling of visible far views, and far-view checkpoint selection. The
selected update 1,400 achieved 1.80 px median / 14.60 px p95 opening error
on 178 held-out **far** views in the training collection; near-view opening
p95 was 5.69 px. The checkpoint, exact command, metrics, and predictions are
under `/var/tmp/chmin_aic_20260918_act/sc_global_far_locator_gpu_20260925/`.
The fresh scene was absent from all fit and selection scenes. On its native
RGB trace, initial frames 0–40 had left/right visible-opening errors of
**3.3/3.8 px median and 4.6/9.8 px p95**, versus the old locator's
**382/348 px median**. Transport frames 40–120 were 1.0–1.4 px median.
Late drifted near views remained worse (p95 29–37 px). See the
[fresh phase audit](2026-09-25-sc-far-weighted-locator-fresh-phase-audit.json).
This establishes a useful **full-route visual localization ablation**, but
pixel localization alone is not a calibrated 3D pose input. A posthoc
calibrated triangulation test exposed an important visibility effect:
including all three cameras at initial frames gives **103.8 mm median**
lateral port error because the opening is outside the center view but the
locator still emits an in-frame center guess. Using the **fixed left/right
camera pair**, selected without geometry or scored labels, reduces initial
lateral error to **1.2 mm median / 2.3 mm p95**, with axial p95 **4.2 mm**.
Transport frames 40–120 improve to 0.4/0.7 mm lateral median/p95. The
old locator with the same fixed side pair had 125.7/140.5 mm initial
lateral median/p95. Full phase values, per-frame records, calibrated camera
constants, and scoring geometry provenance are in the matched
[old](2026-09-25-sc-global-old-triangulation-side2.json) and
[far](2026-09-25-sc-global-far-triangulation-side2.json) reports; the
[all-three report](2026-09-25-sc-global-far-triangulation-all3.json)
preserves the failed view-fusion ablation. Near-port frames after runaway
still have **6.1 mm p95** lateral error with side cameras, so this does not
meet the 0.5 mm insertion corridor. The old near-port causal tracker remains
a separate, more accurate near regime, and a switch must be based on visual
confidence or measured state, not scored geometry.

A further **single-scene diagnostic** used the fixed side-view pair and a
causal median of up to 40 accepted RGB triangulations for the static port.
The acceptance rule uses predicted two-view reprojection residual ≤3 px,
≤5 mm change from the current visual anchor, and ≥0.5 s between accepted
updates. It has no scored-geometry or teacher input and permits observations
while the plug is far away. On this same fresh trace, it initialized before
the near-port phase and gave **0.134/0.236 mm near lateral median/p95**;
the unfiltered side pair had 0.389/6.113 mm. Initial p95 was still 2.338 mm.
This filter was chosen after examining that scene and is **not independent
validation**. Its [per-frame report](2026-09-25-sc-global-far-triangulation-side2-causal40.json)
is preserved; the frozen locator/filter pair is now being checked on eight
already collected, episode-grouped development scenes that had no overlap
with locator training. The four reserved final IK configurations remain sealed.

That eight-scene frozen check **failed the far-route gate**. With the same
side-view pair and causal filter, near-port lateral error was 0.301/1.624
mm median/p95 over 411 frames (seven scenes reached the near interval),
versus 0.397/58.19 mm without filtering. The filter initialized in all
seven near episodes, but `trial_950007` retained a 1.62 mm bias. More
decisively, initial pose errors were very large in six of eight scenes.
The scored projection audit shows the selected SC opening was **outside
both side-camera images** at frame 0 in six of eight yaw-edge starts; in
several it was outside all three. This is an actual observation limit for
the visible-opening locator, not just a training weakness. Exact rows are
in the [raw](2026-09-25-sc-global-far-eight-raw.json) and
[causal](2026-09-25-sc-global-far-eight-causal40.json) reports. No policy
promotion follows. A bounded training-only test now supervises projected
offscreen target pixels from the visible board silhouette on separate
training scenes; the untouched eight scenes will test whether that inference
generalizes. If it does not, transport must first move to a safe view where
the opening becomes observable, using an observation-only route strategy.

The offscreen extrapolation ablation **did not pass**. It trained on 36
scene-grouped training configurations from the existing yaw-augmented native
capture, with 10× far-view sampling and projected offscreen labels allowed
only when the port remained in front of a camera. Update 600 was selected
before patience stopped the run at 1,300. On its 12 held-out validation
scenes, offscreen-opening error was **57.3/286.2 px median/p95** across 108
views, and all far-opening p95 was 222.1 px. Frozen inference on the eight
yaw-edge scenes missed the offscreen initial target by 114–524 px in the
affected cameras; only the two scenes with the target visible in all views
had single-digit errors. The checkpoint and exact metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_global_offscreen_locator_gpu_20260925/`,
and eight-scene predictions under
`/var/tmp/chmin_aic_20260918_act/sc_global_offscreen_locator_eight_20260925/`.
This small direct regression model is not a reliable offscreen port pose
estimator. It is a negative bounded ablation, not proof that board-based
geometry or active perception is impossible.

The next isolated **training-only upper bound** starts a shared actor from
ImageNet initialization with exact SC pose conditioning during BC. This is
not deployable: true geometry is supplied only to test whether the action
head can learn the full SC route when pose information is perfect. The prior
1,500-update true-pose continuation from a biased actor failed; a from-scratch
comparison is needed before committing to an observation-only pose interface.
Its exact command and checkpoints are in
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_scratch_allpose_oracle_20260925/`.

The true-pose scratch upper bound finished all 6,000 updates and selected
update 5,000. It used the **same 154,730 training rows, 22,683 grouped
validation rows, seed, architecture, sampling, and loss weights** as the
earlier scratch +7 control; only the SC pose conditioning changed. Its
selection score was **5.852 mm versus 4.425 mm** for the observation-only
control. SC initial-command p95 was **5.18 versus 2.12 mm** and near-port
lateral p95 **1.93 versus 1.11 mm**. This upper bound is not deployable and
did not pass even an offline action gate, so no autonomous trial or pose
interface promotion is justified. The action mapping remains a separate
problem from visual localization. A 2 mm command-cap autonomous diagnostic
on the same already used development scene is being run to test whether
smaller closed-loop steps arrest the early runaway; it is not a final score.

The 2 mm cap diagnostic completed on the same development scene and did
**not** arrest failure: Tier 3 = 0, total = −35, wrist/enclosure contact,
47.11 s above 20 N, maximum force 79.17 N. The [matched posthoc trace](2026-09-25-shared-corrected-fullstart-cap2mm-pose-action-audit.json)
shows slower but continuing drift: at frame 20 pose/first-command errors
were 23.6/31.3 mm; at frame 40 they were 94.7/119.5 mm. The 5 mm-cap
reference had 62.4/44.1 mm at frame 20 and 134.6/134.8 mm at frame 40.
The same scene's fixed scored geometry was reused only in the posthoc audit;
the actor received RGB/state/task, and the archived raw bag is in the
[cap-2 mm manifest](2026-09-25-shared-corrected-fullstart-cap2mm-mcap-archive.json).
Command magnitude limiting alone cannot teach the missing route correction.

The shared scratch training set contains **274 training episodes in total,
but only 38 are SC-to-SC**: 14 canonical, 11 supplemental, two first SC
expansion, four scored corrective, and seven fresh diversity successes.
It also has 39 validation episodes across both families. The much larger
268-episode SFP source does not provide SC cable route labels. Thus the
SC autonomous failure is consistent with sparse route diversity, especially
when the target is initially offscreen. More scene-grouped successful SC
Gazebo data and teacher-rescued off-policy states are a justified supervised
next step; the few corrective episodes so far did not suffice. New collection
must keep score, image, target, force, and physical grasp audits and admit
only verified full insertions. No RL begins until the full-start BC gate.

A quick route-mode check on frame 20 of the 36 available SC training
episodes with that frame found a **dominant initial motion direction**:
33/36 first targets moved negative body-frame Y, 29/36 negative X, and
29/36 positive Z. Median body-frame translation was approximately
`[-8.0, -29.4, +9.6] mm` to the first target. The three positive-Y
exceptions are lift-route successes. This check does not rule out later
multimodal recovery, but it argues against blaming the **initial** runaway
primarily on regression averaging between left/right route modes. More SC
start coverage and closed-loop corrective supervision remain higher priority
than a mixture head at this stage.

The next SC teacher collection uses the official qualification-like
generator with seed `202609252`, trial IDs `998001`–`998020`, and the
privileged CheatCode **only to collect training demonstrations**. Exact
generator output, scene config, source manifest, `run.sh`, and gated
`postprocess.sh` are in
`/var/tmp/chmin_aic_20260918_act/sc_diversity_batch2_20260925/`.
The postprocess script will check scored full success, causal commands and
three RGB streams, scored physical grasp, target frame roundtrip, force
stall, split/scene/episode overlap, and then SHA-archive MCAPs. No episode
from this run is admitted until that pipeline passes.

Before inspecting its outcome, six **new development-only** SC starts were
reserved with seed `202609253`, trials `999001`–`999006`: two 1-card,
two 2-card, and two 3-card cases, both SC target ports represented. Their
exact scene hashes and engine SHA are in the
[reservation](2026-09-25-shared-bc-new-dev6-reservation.json). They are
sealed from BC gradient updates and checkpoint selection; they are separate
from the four reserved final IK configurations. New 4/5-card development
starts will be needed for the later rare-card promotion gate.

The first 20-scene teacher run stopped **after six scored trials** while
spawning `trial_998007`: Gazebo's `ros_gz_container` exited with code −11
immediately after cable attachment, before simulator readiness. The engine
had completed trial 998006 but never wrote a final batch scoring YAML.
Its six raw MCAPs are SHA-verified in the
[crashed-prefix archive](2026-09-25-sc-diversity-batch2-crashed-prefix-mcap-archive.json)
and are **not admitted** without full official score/causal checks. Free
disk and memory were ample (~23 GB and ~975 GB available), so ordinary
out-of-space/OOM is unsupported. The exact failure log remains under
`sc_diversity_batch2_20260925/eval_collection/collection_config/attempt_0001/logs/`.
The unchanged generated scenes were split into score-complete chunks
`part01_03`, `part04_06`, `isolated07`, `part08_11`, `part12_15`, and
`part16_20`; each has its own exact engine config, collection config, and
rootless `run.sh`. `isolated07` is running first to determine whether the
segfault is scene-specific. A repeated crash will be preserved and excluded;
other chunks can still finish independently. No partial score is inferred
from the crashed batch.

`isolated07` completed normally with engine return code 0 and official Tier 3
**20.47** (partial insertion); it is **not a BC success**. This falsifies a
deterministic scene-7 spawn failure. The segfault may reflect accumulated
Gazebo state across resets; no definitive root cause is claimed. The
remaining unchanged scenes are running in short, independent chunks with a
new engine instance for each and a machine-readable completion status. Each
chunk must finish and write its official scoring YAML before any of its
episodes can be considered for BC.
`sc_diversity_batch2_20260925/postprocess_chunks.sh` records the exact
planned TF/physical-grasp/causal-label/admission/archive pipeline for the
completed chunks. It refuses an incomplete engine result before reading a
bag, preserving the crashed prefix as separate diagnostic evidence.

All six short runs completed with engine return code 0. The original 20
scenes yielded **8 full Tier-3 insertions** (four per SC target port) and
12 partial failures. The exact scored TF, image/command causality, physical
grasp, port-frame roundtrip, force/stall, and all-source split/duplicate
checks admitted **8 episodes and 3,945 chunks**, with no failed route in
BC. The [admission report](2026-09-25-sc-diversity-batch2-admission.json),
[label report](2026-09-25-sc-diversity-batch2-labels.json),
`sc_diversity_batch2_candidates_20260925/physical_waypoint_chunks.jsonl`,
and six SHA-verified `2026-09-25-sc-diversity-batch2-*-mcap-archive.json`
manifests preserve the result; the original crashed prefix is separate.
The 12 failed original scenes now have a **distinct training-only 8 cm
pre-route lift retry** with new trial IDs `998101`–`998112`. The
`sc_diversity_batch2_lift_retry_20260925/retry_mapping.json` fixes the
one-to-one original scene/score mapping, and three independent four-scene
rootless batches are running. Only newly scored full insertions will be
considered, with scene overlap checked against the eight admitted originals.

All 12 lift retries finished with official scores. **Two** previously
partial scenes became full Tier-3 successes (both port 0); ten remained
partial. The clean teacher/physical-grasp/force/causality and all-source
overlap gates admitted **2 episodes and 1,755 chunks**. See the
[lift admission](2026-09-25-sc-diversity-batch2-lift-admission.json),
[lift label audit](2026-09-25-sc-diversity-batch2-lift-labels.json), and
three `2026-09-25-sc-diversity-batch2-lift-part*-mcap-archive.json`
manifests. The new batch therefore adds **10 unique SC successes and 5,700
causal chunks**, raising the SC training episode count from 38 to 48 in
the treatment arm. This is still limited route coverage, and 8 cm lift
rescued only 2/12 failures in this batch; it is a data-collection variant,
not a deployable recovery claim.

A matched 1,500-update shared BC continuation is running from the same +7
parent as the already completed no-batch control. The only treatment change
is adding these 10 verified SC episodes; seed, task encoding, one actor,
action representation, SC sampling fraction, losses, validation groups,
and update budget are identical. Control root:
`shared_port_tcp_bc_early2mm_control_20260925/`; treatment root:
`shared_port_tcp_bc_diversity_batch2_treatment_20260925/` (both under
`/var/tmp/chmin_aic_20260918_act/`). The pre-registered six development
starts remain sealed until the offline checkpoint is chosen. No RL.

The matched treatment completed 1,500 updates and selected update 250;
the no-batch control had selected update 750 from its own 1,500-update
run. Grouped validation selection score improved only **3.189→3.089 mm**.
SC initial-command p95 improved **2.791→2.565 mm**, while near-port lateral
command p95 worsened **0.319→0.544 mm**. The treatment therefore does
**not** pass the 0.5 mm near-port offline threshold, and no insertion claim
follows. Exact run scripts, checkpoints, logs, and summaries are in the
matched roots above. A bounded autonomous development diagnostic has been
pre-registered on `trial_999001` (2-card, port 0) and `trial_999003`
(3-card, port 1), with the same first-target replan runtime for both arms.
The remaining four reserved development starts and final four IK starts
remain unopened. Evaluation uses RGB/state/task only, with no CheatCode,
simulator geometry, guide, or exploration input. The exact command is
`sc_shared_bc_new_dev6_20260925/eval_new_dev2.sh`.

Both actors completed those two **new, identical autonomous development
scenes** with no guide, exploration, privileged geometry, or teacher input.
Control and treatment each scored **Tier 3 = 0/2 and total −23 on both
trials**, including the same category of contact penalty. The exact
[control](2026-09-25-shared-bc-diversity-batch2-dev2-control-mcap-archive.json)
and [treatment](2026-09-25-shared-bc-diversity-batch2-dev2-treatment-mcap-archive.json)
MCAPs are SHA-archived; both full engine score files remain in their run
roots. The added 10 SC successes improved one teacher-state offline metric
but did not repair autonomous full-start routing. The treatment failed the
near-port offline threshold as well, so the four other reserved development
starts are left unopened until a materially different supervised fix is
ready. **Step 5 remains failed; no RL or final evaluation.**

### September 25: SC sampling diagnostic and broader teacher collection

A bounded SC-heavy continuation tested whether the treatment had merely
underweighted its 48 SC training episodes. It resumed the same checkpoint,
rows, and 1,500-update budget at 99% SC sampling (the trainer requires a
fraction strictly below 1). The exact command and selected checkpoint are in
`/var/tmp/chmin_aic_20260918_act/shared_port_tcp_bc_diversity_batch2_sc_only_probe_20260925/`.
The selected update was 250. On grouped held-out SC rows, first-command p95
improved from **2.565 to 1.995 mm** and near-port lateral command p95 from
**0.544 to 0.409 mm**. This is an offline diagnostic, with degraded SFP
balance. The identical two already-open autonomous development trials
`999001` and `999003` each scored **Tier 3 = 0**, with wrist/enclosure
contact; total was **−46** (−23 per trial). The second trial ended 0.90 m
from its target. Exact scored results and logs are at
`.../eval_new_dev2_replan1/best/attempt_0001/`; the two raw bags have
[verified archive hashes](2026-09-25-shared-bc-sc99-dev2-mcap-archive.json).
This rejects sampling weight alone as a fix. It is not evidence against
SC training or against an actor trained on a broader set of successful
full-start routes.

A further teacher data arm generated **60 new training-only SC scenes**, seed
`202609254`, trials `999101`–`999160`, using the released
`qualification_eval_like` generator. The manifest and exact scene config are
under `/var/tmp/chmin_aic_20260918_act/sc_diversity_batch3_20260925/`.
`scripts/run_sc_bc_expansion_chunks.py` captures four-scene scored groups and
automatically isolates any group whose engine did not complete. Each group
has its own config, command array, logs, and raw bags. Collection does not
admit failures to BC: full score, causal command/image alignment, scene split,
and calibration checks remain mandatory before any new fit. The four
unopened development starts and four final IK starts remain sealed. Step 5
remains failed; no RL or reserved-final evaluation.

### September 25: scored SC route expansion and five-card retry

The new broad source completed all **60/60** frozen training scenes in 15
four-scene Gazebo groups. The unchanged privileged teacher scored **31 full
insertions** and 29 partial/failed attempts. Full insertions by NIC-card
count and target were: one card **6/9 port 0, 9/11 port 1**; two cards
**5/9, 6/16**; three cards **3/6, 1/6**; four cards **0/0, 1/1**;
five cards **0/1, 0/1**. The exact scene IDs, scores, source seed, and raw
bag archive pointers are in the [scored source report](2026-09-25-sc-diversity-batch3-scored.json).
All 31 full attempts passed the admission checks for causal three-camera
frames, exact teacher/executed commands, stable physical grasp, distinct
scene splits, and port-frame TCP target round trip. They yielded **17,796**
new chunks; see the [admission report](2026-09-25-sc-diversity-batch3-admission.json).
No partial attempt was admitted.

A separate frozen five-card source sampled three scenes per SC target port.
With the unchanged teacher it scored **0/6** full insertions (Tier 3 about
15–24), as recorded in the [attempt table](2026-09-25-sc-fivecard-teacher-attempts.json).
An exactly matched teacher-only **80 mm pre-route lift** replay used new trial
identities on those same six scenes. It raised three port-0 scenes to full
Tier 3=75; the remaining one port-0 and all three port-1 scenes stayed
partial. The [paired scores](2026-09-25-sc-fivecard-lift-scored.json)
and raw bag hashes retain both arms. The three full retries passed the same
admission gate and yielded **2,287** causal chunks; see the
[lift admission](2026-09-25-sc-fivecard-lift-admission.json). The lift is a
teacher data-collection intervention here, not a deployable actor or proof
of hierarchical backoff. Five-card/port-1 route diversity remains thin.

A matched shared-actor comparison is now running from the completed batch-2
treatment checkpoint: the control continues for 1,500 updates on its prior
data; the treatment uses the same parent, architecture, optimizer, seed,
update budget, and SFP source plus these **34** new admitted SC episodes.
The roots are `shared_port_tcp_bc_batch3_{control,treatment}_20260925/`.
The control finished and selected update 250 with offline selection 2.772 mm.
The treatment trained on 318 total episodes (82 SC), selected update 750,
and had **worse** offline selection **3.182 mm**; SC near-port lateral
command p95 was **0.399 mm** versus control **0.294 mm**. The same two
already-open autonomous development starts scored **Tier 3=0/2 for both
arms**, each with wrist/enclosure contact and total **−46**. The control
ended 0.86/0.90 m from the port and the treatment 0.87/0.88 m. The exact
[machine comparison](2026-09-25-shared-bc-batch3-matched-gate.json),
[control bags](2026-09-25-shared-bc-batch3-control-dev2-mcap-archive.json),
and [treatment bags](2026-09-25-shared-bc-batch3-treatment-dev2-mcap-archive.json)
preserve this negative result. Additional successful SC routes alone did
not fix autonomous full-start routing. A teacher replay of these *already
opened development scenes* is being recorded for posthoc pose/action
diagnosis only; its frames are forbidden from BC. The four unopened
development starts and four reserved final IK starts remain sealed. No RL
or supervisor promotion.

### September 25: expanded BC full-start diagnosis

The same two already-open development scenes were replayed under CheatCode
strictly for posthoc diagnosis. `trial_999001` remained partial (Tier 3 =
24.95); `trial_999003` fully inserted (Tier 3 = 75). Neither replay enters
BC or model selection. Raw bags are in the [archive manifest](2026-09-25-shared-bc-dev2-teacher-diagnostic-mcap-archive.json),
and the frozen actor audits are in the [summary](2026-09-25-shared-bc-batch3-dev2-teacher-diagnostic-summary.json).
On the successful teacher scene at frame 0, control/treatment actor pose
errors were 44.34/41.81 mm, although command translation errors were only
0.47/1.62 mm. Replacing predicted pose in the frozen action head with scored
true pose worsened command errors to 44.19/40.98 mm. Pose and action errors
have coadapted; a better pose cannot simply be plugged into this head.

On 60 new training-only scenes, the selected SC opening was outside the
center camera on 58/60 first frames and visible in at least one of the three
cameras on only 26/60. Both diagnostic starts had an offscreen center
opening. The previous keypoint soft-argmax mathematically limited its
output to the image rectangle, invalidating that architecture as a test of
offscreen extrapolation. An optional unbounded offset head fixed this
representation limit. It was trained with simulator projection labels on
90 training scenes and selected on 12 held-out scenes. At update 700 its
offscreen opening error was **22.46 native pixels median / 118.78 p95**
over 70 held-out views; its two diagnostic first-frame center errors were
**201.46 and 35.78 pixels**. It still cannot reliably localize offscreen
openings. The exact command, checkpoint, metrics, and predictions are under
`/var/tmp/chmin_aic_20260918_act/sc_center_offscreen_offset_locator_batch3_20260925/`.

The earlier frozen visible-opening side-view locator found the first
diagnostic start within **4.68/4.80 native pixels** in left/right images,
but missed the second by **112.33/98.29 pixels**. Its predictions are under
the same root's `dev2_old_far_eval/`. A bounded three-camera far-view
training ablation on expanded training-only scenes is the next test. No
scored geometry entered autonomous inference; step 5, RL, supervisor
promotion, and the reserved final split remain closed.

The three-camera far-view ablation completed 1,500 updates and selected
update 1,100 on the separate 12-scene validation split. Its 90 training
scenes included all 60 new scored SC starts; failed trajectories supplied
**perception labels only**. The selected model reached **5.93 native pixels
median / 14.44 p95** for the opening on 178 held-out visible far-camera
views. Exact command, checkpoint, 151,725 training views, 3,822 validation
views, metrics, and predictions are under
`/var/tmp/chmin_aic_20260918_act/sc_multiview_far_locator_batch3_20260925/`.
On the same two diagnostic starts, first-frame left/right opening errors
were **2.99/12.71 px** on `trial_999001` and **120.23/104.69 px** on
`trial_999003`. The latter's only geometrically in-frame opening is at
native image x = 14.17 px (about 3.5 px from the left edge of the 288 px
recorded image); the camera picture shows card structures covering much of
that border. The label's `visible` boolean checks geometric projection
inside the camera rectangle, not ray-tested visibility of the opening.
This edge/occlusion distinction is a plausible explanation for the large
miss, not yet a proven causal attribution. A 40-native-pixel image margin
leaves only **20/60** new training starts with the opening in any view, even
before accounting for occlusion. Repeating locator fitting on the same
frustum labels is therefore not justified as a full-route fix. The BC gate
remains failed; a different observation/trajectory strategy is needed.

### September 25: direct body-frame full-target ablation

As an **explicit ablation** of the user-preferred port-frame target, the
same 13.615M-parameter shared actor was trained to predict the recorded
teacher's *complete* four TCP targets relative to the **currently observed
TCP**. These are absolute goals in a changing body coordinate system, not
additive corrections to an old BC output. The auxiliary port-pose head is
still supervised, but its output is zeroed at the action-conditioning
interface; autonomous runtime uses RGB, ordinary measured state, task, and
the nominal decision clock. Simulator geometry supplies training labels
only. The SE(3) target conversion was round-trip checked before training.
The model was trained from scratch on the same 318 training / 39 validation
episodes as the batch-3 port-frame treatment for 4,000 updates, selecting
update 3,750. Its selection score was **2.817 mm** versus **3.182 mm** for
the warm-start port-frame treatment, although its SC initial-command p95
was **2.366 vs 2.145 mm** and near-port lateral p95 **0.644 vs 0.399 mm**.
This is an architecture probe, not a fully matched training-budget claim.
The exact run, checkpoint, history, and machine gate are under
`/var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_20260925/`
and [the gate report](2026-09-25-shared-body-bc-batch3-gate.json).

On the **same two already-open Gazebo development scenes**, with no guide,
privileged pose, exploration, recovery supervisor, or RL, the body-target
actor reached final plug-port distances **0.04 and 0.14 m** with no scored
contact. Tier 3 was **22.09 and 8.51**, neither a full insertion. The
port-frame treatment had scored **0/2** with wrist/enclosure contacts and
ended roughly **0.9 m** away. Exact scores and logs are in the local eval
root; [both raw bags](2026-09-25-shared-body-bc-batch3-dev2-mcap-archive.json)
are losslessly archived with verified hashes. This supports the hypothesis
that the pose-conditioned port-frame action interface exacerbated the
closed-loop runaway, but the comparison also differs in scratch versus
warm-start budget. It does not establish reliable insertion or prove pose
input is intrinsically harmful. A lower-rate continuation is testing whether
body-target near-port accuracy can improve. Step 5 remains failed; no RL or
reserved-final evaluation.

The selected lower-rate body-target continuation stopped at 2,250 further
updates by its fixed patience rule; update 1 was selected with combined
selection **2.626 mm**, SC initial-command p95 **2.370 mm**, and SC
near-port lateral p95 **0.356 mm**. On the same two already-open autonomous
development scenes it still did **not** insert: final distances were **0.27
and 0.07 m**, Tier 3 **0 and 17.28**, with no scored contact. Its
[two verified bags](2026-09-25-shared-body-bc-batch3-cont-dev2-mcap-archive.json)
and exact command are under
`/var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_cont_20260925/`.
Further optimization on the unchanged expert source shifted the two outcomes
and did not establish a reliable final approach.

For a supervised correction source, four **training-only** teacher-success
SC scenes (two/three cards, both target ports) were rerun with the frozen
body-target student directly executing its own proposals for nominal time
0–2 s; the privileged teacher then resumed. This is training data
collection, not autonomous evaluation. The teacher achieved full official
insertion on **3/4** scenes; the fourth stopped 2 cm away. None had scored
off-limit contact. Exact bag-to-episode and TF joins passed, physical
TCP-to-tip drift was **0.009–0.049 mm**, and the three successes passed
the exact proposal/execution and causal RGB/target checks. They yielded
**2,135** labeled chunks, including **123 actor-owned frames**. The
[label audit](2026-09-25-sc-body-early-rescue-pilot-labels.json),
[four-bag archive](2026-09-25-sc-body-early-rescue-pilot-mcap-archive.json),
and `sc_body_early_rescue_pilot_{20260925,candidates_20260925}/` local roots
preserve the pilot. Failed whole episodes remain outside BC. A frozen
20-scene training-only expansion across card counts and both target ports
is being collected in five isolated four-scene groups before a matched
control/treatment BC update. No RL or final split is open.

The 20-scene expansion completed as five isolated, scored four-scene groups.
Full teacher-rescued insertions by group were **3/4, 2/4, 4/4, 2/4, 4/4**,
or **15/20**. Combined with the pilot this is **18/24**; none is an
autonomous insertion. One interrupted wrapper attempt was preserved under
`group_01/interrupted_*`; its in-container engine was allowed to finish,
and only the clean scored repeat was admitted. The actual repeated group
scored three full insertions. The group-2 automation then stopped *after*
archiving because the host reporting Python lacked PyYAML; groups 3–5 ran
successfully with the corrected interpreter. The [machine gate](2026-09-25-sc-body-early-rescue-batch20-gate.json)
records every trial, card count, target port, command, calibration, score,
and archive pointer.

The strict combined admission check accepted **18 distinct training
scenes**, excluded all six no-insertion attempts, verified exact student
proposal/execution during takeover, and found maximum accepted physical
grasp drift **0.049 mm**. The [causal label audit](2026-09-25-sc-body-early-rescue-combined-labels.json)
reports **9,524** four-target chunks, of which **467** are actor-owned
anchor chunks from **738** actor-owned recorded frames. All accepted rows
are training split, and the generated teacher targets round-trip through
the selected port frame to numerical precision. This source is marked
teacher-rescued correction data, never clean expert or RL evidence.
A same-parent, same-update-budget, same-sampler-seed body-target BC
control/treatment comparison completed. Both started from the selected
scratch body-target checkpoint, ran 1,500 updates, and selected update 250.
The control had 318 training episodes; the treatment added the 18 successful
teacher-rescued episodes, for 336. The actor-owned correction rows received
the same tenfold sampling weight in both arms. Offline selection changed only
from **2.645 to 2.633 mm**; SC initial-command p95 worsened **1.760 to
1.876 mm**, and SC near-port lateral p95 changed **0.312 to 0.317 mm**.

Both arms were then run autonomously, without a guide, privileged geometry,
recovery rule, or RL, on the same two already-open Gazebo development scenes.
The control scored **43.80** total; the treatment scored **41.50**. Neither
inserted in either scene. For `trial_999001`, Tier 3 was **15.24 versus
15.82**, with final plug-port distance **0.08 m** for both. For
`trial_999003`, Tier 3 was **7.27 versus 7.49**, with final distance
**0.14 m** for both. Neither arm had scored contact. Four raw bags are
[hash-verified and archived](2026-09-25-shared-body-bc-rescue18-matched-gate.json),
with separate control and treatment archive manifests. The exact commands,
checkpoint hashes, per-trial scores, grouped offline metrics, and reported
latency snapshots are in the machine gate. This is a negative correction-data
result: the 18 examples did not turn the full-start actor into an inserter.
Step 5 remains failed. The other development scenes and all four reserved
final scenes stay sealed; no supervisor or RL experiment has started.

The next bounded training-only diagnosis let the frozen matched-control
body-target actor execute nominal seconds **15–17** of the teacher route on
four SC training scenes before the teacher resumed. Only **1/4** complete
episodes inserted; the others ended **0.02–0.06 m** from the port without
scored contact. Each takeover contributed 41 actor-owned frames. Actor
target versus teacher target translation disagreement at the *first* active
frame was small in three scenes, but was already about **72 mm** in the
fourth; over the two-second takeover the maxima reached **33–93 mm**.
This shows why a long near-port teacher-rescue window is a poor reliable
source of successful correction labels for this checkpoint. The sole
successful episode passed the causal and grasp admission checks, while the
three failures were excluded. The [pilot gate](2026-09-25-sc-body-near-rescue-pilot-gate.json)
and [four-bag archive](2026-09-25-sc-body-near-rescue-pilot-mcap-archive.json)
preserve the evidence. A one-second takeover on the same training scenes is
the next bounded test; this is still teacher-assisted data, not autonomous
performance.

The one-second rerun also yielded only **1/4** full teacher-rescued
insertions. Its third scene scored off-limit contact and measured **18.17 mm
physical grasp drift**, so it was rejected independently of insertion.
Teacher-target disagreement at the first active frame varied sharply across
repeats of the same scene definitions (**1.7, 68.4, 231.6, and 46.1 mm**),
showing that deterministic scene YAML does not imply identical takeover
state. Shortening the single continuous intervention did not make this a
reliable correction source. Only its one safe full insertion passed
admission. See the [one-second gate](2026-09-25-sc-body-near-rescue-1s-pilot-gate.json)
and [raw-bag archive](2026-09-25-sc-body-near-rescue-1s-pilot-mcap-archive.json).
The next causal data test should interleave much shorter actor pulses and
teacher correction throughout the route, retaining complete scored success
as the admission condition. Do not infer a policy benefit from these
teacher-assisted success counts.

The interleaved test used **0.2 s direct actor pulses every 4 s** over
nominal seconds 0–30, with the same frozen body-target actor and the teacher
controlling the intervals. On the same four training scene definitions,
**4/4 scored full insertion** and none had a scored off-limit contact.
Admission remained stricter than the insertion score: one success had an
official insertion-force penalty despite peak 44.79 N, and another had
**0.311 mm** measured TCP-to-tip grasp drift, above the fixed 0.25 mm limit.
Thus **2/4**, with 65 exact actor-owned frames, were admitted as supervised
correction candidates. This supports interleaving as a *data collection*
strategy, not as autonomous policy evidence. The [pilot gate](2026-09-25-sc-body-interleaved-rescue-pilot-gate.json)
and [four archived bags](2026-09-25-sc-body-interleaved-rescue-pilot-mcap-archive.json)
record all outcomes. Twenty additional frozen training-only scenes completed
in five isolated groups. Full teacher-rescued insertions by group were
**2/4, 3/4, 3/4, 3/4, 3/4**, or **14/20**. Combined with the pilot,
**18/24** inserted; the strict admission audit accepted **16/24 distinct
training scenes** and excluded eight. The accepted episodes contain
**10,042** causal four-target chunks, including **347** actor-owned anchor
chunks from **468** actor-owned recorded frames. The SE(3) teacher-target
round trip was within numerical precision. The
[batch gate](2026-09-25-sc-body-interleaved-rescue-batch20-gate.json),
[label audit](2026-09-25-sc-body-interleaved-rescue-combined-labels.json),
and six hash-verified raw-bag archive manifests retain failures as well as
successes. No RL or reserved-final scene was used.

A same-parent, same-sampler-seed body-target BC control/treatment comparison
completed sequentially on one GPU. Both ran 1,500 updates with corrective
episode sampling weight two and actor-owned rescue weight 20; only the
treatment added these 16 interleaved episodes. The grouped offline selection
was **2.634 mm control versus 2.722 mm treatment**. The treatment selected
update 1 and worsened SC initial-command p95 **1.560→2.382 mm** and SC
near-port lateral p95 **0.406→0.480 mm**. A bounded lower-weight rerun
(episode weight one, active-row weight ten) selected update 1 at **2.624
mm**, marginally better than its prior matched control's **2.645 mm**
combined selection, but SC initial-command and near-port p95 worsened
**1.760→2.344 mm** and **0.312→0.417 mm**. Neither treatment passed the
SC offline gate, so neither was promoted to live evaluation. The
[offline gate](2026-09-25-shared-body-bc-interleaved16-offline-gate.json)
retains both negative comparisons and exact checkpoints. Teacher-assisted
data collection worked; simple whole-episode replay did not make the
autonomous SC actor better.

The next bounded supervised test isolates actor-owned observations and the
short teacher correction immediately after each pulse, rather than treating
the much larger teacher-controlled remainder of each rescued episode as new
training signal. It will use the same complete-success admission and grouped
validation. No RL or reserved-final scenes have been opened.

That focused selection kept **1,458** causal rows from all 16 admitted
episodes: 347 actor-owned anchors plus 1,111 teacher-correction rows within
0.5 nominal second after a pulse. The [selection report](2026-09-25-sc-body-interleaved-rescue-window0p5s-selection.json)
contains exact source/output hashes and per-episode counts. The matched
1,500-update control/treatment selected **2.645 versus 2.621 mm** combined
offline error, but SC initial-command p95 worsened **1.760→2.315 mm** and
SC near-port lateral p95 **0.312→0.439 mm**; treatment again selected
update 1. On the same two already-open autonomous Gazebo scenes, both
remained **0/2 insertions**. The paired overall scores were **28.17 control
and 45.64 treatment**, without scored contacts; treatment ended 0.11/0.07 m
away versus control 0.19/0.15 m. This single score difference does not
establish a reliable policy gain: the *identical control model weights*
scored 43.80 on an earlier repeat of the same scene definitions. The
[offline gate](2026-09-25-shared-body-bc-window0p5s-offline-gate.json),
[live gate](2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json),
and four archived bags preserve both outcomes. The autonomous insertion
gate remains failed.

Before generating still more corrections, a frozen runtime cadence test
executed each predicted four-target chunk at the recorded 20 Hz target
spacing, instead of discarding targets 2–4 and replanning every tick. It
changed only inference scheduling, not the checkpoint, scenes, or labels.
On the two already-open SC development starts, both cadences remained **0/2
autonomous full insertions**, with no scored contact. Replanning scored
50.25 overall and ended 0.04/0.14 m from the opening; chunk execution
scored 31.43 and ended 0.20/0.06 m away. The per-scene direction of partial
score changed, and Gazebo repeats are variable, so this is no reliable
cadence improvement. Chunk-run p95 inference was 109.61/117.50 ms, below
the 300 ms limit. Exact scores, latencies, and archived bags are in the
[machine diagnosis](2026-09-25-step5-body-cadence-scored-diagnosis.json).
The frozen evaluation command is saved in
`/var/tmp/chmin_aic_20260918_act/run_body_parent_chunk4_eval.sh`.
To reproduce the posthoc trace after restoring a bag through the archive
manifest, run `scripts/diagnose_sc_body_actor_trace.py --bag <restored-bag-directory>
--port 0 --output <trace.json>` in the rootless ROS environment. The
script reads scored TF only during analysis; it is not loaded by the actor.

One bounded posthoc diagnosis used the archived scored bags for the first
opened scene, retaining the checkpoint and scene definitions. During the
first sustained regression (about 20–30 s), the chunk actor's physical
plug-to-opening distance grew **94→151 mm**, while its commanded fixed-grasp
tip proxy grew **92→159 mm**. Commanded and measured tip-motion directions
agreed (cosine **0.995**); the replan trace also commanded and physically
followed a temporary regression (**103→120 mm**, cosine **0.998**). These
are centimeter-scale route errors, not a demonstrated local lip collision
or cable snag. No scored contact occurred. The opening projected inside
all three calibrated camera frustums during the sampled interval, but the
bags lack synchronized RGB and actor auxiliary pose output. A geometric
frustum is not proof of actual visual visibility; this diagnosis cannot
separate target-recognition failure from the actor's action mapping. The
[1 Hz traces](2026-09-25-body-parent-chunk4-trial999001-scored-trace.json)
and [comparison plot](figures/2026-09-25-step5-body-cadence-first-divergence.png)
preserve the measurements. Scored geometry was used **only after** the
autonomous runs for diagnosis.

**Step 5 decision, September 25:** the teacher-rescue matched comparisons
and frozen cadence check did not improve autonomous full insertion. Stop
routine BC replay, data expansion, and extra-update runs. No supervisor or
SERL branch is promoted; final scenes remain sealed. The specific fault
worth testing is an *outward command at an observed approach state*, before
any relevant recovery trigger. The [revised next experiment and finite
budget](2026-09-25-step5-next-steps-recommendation.md#executed-update-and-revised-next-experiment-september-25)
require synchronized observation/action evidence before allowing one
targeted command-generation repair. If it fails its declared gate, stop
this BC repair branch and consider only an explicit bounded RL diagnostic
on observable, near-completion incidents.
