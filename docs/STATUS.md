# Current experiment status

**September 26, storage cleanup.** `/data1` filled. Old May smoke artifacts
and non-selected sweep checkpoints were removed; sweep configs and metrics
were verified on NAS first. The older May 27/28 policy-run trees passed
full-content checks (26,164 and 3,372 files) before local release and now
resolve through NAS symlinks. `/data1` has about **244 GB free**, up from zero.
Canonical verified expert episodes, original failed trajectories, September results,
and selected checkpoints remain protected. See the
[cleanup ledger](experiments/2026-09-26-output-space-cleanup.md).

**September 26, SC route sensitivity and proposed two-actor handoff.** A
five-card, all-left-rail Gazebo teacher route now crosses farther behind the
row, lowers while behind it, travels in an outside lane, passes the last card,
then returns to the selected SC port. On one development scene, its fixed
baseline and five of six one-axis ±10 mm variants fully inserted; the shorter
forward variant inserted only about 10 mm. A larger forward-margin check also
failed its preregistered all-insertion gate. The candidate therefore is **not
yet a robust cable-safe route**. Overhead/side video and sampled cable-link
geometry support a better outside path but cannot prove zero cable/card
contact or generality. A separate corrective teacher has inserted on this
scene; no autonomous actor was trained or promoted. Exact variants, failures,
videos, and limitations are in the [route-sensitivity record](experiments/2026-09-26-sc-far-return-route-sensitivity.md).
The [perception and handoff design](POSE_VISIBILITY_HANDOFF.md) specifies two
task-conditioned actors, selected-port visibility/confidence labels, and a
proposed temporal gate. It is a proposal, not a trained system. Step 5 stays
open, final scenes sealed, and RL deferred.

**September 25, SC teacher route correction.** The old teacher lift still
crossed the card region and is no longer a route-quality example for crowded
SC scenes. An opt-in privileged Gazebo teacher now moves behind the five-card
row, travels beside it in an outside-left lane, and approaches the port after
the last card. It scored full insertion on a five-card port-0 egress scene
and on a fixed five-card wide-camera diagnostic, with sampled cable-center
to main-PCB gaps **11.50 and 13.81 mm** respectively. The first port-0
attempt was deliberately rejected before motion by its clearance gate. A
five-card port-1 scene inserted with an 85 mm lane but passed within
**2.11 mm** of a main PCB, so it is not route-quality BC data. Widening that
lane to 110 mm only improved the gap to **3.25 mm** and lost insertion,
ending 40 mm from the port. Stop the simple lane-width sweep. These are
privileged teacher runs, not autonomous policy insertions; causal/grasp
admission and broader scene coverage remain open. The
[experiment and videos](experiments/2026-09-25-sc-lateral-bypass-and-policy-handoff.md)
include a synchronized 20 Hz overhead/side full-cable view, failures, exact
commands, geometry, scores, and NAS bag archives. Step 5 and RL remain gated.

**September 25, current gate.** The shared SFP/SC BC actor still has no
autonomous full-start SC insertion on the two opened, preregistered
development scenes. Adding ten new verified SC successes improved one
offline starting-command metric but scored Tier 3 = 0/2 in Gazebo. A further
SC-heavy continuation improved held-out SC initial-command p95 to 1.995 mm
and near-port lateral p95 to 0.409 mm, yet also scored Tier 3 = 0/2 and
contacted the enclosure. Its [raw bags](experiments/2026-09-25-shared-bc-sc99-dev2-mcap-archive.json)
are archived. A fresh 60-scene SC teacher collection has now scored **31/60**
full insertions and admitted **17,796** causal chunks. A matched six-scene
five-card teacher replay scored 0/6 without a route lift and 3/6 with an
80 mm pre-route lift; the three full lift trajectories added **2,287**
admitted chunks. The [source scores](experiments/2026-09-25-sc-diversity-batch3-scored.json)
and [paired lift scores](experiments/2026-09-25-sc-fivecard-lift-scored.json)
preserve successes and failures. A same-parent, same-budget BC control versus
34-new-episode treatment then scored **Tier 3 = 0/2 in both arms**, with
wrist/enclosure contact and final plug-port distances about 0.9 m. The
treatment's offline selection error also worsened, 2.772→3.182 mm.
The [matched gate](experiments/2026-09-25-shared-bc-batch3-matched-gate.json)
keeps this negative result. The posthoc teacher trace on the same two opened
development scenes is complete and excluded from BC. At the successful
teacher start, the frozen actor pose was wrong by about 42–44 mm while its
first command was within 0.5–1.6 mm; substituting true pose in the frozen
action head made that command much worse. A corrected offscreen locator
still missed held-out offscreen openings by 22.46 px median / 118.78 px p95.
The expanded three-camera visible-opening locator reached 14.44 px p95 on
held-out far views, but its only geometrically visible view in one difficult
development start missed by 104.69 px. It is not a usable full-start pose
cue. See the [diagnosis](experiments/2026-09-24-shared-actor-sc-serl-execution.md#september-25-expanded-bc-full-start-diagnosis).
An explicit direct TCP-body full-target BC ablation (same shared actor,
pose head auxiliary only) changed the same two autonomous SC starts from
runaway/contact at about 0.9 m to no scored contact at **0.04/0.14 m**.
Neither inserted; its SC near-port lateral held-out p95 **0.644 mm** was
worse than the port-frame treatment's **0.399 mm**. The training budgets
differ, so this is a promising diagnostic rather than a promoted policy.
See the [body-target gate](experiments/2026-09-25-shared-body-bc-batch3-gate.json).
Its lower-rate continuation also failed insertion on the same two scenes.
A training-only two-second body-actor takeover followed by teacher rescue
fully inserted on **3/4** scenes and produced **2,135** verified correction
chunks, including 123 actor-owned frames. Twenty additional frozen training
scenes completed; 15 inserted under teacher rescue. Across pilot and
expansion, 18 distinct successful scenes yielded 9,524 causal chunks,
including 467 actor-owned anchors. A matched same-parent body-target BC
comparison trained both arms for 1,500 updates. The treatment's offline
selection error was 2.633 mm versus 2.645 mm for control, but both remained
**0/2 autonomous insertions** on the same opened SC development scenes.
Overall official scores were 41.50 treatment and 43.80 control; neither
had scored contact. See the [matched gate](experiments/2026-09-25-shared-body-bc-rescue18-matched-gate.json).
This is supervised teacher rescue, not autonomous success or RL.
Continuous near-port body-actor teacher-rescue pilots scored only **1/4**
full insertion each at two and one seconds; one short rerun had off-limit
contact and 18.17 mm grasp drift. A separate interleaved training-only
pilot used 0.2 s actor pulses every 4 s and scored **4/4** teacher-rescued
insertions. Strict force and grasp checks admitted only **2/4** episodes
as correction labels. The 20-scene expansion then scored 14/20 full
teacher-rescued insertions; the combined 24-scene strict audit admitted 16
episodes with 10,042 causal chunks and 347 actor-owned anchors. Two whole
episode BC replay weights worsened held-out SC initial and near-port
metrics. A focused 0.5 s post-pulse correction selection kept 1,458 rows;
its matched treatment improved combined offline selection 2.645→2.621 mm
but worsened SC initial p95 1.760→2.315 mm and near-port lateral p95
0.312→0.439 mm. The frozen two-scene autonomous comparison finished
**0/2 full insertions in both arms**. The treatment's higher partial score
(45.64 versus 28.17) is not reliable evidence of improvement: identical
control weights scored 43.80 in an earlier repeat. See the
[collection gate](experiments/2026-09-25-sc-body-interleaved-rescue-batch20-gate.json)
and [matched live gate](experiments/2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json).
The frozen four-target cadence comparison also finished **0/2 insertions
for both cadences**. Its scored-trace diagnosis found the actor commanding
away from the opening at ~20–30 s, with measured plug motion following
that direction and no scored contact. The bags lack synchronized RGB, so
visibility and target-recognition versus action-mapping remain unresolved.
See the [diagnosis and plot](experiments/2026-09-25-step5-body-cadence-scored-diagnosis.json).
Routine BC/data expansion has stopped. The [revised, bounded next
experiment](experiments/2026-09-25-step5-next-steps-recommendation.md#executed-update-and-revised-next-experiment-september-25)
records synchronized frozen-actor observations on at most four training-only
starts and permits one targeted command-generation repair only if a common
cause is demonstrated; its budget and continuation gate are predeclared.
The other four development scenes and four reserved final IK scenes remain
sealed. Step 5 is open; supervisor and SERL experiments have not begun. See
the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md).

**September 25, full-start root-cause update.** A fresh SC rollout of the
latest corrected shared BC actor still scored Tier 3 = 0 (total −35) with
enclosure contact. Posthoc scored geometry shows its auxiliary pose already
wrong by 28.21 mm at frame 0; action error grows from 5.81 mm at frame 0 to
141.1 mm by frame 35. A separate frozen native RGB locator misses the
opening by roughly 350–390 px in initial side views, despite subpixel median
error near the port. A bounded far-view RGB retrain brought initial side-view
opening error to 3–4 px median. Fixed left/right calibrated triangulation
then reduced initial lateral port error from 125.7/140.5 mm to **1.2/2.3 mm
median/p95**; including the center view gave 103.8 mm median because the
opening was outside that view. Near-port p95 after actor runaway remains
6.1 mm, so this does not meet the insertion corridor. A nondeployable
exact-pose substitution **worsened** frame-0 action error to 24.36 mm, so
perception and action mapping both need work. See the
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md),
[action audit](experiments/2026-09-25-shared-corrected-fullstart-pose-action-audit.json),
and [matched triangulation](experiments/2026-09-25-sc-global-far-triangulation-side2.json).
The shared BC gate remains closed; no RL or final scene evaluation.

An eight-scene yaw-edge development check then found that the opening is
outside **both side cameras** at frame 0 in six of eight starts. The causal
side-view filter's near-port lateral p95 was 1.624 mm across seven near
episodes, so the single-scene 0.236 mm p95 did not generalize. A bounded
offscreen visual extrapolation ablation was tested with training-only
labels. See the [eight-scene report](experiments/2026-09-25-sc-global-far-eight-causal40.json).

That offscreen regression failed: eight-scene offscreen opening p95 was
482 px. A from-scratch true-pose actor upper bound also failed its matched
offline gate (selection score 5.852 mm versus 4.425 mm for the
observation-only control). A 2 mm action cap slowed the fresh actor's drift
but still scored Tier 3 = 0, total −35 with enclosure contact. The data
inventory showed only **38 SC training episodes** in the 274-episode shared
BC set. The subsequent 20-scene training-only SC teacher batch and lift
retries yielded ten causally audited successes. The
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
contains the commands and negative comparisons. No RL or final scenes.

The 20-scene SC teacher batch yielded eight audited full insertions; an
8 cm lift retry of its 12 failures yielded two more. The matched BC
treatment added 10 SC successes and 5,700 chunks. Its grouped initial SC
command p95 improved **2.791→2.565 mm**, but near-port lateral p95 worsened
**0.319→0.544 mm**. On two new, pre-registered development starts, both
control and treatment scored **Tier 3 = 0/2 and total −23 on each trial**
with contact penalties. The four remaining development starts and four
reserved final IK configurations remain sealed. Step 5 is still failed;
no SERL training has begun.

**September 25, latest shared SFP/SC actor gate.** The 14-step plan remains at
step 5: the single shared BC actor has **no autonomous full-start SC insertion**
on the new development start, so recovery RL and final evaluation remain
closed. Nine additional scored, successful 3 mm teacher-bounded SC episodes
(4,859 causal chunks) did not change that outcome in a matched 1,500-update
BC comparison. Correcting an SC/SFP clock mismatch improved grouped offline
SC initial-command p95 from 2.764 to 2.156 mm, but autonomous Tier 3 stayed
zero. Teacher-prefix handoff showed the actor could finish the last ~1 mm
when the privileged teacher had already placed the plug ~13.8 mm inside the
opening; when handed control ~20.8 mm before the opening, it drifted laterally,
hit the wrist/card, and failed. A frozen high-resolution SC estimator was
accurate at that handoff (~0.14 mm lateral error) but lost the target after
large actor-driven drift. Four training-side direct-takeover failures supplied
only 25 safe local teacher-correction chunks; their matched BC continuation
still scored Tier 3 = 0. A separate, nondeployable true physical-tip input
continuation was trained as an information upper bound and has not fixed the
failed-handoff action error in offline replay. A bounded 15× local-row
sampling ablation still scored Tier 3 = 0. A direct two-nominal-second actor
takeover followed by teacher rescue produced three scored full SC successes
in four training-side scenes and 2,046 causal chunks. Adding them in a
matched BC continuation still scored Tier 3 = 0 autonomously, and grouped
held-out near-port lateral p95 worsened from 0.470 to 0.568 mm. A runtime
ablation that replanned after each command also scored zero. Stronger
sampling of actor-owned rescue rows and an earlier takeover window were
also tested. The 15× near-port actor-row sampling still scored Tier 3 = 0
autonomously with wrist/enclosure contact. Direct actor control during the
first 2.0 or 0.5 nominal seconds followed by teacher rescue produced only
one full success per four-scene batch. Their two admitted episodes and 1,037
chunks did not pass the offline BC checkpoint gate when actor-owned rows
were sampled 20×. A nondeployable all-phase true-SC-pose upper bound is
training to test whether early geometry alone can repair action learning.
That privileged all-phase geometry upper bound also failed its offline
gate: held-out SC initial-command p95 was 4.817 mm versus 2.156 mm for the
observation-only parent. A provenance re-audit found nine historical
matrix/lift successes already present in the original SC supplemental
source; a later arm that duplicated them is invalid as a diversity test.
The trainer now rejects duplicate episode/frame rows. A genuinely new
eight-scene clean-teacher batch scored **5/8 full SC insertions** and
admitted **2,463 unique causal chunks** after all-source overlap checks.
A lift retry rescued two more starts, giving **seven unique full-success
episodes and 3,685 causal chunks**. In a matched BC continuation,
held-out SC initial-command p95 improved from 3.124 to 2.169 mm with
seven new scenes, but near-port lateral p95 worsened from 0.312 to
0.446 mm. Control, +5, and +7 actors all scored **Tier 3 = 0** on the
same fresh autonomous SC start with enclosure contact. The supervised
gate remains closed.
Two more full-success SC teacher recoveries were collected after the +7
actor directly controlled the first two nominal seconds with a 2 mm
per-command cap. Their 1,387 causal chunks improved matched grouped BC
metrics, but control and corrected actors both still scored Tier 3 = 0
on the fresh autonomous SC start; the corrected arm contacted the
forearm/enclosure wall. A longer from-scratch +7 BC fit also failed the
offline gate. The next diagnosis is the earliest autonomous command and
pose trace, before changing the architecture or opening RL.
See the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
and [artifact map](../outputs_README.md). No offline or online SERL began;
the four reserved final configurations remain sealed.

**September 25 latest shared BC gate.** Three clean 3 mm early teacher-assisted
SC episodes were admitted as causal labels and compared in a matched
1,500-update continuation against the same BC run without them. Both actors
scored **Tier 3 = 0** on the same new autonomous SC start with wrist/enclosure
contact; the added episodes did not solve full-start transport. On the same
training-side teacher scene, 8 mm and 6 mm bounded early actor pulses followed
by teacher control scored **16.88** (no insertion) and **46.05** (partial),
respectively. Their nominal 1.0/0.5 s intervention windows were actually
5.6/3.2 simulation seconds; neither episode was admitted as successful BC.
A 3 simulation-second observation-only startup hold followed by the same
actor still scored Tier 3 = 0 with wrist/enclosure collision on that fresh
scene. An SC-weighted 1,500-update continuation improved held-out first-command
p95 from 3.47 to 2.46 mm but also scored Tier 3 = 0 with the same contact.
On 242 held-out teacher-state SC transport frames, its direction agreed with
the teacher (median cosine 0.998); the fresh closed-loop route still fails.
A separate four-scene training-only bounded student collection is running.
The supervised autonomous gate remains closed, so offline/online SERL,
Gazebo transfer, and reserved final configurations have not started. See the
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md).

**September 24, shared-actor correction gate.** The guarded shared BC
actor still has no autonomous full-start insertion and stopped at 42 N in
its SC development rollout. A DAgger-style collector now executes short,
bounded actions from that same observation-only actor while the privileged
teacher supplies correction targets. Isolated repeats of a previously
successful SC teacher scene ended at partial insertion even with zero
student input. Replaying the original two-scene reset order restored a
Tier 3 = 75 control. A second-only student intervention in that order
also scored 75; its admitted episode has 400 student-active frames and
910 causal supervised chunks, 317 anchored during student control.
The warm-up scene and all failed/zero-intervention probes remain outside
corrective BC. The matched 2,750-update continuation finished:
SC held-out 0–5 mm lateral command p95 was **0.380 mm without** and
**0.629 mm with** fivefold-sampled corrective data. Both autonomous
guarded SC rollouts scored Tier 3 = 0 and hit a scored wrist/enclosure
contact; the corrective arm also incurred a sustained-force penalty.
This one-scene correction did not solve transport. The unit-weight
ablation improved offline near-port SC error, and three more scored
student-correction successes were admitted from an eight-scene collection.
A start-aware shared model trained on those additions, with newly generated
actor development scenes kept separate from training. No RL,
reserved final scene, or autonomous promotion has followed from this
teacher-assisted result. See the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
and [artifact map](../outputs_README.md).

**September 25 supervised gate:** a start-aware shared actor still scored zero on
four new SC development starts, all with wrist/enclosure contacts.
Training used 288×256 JPEG images followed by PIL 224×224 bilinear resize;
live inference had used direct native-to-224 OpenCV area resize. On 64
paired saved frames, this changed predicted targets by **0.90/3.25 mm
median/p95**. Matching the training JPEG path reduced the disagreement to
**0.075/0.256 mm**. The converter and one-hot task encoding passed
posthoc checks. The same-checkpoint, same-scene corrected-image rerun
also scored **0/4**. A second model with three extra successful
teacher-correction episodes scored **0/4** on those scenes. The runtime
then proved to have a 5 mm per-coordinate target cap despite teacher
setpoints commonly being 10–40 mm away; raising the cap to 20 mm scored
**0/4** and worsened off-limit contact. A scored teacher-only reference
on the first fresh scene inserted fully, while the actor moved in the
opposite board-direction within seconds. The actor predicts 3.35 mm
median first movement on SC training starts where the teacher almost
holds still. Corrective collection previously began after 3 s, missing
this failure. A bounded 3 mm teacher-assisted early-intervention probe
fully inserted on one development scene without scored force/contact
penalty, but it is not an autonomous actor result. A stronger first-command
loss improved teacher-state predictions yet the actor still went the wrong
way and hit its 44 N live force stop. The first early corrective
training-suite run filled scratch before its final score file and admitted
no episodes; its seven bag byte streams are archived. A replay with more
space and the same scene order is active. These findings
are in the
[experiment record](experiments/2026-09-24-shared-actor-sc-serl-execution.md);
there is still **no autonomous shared-actor full-start insertion** and
offline or online RL remains closed.

**September 24, latest gate.** The fourth SC perception suite was frozen
before eight new scenes were generated. Four scenes reached near-port views
(154 frames); the observation-only physical plug-to-opening lateral error was
**0.156/0.294 mm median/p95**, axial **0.201/0.309 mm**, and the frozen
board-line yaw error **0.048/0.084°**. Both position histories and yaw history
initialized on every near-port frame. These numbers are independent
confirmation of SC perception, not autonomous insertion. Eight raw MCAPs were
losslessly archived and SHA256 verified on NAS; images and scored metrics
remain under `sc_yaw_boundary_refreeze_eval_20260924/` in the existing
`/var/tmp/chmin_aic_20260918_act/` root. Only four of eight scenes reached
near-port, so broader coverage and complete live latency are still open.

The shared BC actor still fails its supervised insertion gate. A causal
elapsed-time variant improved the matched initial SC command error to
**5.29 mm**, but grouped near-port SC body-command error remained
**1.158/3.338 mm median/p95**. A bounded **training-only oracle pose**
diagnostic then supplied noisy ground-truth **robot TCP** pose near the opening and
selected update 2,000 of 3,000. Its grouped SC near-port body-command error
was **1.167/3.270 mm**: accurate TCP localization alone did not close the
action gap.
This diagnostic checkpoint explicitly refuses deployment. It does **not**
test the physical plug-tip pose estimated by the high-resolution SC cameras.
A separate noisy **ground-truth physical-tip** input test reached
**0.289/0.714 mm** lateral command error in the 0–5 mm SC band, a modest
improvement over the elapsed-time actor's **0.369/0.820 mm**. Giving both
oracle TCP and plug-tip inputs worsened this to **1.351/1.868 mm** in its
selected checkpoint; that branch failed and is not evidence that RGB pose
cannot help. All oracle checkpoints are nondeployable and selected by an
all-phase validation median. A near-port lateral-weighted
nonprivileged BC arm lowered SC 0–30 mm lateral p95 from **1.204 to 0.916
mm**, but worsened its median from **0.366 to 0.595 mm**; the 0–5 mm band
was **0.586/0.752 mm median/p95**. SFP near-port lateral p95 was **2.736
mm**. A new eight-start qualification-like SC teacher batch yielded **2/8
full insertions**, both admitted as 1,579 causal BC chunks. Four failures
ended with less than 0.4 mm TCP motion over the final five seconds while
32–76 mm from the issued setpoint; a specific cable/card contact is still
unproven. Matched 8 cm teacher pre-route lift rescued **0/3** starts.
A 5 mm near-port cap rescued **0/2** closer failures. The
near-lateral-selected BC rerun with the two new successes selected update
1,750/2,750 and reached **0.198/0.337 mm** lateral command error in the
0–5 mm SC TCP-axial band over 65 frames from five held-out episodes.
SFP 10–30 mm lateral p95 remains **2.650 mm**, and the matched first SC
command still misses by **7.17 mm**. The guarded absolute-base Gazebo
development rollout **failed**: the actor moved about **0.416 m** along a
different transport route from the scored teacher and stopped itself at
42.0 N near an enclosure-looking structure. Scored Tier 3 was zero.
Its exact contact pair is unverified. Near-port offline precision therefore
does not establish autonomous transport or insertion. Next is bounded
student-perturbed teacher correction data, then the same guarded start.
No recovery RL or final
reserved scenes have been opened.

**September 24, current continuation.** A third, independently generated
eight-scene SC perception suite completed with the revised frozen
observation-only tracker. Six scenes reached near-port observations, giving
228 frames. Combined physical plug-to-opening lateral median/p95 was
**0.213/0.443 mm**; 227/228 frames initialized. One scene's own p95 was
**1.004 mm**, and yaw p95 was **0.672°** overall with a **0.981°** scene
outlier. Complete offline native-image perception took **147.36 ms p95** on
one GPU, excluding ROS, actor, and command conversion. This supports the
pooled 0.5 mm perception gate but does not prove every scene or autonomous
insertion. A principled rail-yaw search-boundary rejection improved the prior
development suite from 0.132° to **0.083° p95** and the now-inspected third
suite from 0.672° to **0.085° p95**. It was frozen before a fourth independent
scene suite (`sc_yaw_boundary_refreeze_eval_20260924/`) was generated; its
completed result is summarized above. The [audit](experiments/2026-09-24-sc-supervision-audit.md#third-independent-frozen-check-september-24)
has exact metrics and artifacts; all eight raw MCAPs were losslessly archived
with SHA256 verification. Shared BC labels use complete recorded TCP targets
in port-opening frame. A first **13.60M-parameter shared supervised actor**
was trained from 268 SFP plus 32 successful SC episodes on one GPU in rootless
Docker. It stopped at 6,500 updates after a 2,000-update plateau, selecting
update 4,500. Held-out first-target translation median/p95 was **2.48/11.81
mm SFP** and **2.63/5.55 mm SC**. A no-privilege SC development rollout
failed badly: no insertion, final plug 0.51 m from the port, peak force
785.95 N and an off-limit wrist contact. On the matched scored teacher
scene, the model's first command was 16.01 mm from the near-zero teacher
command; its current TCP-in-port pose estimate was 6.04 mm and 8.06° wrong.
No further live rollout of that checkpoint is planned. A warm-started
command-loss/initial-phase-balanced supervised run completed 4,000 updates.
It reduced held-out SC near-port command error from **1.57/4.98** to
**1.46/3.84 mm median/p95**, but SFP near-port p95 remained **9.16 mm** and
the matched initial SC command still missed by **10.66 mm**. The guarded
absolute-setpoint runtime has not been live tested because these offline
gates failed. **No SC recovery RL checkpoint exists.**

**September 24, earlier continuation.** The second frozen eight-scene SC
perception check exposed a genuine port-tracker failure: the original
three-view causal tracker seeded a false position before the target was
visible and held it after correct detections appeared. Its physical
plug-to-opening lateral p95 was **45.5 mm**, so that frozen candidate failed.
The apparent **1.44° yaw p95** was a *scoring bug*: the metric combined a
scene-specific physical-tip label with the old fixed TCP-to-tip orientation.
Scoring against the selected port's bag TF directly gives **0.132° yaw p95**;
the image estimator and its inputs were unchanged. The separate plug crop
gave **0.197 mm physical-tip lateral p95**. An observation-only three-view/
two-view fallback tracker, with proximity acquisition and stable-target
history, gives **0.395 mm combined lateral p95** on those now-inspected
scenes and **0.164 mm port lateral p95** on the earlier eight-scene suite.
These are development results. Exact scorer/tracker code and thresholds were
frozen before generating a **third, separate eight-scene confirmation**;
that completed run is under
`/var/tmp/chmin_aic_20260918_act/sc_port_tracker_refreeze_eval_20260924/`.

The SC teacher search now has at least one scored, causally audited
full-insertion candidate in **all 12** `(0–5 NIC cards, selected SC port)`
cells. A 5 mm near-port integrator cap turned the previously missing
two-card/port-1 start from partial to **Tier 3 = 75**, but reduced the matched
five-card/port-0 control from 75 to **16.0**. This cap is a teacher data probe,
not a deployable universal rule. The 21 canonical SC experts plus 11
supplemental successful candidates have physically corrected port-frame
waypoint labels. A separate [admission audit](experiments/2026-09-24-sc-bc-admission.json)
passed all 11 new episodes and 6,220 causal chunks after scene-split,
image/command, physical-grasp, and force/stall checks; the original candidate
manifest remains unchanged for provenance. See the
[matrix](experiments/2026-09-24-sc-near-bias-and-success-matrix.json) and
[execution log](experiments/2026-09-24-shared-actor-sc-serl-execution.md).
That earlier audit preceded the shared BC checkpoints summarized above;
no SC recovery RL checkpoint exists.

The active [shared SFP/SC actor and recovery execution plan](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
starts by freezing scene/data inventory and validating Isaac distal-cable
fidelity, SC action targets, and observation-only SC pose. It then fills
missing successful Gazebo SC demonstrations before shared BC and gated Isaac
RL. One actor will own normal motion and post-retreat choices; the measured
backoff supervisor is deterministic. Shared BC has since trained and failed
its autonomous gate; SC recovery RL remains unopened.

The bounded Isaac SC mechanics gate now passes with robot self-collision off:
0/3/5-card scripted scenes insert, and external plug/port and gripper/card
contacts remain active. The [fidelity record](experiments/2026-09-24-isaac-sc-fidelity-fix.md#cable-self-collision-diagnosis-and-corrected-moving-view)
preserves the failed kinked cable and corrected 20 Hz video. The next SC
supervision audit found a quaternion-order bug in the old port-frame TF
conversion. It is corrected in the [v2 audit](experiments/2026-09-24-sc-port-target-audit.json);
the old label orientations were wrong by 153.7° in a checked terminal sample.
Strict and held-target connector chunks are now built from scored Gazebo
episodes. The older low-resolution [SC pose probe](experiments/2026-09-24-sc-pose-probe-results.json)
reached 1.70 mm near-port lateral p95 on seven held-out episodes. The
[native-image continuation](experiments/2026-09-24-sc-native-pose-ablation.md)
recollected 1152×1024 RGB from all 21 verified SC reset groups and generated
new, episode-grouped training and development scenes. It exposed wrong-port
detections after occlusion, so raw per-frame predictions are not treated as
ready for control. A three-camera landmark model and causal fixed-port filter
use RGB, measured TCP, and fixed calibration only. After adding 12 yaw-edge
training scenes, the position head reached **0.141 mm lateral and 0.106 mm
axial p95** over 208 near-port frames in four separate yaw-edge episodes;
its history initialized in all four. The yaw-constrained orientation head
reached **0.804° p95** on the same suite, but one episode had no trusted
orientation history. A position-anchored fallback for only the first two
orientation updates reached **0.806° p95 with 100% initialized coverage**
there; on the older 12 development scenes it gave **0.596° p95**. This
fallback was designed after inspecting the yaw-edge suite, so these numbers
are **development evidence**, not an untouched confirmation. The frozen
eight-scene follow-up and its failure are described below. Offline CNN
perception p95 was 37.5 ms after retraining, excluding the rail stage;
live ROS+actor latency and autonomous insertion are unmeasured.
The SC perception gate and shared BC/RL remain open. The
[supervision audit](experiments/2026-09-24-sc-supervision-audit.md) records
the quaternion fix, timings, camera provenance, and label limits.

That eight-scene check confirmed the position filter (**0.214 mm lateral
p95**, seven near-port episodes) but exposed **1.608° orientation p95** in
one partly occluded scene. A new [board-rail yaw estimator](experiments/2026-09-24-sc-native-pose-ablation.md#frozen-suite-failure-and-board-rail-yaw-ablation)
uses long visible rail lines in the original RGB views, measured camera pose,
and a coarse learned visual yaw prior. It reached **0.064° p95** on the 12
grouped development scenes and **0.078° p95** on the already inspected
eight-scene suite, with every near-port episode initialized. The rail stage
measured **88.8 ms p95 offline**, separately from the 37.5 ms CNN stage.
Because the failed scene motivated the rail method, a different eight-scene
development suite was generated only after checkpoint/code/threshold hashes
were frozen. The [fresh evaluation](experiments/2026-09-24-sc-board-lines-fresh-evaluation.json)
measured **0.165 mm lateral / 0.183 mm axial / 0.109° board yaw p95**
over 236 near-port frames in seven of eight new scenes, with both causal
histories initialized on all near-port frames. The eighth scene never reached
the near-port region. This establishes development accuracy for the **port**
and board yaw. The [physical-tip audit](experiments/2026-09-24-sc-supervision-audit.md#fresh-native-image-confirmation-and-remaining-physical-tip-question)
then found that the fixed TCP-to-tip proxy is wrong by 1.15–10.58 mm across
16 scored pilot scenes, while the episode-specific physical grasp is nearly
rigid. The frozen RGB tip head failed on the new eight-scene suite at
**6.83 mm lateral p95** against the actual tip. A first retrained shared-crop
head reached **0.356 mm lateral p95** on 12 grouped development episodes,
but **0.555 mm** on the inspected eight-scene check, with a cold-start
failure. The revised [separate plug-crop candidate](experiments/2026-09-24-sc-physical-tip-model-comparison.json)
uses scored physical-tip labels for training and a frozen RGB locator for
crop selection. It reached **0.290 mm tip lateral p95** on 12 held-out
development episodes. On the earlier eight-scene check, its combined
physical plug-to-opening error was **0.450 mm lateral / 0.174 mm axial p95**
with both histories initialized throughout near-port motion. Because that
check motivated the new crop, a second eight-scene suite was generated only
after freezing the candidate. Offline four-model perception measured
**128.3 ms p95** including native image decode and rail extraction; full
live ROS+actor latency and autonomous insertion remain unmeasured. Shared BC
stays gated on the new confirmation and SC teacher-data admission.

## September 24 corrected Isaac SC route contacts

After the [mechanics repair](experiments/2026-09-24-isaac-sc-fidelity-fix.md),
controlled five-card routes reproduced named **gripper/card** blockage twice
and **plug/card** blockage on a different route. In the instrumented across-card
run, the gripper moved only 6.74 mm over 75 commanded steps while gripper/card
contact exceeded 100 N; route reversal cleared it. Cable links 1/2 also touched
a card, but only during the gripper-led jam. A nearby clear lane had no sampled
scene contact and reached the transport waypoint. See the [route record, plot,
and raw trace map](experiments/2026-09-24-isaac-sc-contact-routes.md).
This is one constructed scene with a privileged scripted controller. An
**independent cable snag is still unproven**, and the high-force routes are
unsafe as training demonstrations. Gazebo's retained bridge logs only
off-limit contact pairs, so the historical Gazebo cable-trap candidate remains
causally ambiguous. The newer Isaac container now passes a three-camera
[RGB smoke test](experiments/2026-09-24-isaac-sc-fidelity-fix.md#camera-follow-up-september-24)
after the same local driver-check override used by the earlier working
container. A diagnostic view visibly shows orange cable and the card row.
A 61-frame moving clear-lane clip also renders, but its distal cable looks
sharply kinked/looped relative to the retained Gazebo wide view. Distal cable
fidelity and cross-simulator appearance checks remain before visual SC
recovery training. No SC BC or RL training was started.

## September 24 Isaac SC mechanics repair

The [SC fidelity repair](experiments/2026-09-24-isaac-sc-fidelity-fix.md)
calibrated the SC grasp and first cable links from scored Gazebo TF, restored
the source SC-port SDF colliders in the correct port frame, and removed two
measured internal grasp-contact artifacts without disabling scene collisions.
In one constructed five-card development scene, a privileged scripted
full-start Isaac probe reached the scored Gazebo port-base target with
0.073 mm final axial and 0.054 mm lateral error; peak plug/port contact was
1.04 N. Two deliberate near-port lateral commands yielded plug/port
blockage at 54 and 92 N, with gripper/first-link contact zero. The old
gripper/card item-2 collision did not recur on the centered route with the
calibrated grasp. Deliberate card-crossing routes above reproduce a separate,
named gripper/card incident.
These are [physics-only diagnostic traces and exact hashes](experiments/2026-09-24-isaac-sc-fidelity-summary.json),
not autonomous insertions or RL replay. The original camera attempt failed
under the rootless Isaac Sim 5.1 driver check; the later container-local
override passed a short repaired-scene RGB smoke test. Distal cable and cable/card snag
fidelity also remain unverified. No SC BC or SERL training was started.

## September 24 hierarchical recovery continuation

The fixed five-card Gazebo route probe now has seven 20 Hz wide-view reruns:
six partial insertions and one no-insertion stop with 52.9 N peak raw wrist
force norm (72.16 N vector-tared official force maximum),
38.7 mm plug axial distance before the opening, 4.45 mm lateral offset, and
111.6 mm final TCP command error. The cable visibly crosses the cards, but
the contact pair is unknown and the new run differs from the older nearly
stationary-cable candidate. Official post-friction evaluation image and
single-trial reset evidence make the historical ghost-collider bug a poor
explanation. Preserve both incidents as **ambiguous obstruction**, not proven
cable snags. Videos, commands, geometry, and toolkit cross-check are in the
[fixed-route record](experiments/2026-09-23-fixed-five-card-route-probe.md).

The approved branch is [hierarchical recovery](HIERARCHICAL_RECOVERY.md):
learned policy for normal motion and retry choices, deterministic measured-pose
retreat on a validated force-plus-stall trigger, and escalation to an earlier
safe route checkpoint after repeated same-corridor blockage. The existing
Isaac controller only performs local position backtracking; neither the
hierarchy nor SC-to-SC learned recovery has been validated. The
[execution plan](experiments/2026-09-24-hierarchical-recovery-continuation.md)
ranks failures, specifies the SC BC gate, matched no/local/hierarchical
ablations, optional offline critic warm-up, online SERL, Gazebo transfer, and
unchanged official-evaluation gate. Hierarchical assembly and cable recovery
have prior art, but the two closest papers reviewed in the
[core design](HIERARCHICAL_RECOVERY.md#related-work-and-claim-boundary) do not
describe our proposed local-to-route measured-path checkpoint escalation.
Any publication claim still depends on a measured gain and a wider novelty
review.

Execution update: an [indexed incident audit](experiments/2026-09-24-hierarchical-recovery-continuation.md#execution-log)
contains 24 retained Gazebo SC route/targeted episodes (8 full, 11 partial,
5 none); fixed-seed repeats are one scene group. Seven 20 Hz bags now have
measured TCP, command, and force-vector traces. The initial two-sample
force-plus-stall trigger, when correctly tared as a vector, marked **7/7**,
including six partial-insertion runs; a longer-window variant also marked
**7/7**. Subtracting force norms had misleadingly marked zero and two runs,
respectively. Neither is approved for robot control. A simulator-independent hierarchical
checkpoint prototype passes four isolated tests but has not driven an
autonomous rollout. Canonical SC supervision remains 21 verified Gazebo
episodes, all NIC1–3. The later corrected Isaac SC grasp/port now passes one
scripted five-card mechanics scene, but image collection and broader scene
validation remain open. **No SC BC, offline SERL, online SERL, Gazebo transfer, or
official score is claimed for this continuation.** The precise gate and
commands are in the execution record.

A broader read-only audit of 17 Gazebo traces across five scene/config groups
found peak TCP target-tracking error of 89–127 mm in the three no-insertion
runs versus 56–73 mm in fourteen full/partial runs. An 80 mm line separated
these saved cases, but it was selected after seeing them, and neither force
nor tracking error reveals the contact pair. This remains a candidate
diagnostic for new-scene validation, not a deployed backoff trigger.
Three fresh normal-collision five-card Gazebo variations then yielded one full
and two partial insertions, using either the privileged route plus stock
CheatCode or stock CheatCode alone. None exceeded 80 mm tracking error; there
is still no fresh failed-scene sensitivity check. The stock run's layout was
only 7 mm in X and 4 mm in Y from an earlier no-insertion layout, yet became
partial. Their [scores, config hashes, traces, and videos](experiments/2026-09-24-hierarchical-recovery-fresh-scenes.json)
are saved. These are diagnostic teacher runs, not learned recovery.

The SC supervision audit then reconstructed 13,349 recorded teacher TCP
targets in the selected port base frame from all 21 canonical scored bags.
It found 84 missing command indices and nine image-time gaps over 250 ms;
four adjacent recorded targets can span 150–450 ms. The labels are useful,
but they are not yet exact fixed-50-ms connector-waypoint chunks, and there
is no validated observation-only SC port estimator. See the
[port-frame audit](experiments/2026-09-24-sc-port-target-audit.json) and
[execution notes](experiments/2026-09-24-hierarchical-recovery-continuation.md#sc-teacher-target-frame-audit-details).

## Fixed five-card route probe and S3 audit (September 23)

The SC cable's opposite SFP-module end is free. A free end can still drape
over or catch on the NIC cards. The five-card collection's accepted video
**was downloaded from S3** and matches the local accepted MP4 byte for byte.
The original failed seed-51500 VLM/MoveIt video was not found under the
checked clean, development, or EC2-transfer prefixes; the clean collection
contains accepted attempts 27 and 28, not that old failed attempt.

A privileged diagnostic route used one fixed generated five-card Gazebo scene,
normal collisions, and stock CheatCode insertion after a long transport path.
Across-card runs were **one full, one partial, one none**; outside-left runs
were **one full, two partial, zero none**. The no-insertion across-card run
brought a cable segment center to 0.9 mm from a main PCB collider and later
held it nearly fixed while commanded TCP motion advanced and the plug stayed
46.7 mm before the port. Successful/partial repeats of the same planned
across-card route had at least 8.6 mm measured center clearance. A lower
route failed with the cable far from the cards and is a separate robot-path
clearance confound. This is a **credible route-sensitive cable-trap candidate,
not a named cable/card contact or proven causal snag**. No RL or model training
was run. Review the [fixed-scene comparison, overhead/side plots, and videos](experiments/2026-09-23-fixed-five-card-route-probe.md).

## Ordinary Gazebo and VLM-route cable follow-up (September 23)

The broad development audit now has **19 valid, clean stock-CheatCode Gazebo
scenes** across SFP-to-NIC counts 1--5, SC-to-SC counts 0--5 on both rails,
and two saved randomized SC layouts: **13 full, five partial, one no
insertion**. The original long batch became invalid after Gazebo
physics-entity errors; each affected scene was rerun alone. Three SC rail-1
partial runs with one, three, and four NIC cards stopped at almost exactly
10.5 mm past the port entrance despite 0.09--0.38 mm lateral error. Two
historical SC layouts failed with larger lateral approach error. The
two-card SFP scene was partial. These are insertion/approach failures, not
verified cable-to-card snags. See the [audited scenes and videos](experiments/2026-09-23-ordinary-development-cable-audit.md).

An older [expert-generator note](expert_matrix_template_fixes.md#sc-to-sc-nic-bypass-full-insertion-template)
does record a five-card VLM/MoveIt route where the center-camera reviewer saw
the cable catch on the NIC stack; a later outside-left bypass succeeded with
score 89.17. Its failed raw trajectory/video is absent from the checked local
and S3 clean archives. Of 190 retained SC full-insertion agent/VLM replay
attempts, 150 scored at most 1. A saved three-card score-1 trajectory with a
50.5 N transient force peak was replayed against its exact scene and again
scored 1, but both runs failed later at a roughly 14--16 mm lateral port
handoff miss. The TCP kept moving through the old force spike. Regenerating
the first five-card seed-51500 scene and running stock CheatCode produced a
partial axial insertion without a visible cable snag. **The historical
route-dependent snag is credible but is still missing a retained causal
incident suitable for RL labels.** The outside-left bypass remains a route
prior; cable-specific RL stays gated on a recorded contact/tension incident.
No model training was run in this audit.

## Ordinary-evaluation cable coverage correction (September 23)

The recent Gazebo audit did not cover every normal development setting. The
current three-trial `sample_config.yaml` has one-card SFP-to-NIC scenes and a
zero-card SC-to-SC scene, so it cannot expose SC cable snag among several NIC
cards. The selected production-family stress runs were also not an exhaustive
`training_broad` sweep.

The unfiltered historical archive contains 275 excluded SC-to-SC episodes from
randomized one-through-five-card Gazebo collections. Of these, 212 have official
noninsertion scores from collections deliberately stopped near the gate. The
remaining 63 lack the raw score or trajectory lineage needed for verification;
the exclusion count is not a cable-failure count. These used planned initial
motion followed by CheatCode. Initial timeline review shows cable/card
proximity and mid-trajectory force spikes, but the retained data cannot name
the contacting body or prove that cable contact caused a stall. Cable snag is
therefore reopened as a high-priority causal audit rather than treated as
disproved by the recent bounded runs. See the updated
[ranked failure record](experiments/2026-09-23-ranked-failure-scenarios.md).

## Ranked failure scenarios and targeted reproduction (September 23)

A new evidence-based ranking separates learned-policy alignment drift, local
port-lip blockage, large approach blockage, and unverified cable snag. Ten
additional post-fix Gazebo SC trials used stock CheatCode. Three exact Trial 3
repeats and all three bounded grasp-offset repeats inserted fully. Both
five-card runs to SC rail 1 stopped at nearly aligned partial insertion. Both
five-card runs to rail 0 failed with 89--126 mm terminal TCP tracking error and
brief scorer force peaks of 32--47 N.

Camera review showed the cable clear of the NIC card field in these failures.
They increase the priority of local axial recovery and larger approach-routing
recovery, but they do not establish cable snag. The ordered continuation is:
repair the collision-faithful Isaac SC scene, collect local alignment/contact
incidents first, train matched correlated-exploration and measured-backtrack
RL arms, then add large obstacle routing. Cable-specific RL remains gated on a
causal snag incident. See the
[ranked failure record](experiments/2026-09-23-ranked-failure-scenarios.md).

## Official Gazebo CheatCode failure audit (September 23)

The previous failure taxonomy mixed Isaac development evidence, broad Gazebo
data-collection scenes, and hypotheses. It was not a production failure census.
The corrected audit ran the stock official CheatCode in rootless Docker against
the exact released qualification YAML five times. SFP inserted in 10/10 trials.
SC inserted in 4/5; the remaining run was a partial axial block with about
0.06 mm lateral error and 0.19 degrees orientation error, but the plug stopped
3.21 mm outside the entrance and emitted no insertion event. CheatCode still
returned `True`, so engine task completion is not an insertion label.

A separate, labeled production-family stress suite covered SFP and SC, card
counts 0/1/3/5, both SC rails, a rail-limit position, released board poses, and
norm-bounded grasp changes. Card count alone did not cause a snag: every
0/1/3/5-card SC trial inserted. The SC/Trial-1-board-pose cross-combination had
an 8.05 mm lateral miss and 72.6 mm TCP tracking error. Of three five-card
grasp tests at 2 mm translation norm and 0.04 rad rotation norm, two inserted
and the negative-y/negative-pitch case stopped axially after partial insertion.
No current production run established a cable snag or named gripper-card
collision. The later targeted suite above shows that five-card outcomes can be
worse and intermittent even though the visible cable remains clear of cards.

Further recovery work starts from failures actually recorded in post-fix
Gazebo: intermittent near-aligned SC axial blocking and larger approach
tracking failures. See the
[official audit](experiments/2026-09-23-official-cheatcode-failure-audit.md).

## SC-to-SC mechanics and routing bring-up (September 23)

The approved multi-card SC continuation reached its simulator-mechanics gate.
The audit corrected the SC port mount, reversed cable endpoint topology and
fixed-joint transforms, reset ordering, multi-card scene support, and the SC
contact target. With normal collisions and no intervening cards, a scripted
14 mm retreat plus positive 4 mm lateral retry reached 0.099 mm final lateral
and 0.008 mm axial error. The matched negative retry entered the corridor only
briefly and diverged, confirming that recovery direction matters.

The apparent five-card cable snag was instead dominated by the gripper housing
striking a NIC card. Removing plug and cable collisions did not change the
failure; direct contact sensing measured about 128 N at the gripper/card
contact. The source reversed Gazebo grasp was not physically reachable in the
current Isaac base/board arrangement. Therefore no SC BC or RL was trained and
the failed rollout is not labeled as cable-snag data.

A follow-up source audit found that Gazebo intentionally removes two endpoint
collision groups near the palm and shortens/shifts the first rope collider. The
Isaac builder now reproduces that contract and uses the equivalent wrapped
wrist solution. This did not fix the full scene: a hold still produced a
36.98 kN peak. Removing the board, port, and self collision made the motion
low-force, but physical interpolation stopped 44.27 mm from the requested tip
pose. The remaining fault is narrower but unresolved: grasp transform, scene
placement, or articulation actuation. The SC learning gate remains closed.
The corrected asset is stable at its native initial posture, which localizes
the failure to near-port placement/reset rather than an always-unstable cable
topology. Gazebo welds to `ati/tool_link`; the prepared USD welds to the right
gripper finger. That frame contract and the source board/port placement are the
next items to reconcile.

A declared gripper-collision-disabled proxy tested routing while retaining
plug, cable, card, board, and port collisions. A direct high approach held the
final component gate in 1/3 seeds; a privileged 100 mm route around the card
edge held it in 3/3. This supports hierarchical routed transport as a future
candidate, but it is neither autonomous nor valid policy data. The active gate
is a reachable collision-faithful SC grasp/scene contract, followed by scripted
card-count 0--5 validation. See the
[SC mechanics record](experiments/2026-09-23-sc-mechanics-and-routing.md) and
[machine summary](../outputs/experiments/2026-09-23_sc_cable_bringup/summary.json).

## Probabilistic SERL recovery continuation (September 23)

The deterministic local RPDP controller was converted into a four-component
mixture over complete four-waypoint port-frame trajectories. It is a full
policy rather than a residual correction. Frozen perception plus the actor has
2,586,195 parameters, and complete live inference measured
20.43/22.67/23.78 ms p50/p95/p99. Measured-path backtracking, force plus stall
detection, lateral post-clearance exploration, replay ownership masks, balanced
prior/online replay, twin critics, and mixture SAC updates are implemented.

Frozen stochastic collection produced 3/7 local 8 mm seating successes. The
first backtracking implementation was invalid because force alone triggered on
normal insertion and abort persisted. After repair, no blockage trigger fired
in five episodes; its failures were lateral drift/timeouts, so backtracking had
no opportunity to help. A 12-episode prior contains seven successes and five
failures. A ten-episode critic warm-up ranked its one held-out success above its
one held-out failure, but this two-episode check is weak.

On eight new episode-grouped development starts, frozen BC scored **3/8**.
Initial correlated and backtracking SAC actors both scored **0/8** because the
actor collapsed deterministic probability onto unvalidated alternative modes.
Adding full-mixture behavior regularization stopped that collapse, but the
requested lower actor learning rate was silently overwritten when optimizer
state was restored and the result remained 0/8. Optimizer restoration now
retains momentum while reapplying explicit learning rates. The final tighter
trust-region run scored the same **3/8 on the same three episodes** as frozen
BC. It preserved the warm start but did not improve it.

A separate three-camera recording rerun on the same eight configurations scored
**1/8 BC, 1/8 tight-trust RL, and 3/8 tight-trust RL plus backtracking**. All 72
per-episode clips are on the [video review page](../outputs/experiments/2026-09-22_serl_recovery/videos/index.html).
Because BC and plain RL differed from their earlier 3/8 metrics-only outcomes,
the recording rerun is preserved as separate evidence and does not establish an
RL or backtracking improvement.

Do not promote this online checkpoint as better than BC and do not open the
reserved final split. Before another actor update, collect substantially more
episode-grouped recovery and blockage outcomes and validate critic ranking on
more than two held-out episodes. The selected checkpoint is preserved because
it verifies a mechanically safe continuation path. See the
[experiment record](experiments/2026-09-22-serl-mixture-recovery.md),
[actor explanation](SERL_PROBABILISTIC_ACTOR.md), and
[recovery strategy](SERL_RECOVERY_STRATEGY.md).

### Video diagnosis and current boundary

The three-camera rerun was reviewed configuration by configuration using the
recorded axial depth, lateral offset, orientation, and cable motion. The dominant
failure is lateral divergence: several actors move from a nearly centered reset
to 6--18 mm off-axis, sometimes crossing the entrance plane beside the port.
One configuration stalls at the port lip with roughly 0.04--0.05 rad angular
error. No episode provides conclusive evidence of a cable snag. Cable motion is
a possible contributor in configurations 12 and 14.

The recovery controller does more than retreat. It retraces measured TCP
positions in 0.25 mm commands until at least 1 mm of clearance and low force,
then constrains 12 control steps toward near-lateral motion. One millimeter is
twice the 0.5 mm local success corridor, so this can release a small lip contact.
The recording did not retain per-step recovery-mode traces, however, and the
visible backoff may also be a learned action. The two additional successes do
not prove that hard-coded recovery caused the improvement. See the
[committed failure analysis](experiments/2026-09-23-serl-video-failure-analysis.md).

### Approved SC continuation now in execution

The active work targets SC-to-SC with multiple intervening cards, where
the cable can require a larger retreat and a routed detour rather than local
realignment. Current selected pose/RPDP/SERL evidence is SFP-to-NIC only. The
ordered plan first audits Isaac SC support, defines cable snag causally, records
a bounded full-episode discovery set with compact force/motion/recovery
telemetry, and creates reproducible short incident resets. It then gates SC pose
estimation, port-relative BC, fixed multi-scale recovery, a learned recovery
option, offline critic validation, short-incident online SERL, and finally
full-episode evaluation. See [SC cable-snag recovery plan](SC_CABLE_SNAG_RECOVERY_PLAN.md).

## RPDP and DPPO continuation (September 22)

**Scope correction:** the RPDP evaluations below use
`scene.target.seated_depth_m: 0.008` and privileged near-opening resets. The
physical SFP entrance is about 45.8 mm from the seated port reference and the
cage is about 48.72 mm deep. Therefore these successes are **local 8 mm seating
events**, not official full-depth or full-episode insertions. The controller has
not yet been evaluated from the normal initial distribution or with the
official Gazebo insertion event. Earlier wording saying simply “strict
insertion” was too broad.

The selected supervised controller is now a PoseInsert-inspired, deterministic
AIC-RPDP policy. It predicts the connector pose after each of four recorded
50 ms teacher target commands, always relative to the fixed port opening. A
verified rigid transform converts those four waypoints back to four
TCP-body-frame commands. This is a complete target trajectory, not an additive
correction to ACT and not a base-frame absolute action.

The follow-up audit fixed four material BC problems: recovery data had used the
75% blended executed action instead of the recorded teacher target; four
different commands had been compressed into one 200 ms endpoint; a numerically
oversized checkpoint-improvement threshold kept undertrained update-200
models; and live insertion commands were sometimes clipped from 0.5 to 0.2 mm.
The original 322-row diagnostic also lacked insertion, but the current data
does not: the 823-row successful fit split contains 424 insertion decisions.
A DAgger extension keeps safe teacher queries from failed blended rollouts
while rejecting stale reset images and force at or above 5 N, producing 980 fit
decisions.

The repaired diffusion arm improved to 2/6. A matched deterministic trajectory
regression arm then achieved **6/6 autonomous local seating events** on the fixed
development starts and **8/8** on an additional set unused by RPDP fitting or
selection. Across all 14 episodes, final lateral error was 0.024--0.402 mm,
orientation error was 0.002--0.066 degrees, post-reset maximum force was
1.69--3.69 N, and complete p95 inference was 19.86 ms. Evaluation used no
guide, guard, exploration, privileged geometry, or RL.

A later frozen five-start video check achieved **4/5 local seating events** in the same Isaac
SFP-to-NIC scene and held-out cable-template family. The failure timed out at
5.987 mm depth despite 0.329 mm lateral and 0.032 degree orientation error;
the four successes reached 7.562--7.640 mm. Complete inference was 20.12 ms
p95. The cumulative post-repair development observation is therefore 18/19 and
shows that slow or stalled axial completion remains possible. Three per-episode
camera videos, a review page, and exact metrics are linked from the
[RPDP experiment record](experiments/2026-09-22-rpdp-dppo.md#post-selection-five-rollout-video-check).

The selected supervised arm has 2,004,329 trainable policy parameters plus 536,631 frozen
perception parameters, or 2,540,960 total. It passes the supervised development
gate. The older DPPO round remains a valid but unsuccessful 1/6 result and was
not continued. The four reserved final configurations remain sealed; no new
RL from the original RPDP/DPPO stage ran. The later probabilistic SERL
continuation is reported above. No world-model, SEER, or Gazebo-adaptation
training ran. See the
[RPDP/DPPO and BC repair record](experiments/2026-09-22-rpdp-dppo.md).

The staged
[perception, supervised control, and model-free RL experiment](experiments/2026-09-20-perception-supervised-rl.md)
has now completed its bounded supervised-controller diagnostic. Corrected
plug/opening supervision and oracle triangulation passed earlier, and the later
natural-cable observation-only estimator was frozen despite its remaining p95
tail. A matched action-only versus pose-conditioned GRU comparison then reached
**0/8 strict insertions in both arms** on new development starts. The policy
learned to make almost the same commands when pose inputs were zeroed or
shuffled. The predeclared stop rule therefore keeps RL, SEER adaptation, and
predictive world-model control parked.

The approved explicit-correction continuation is also complete. Force-safe
collection retained 322 fit rows from 25 episodes with all eight lateral
directions paired, plus 124 calibration rows from seven unseen-template
episodes on two paired axes. A 66,596-parameter correction was forced to pass
through the predicted plug-to-opening translation on top of the frozen
546,686-parameter nominal GRU. On calibration it reduced translation command
MAE from 0.04786 to 0.04638 mm, only 3.10%, and produced a 0.02909 mm median
correction; the fixed gates required 5% and 0.030 mm. A second checkpoint
selected directly on translation MAE still failed, shuffled pose slightly beat
correct pose on calibration, and zero pose beat correct pose on the untouched
offline development split. The six live starts were therefore not opened. No
autonomous rollout, final evaluation, or RL ran. See the
[explicit-correction record](experiments/2026-09-22-explicit-pose-correction.md).

The latest explicit-opening continuation used 20 train configurations, six new
development configurations, original-resolution 160x160 crops, six landmark
heatmaps, and per-observation three-camera calibration. Calibration rejected
the landmark and temporal variants and selected the coarse RGB locator. Across
240 near-port development decisions it reached **1.085/1.740 mm median/p95
lateral error**, 91.5% correction-direction accuracy, **1.204/2.928 mm
axial error**, and 0.211/0.363 degree orientation error. Complete inference was
1.406/1.982/58.099 ms p50/p95/p99. A world-coordinate temporal filter improved
lateral error to 0.862/1.505 mm only as a post hoc, promotion-ineligible
diagnostic. The accuracy gate remains closed.

A subsequent calibration-only ImageNet MobileNetV3-small landmark model was
close enough to earn six further fresh development starts. It is only 233,662
parameters and uses a 160x160 native crop. On 173 near-port decisions it reached
the current best observation-only result: **0.278/0.670 mm median/p95 lateral
error**, 94.1% direction accuracy, **0.664/1.363 mm axial error**, and
0.215/0.506 degree orientation error. Complete inference was 5.450/5.949/71.825
ms p50/p95/p99. It still fails the fixed 0.25/0.5 mm gate, so policy and RL
remain stopped.

A post hoc causal world-space opening filter reached 0.250/0.616 mm lateral
error and 97.8% direction accuracy, while axial error regressed to
2.085/4.407 mm. It was tested after that development split was opened and is
diagnostic only. Since its p95 still failed, no additional split was collected.
Heatmap-confidence camera rejection also failed to improve calibration, which
selected all three views; no further development collection was justified.

The approved temporal/multiview continuation is now complete. A replay audit
found strong cable occlusion in the sparse center-camera mask sample but clear
side views, and no independent cable-shape reset variable. It also fixed a
null episode-index bug so causal history resets at actual simulator resets.
Learned current-frame multiview and six-step temporal heads passed calibration,
then were evaluated once on six new reset positions. The current-frame head was
best on fresh data at **0.307/0.573 mm** near-port lateral median/p95, 99.4%
direction accuracy, and **0.304/0.796 mm** axial median/p95. The temporal head
reached 0.289/0.667 mm and did not reproduce its calibration p95. Full temporal
perception plus the existing trunk measured 6.82 ms p95. Accuracy still fails
the 0.25/0.5 mm gate, so no policy, final split, or RL was opened. The next data
collection must independently vary cable shape/occlusion and retain visibility
or depth supervision before another temporal attempt. The articulation has 40
cable joints in addition to the six arm joints, while the current reset event
selects only the arm. A safe follow-up needs seeded cable-shape templates,
physics settling, collision/force rejection, and a compensating arm solve that
restores the requested plug pose; unconstrained cable-joint noise would confound
the experiment.

The cable-shape/visibility continuation and its force-safe follow-up are
complete. Exact 40-joint resets initially caused a 35 N transient and plug
drift. A closed-loop collection controller now waits for settling, rejects
force above 5 N, restores the measured plug pose, and releases to the fixed
CheatCode teacher only after two safe observations. This is privileged data
generation and is never available to the autonomous actor.

The full natural-trajectory run saved 1,500 causal decisions across 50
episodes. Thirty-five episodes handed off successfully; filtering retained 460
safe near-port observations from all five included cable templates. The frozen
complete-template split used 205 fit, 79 calibration, and 175 development rows.
The calibration-selected current-frame model reached **0.322/0.626 mm**
near-port lateral median/p95 and 100% direction accuracy on 153 samples from ten
held-out episodes. A fixed temporal variant reached **0.247/0.617 mm** but still
failed the p95 requirement. Combining static and natural data was worse at
0.331/0.753 mm. Complete perception plus the existing trunk measured 7.18 ms
p95. The 0.25/0.5 mm gate still fails, so no policy, RL, or reserved-final
evaluation was opened. The subsequent frozen-model audit found that the right
camera was strongest and the center camera weakest, but every two-camera subset
had worse lateral p95 than the three-camera model. Exact labelled pixels gave a
sub-0.0011 mm two-camera triangulation ceiling, confirming consistent camera
geometry. A fixed per-camera bias correction worsened p95 to about 0.79 mm.
The tail was concentrated in one cable episode and small-motion rows; clear
views were worse than occluded bins, so occlusion is confounded with trajectory
rather than established as the cause. More independent safe cable layouts and
camera-specific landmark supervision are required before another frozen
representation test.
See the [execution record](experiments/2026-09-22-cable-visibility-perception.md).

A label-free qualitative audit now overlays only model predictions and their
three-camera reprojections; it never reads simulator plug/port pose labels from
the reviewed episodes. The predicted opening aligns visually with the SFP cage,
including cable-obstructed examples, and cross-view reprojection RMS is
0.807/1.309 pixels median/p95. The predicted stationary opening nevertheless
jitters by 0.369/1.316 mm and its normal by 2.98/7.97 degrees. This confirms the
model is attending to the intended region while independently motivating causal
temporal fusion; it is not a substitute for the labelled accuracy metrics.

The frozen-pose supervised control diagnostic is complete. Two matched
546,686-parameter policies used six causal decisions and emitted four 6D
TCP-body-frame delta commands. One received the frozen visual features, state,
measured state change, force, previous action, and visibility; the other also
received predicted relative translation, uncertainty, and observation-derived
phase. On offline held-out episodes, action-only was marginally better
(0.0367/0.0894 mm translation MAE/p95 versus 0.0369/0.0908 mm). On eight new
autonomous starts each, both scored **0/8 strict insertions**. Best lateral
p50/p95 was 1.479/1.678 mm for action-only and 1.417/1.871 mm for
pose-conditioned; terminal medians grew to 6.382 and 6.284 mm. A counterfactual
audit showed that zeroing pose changed development commands by just
0.0095/0.0264 mm p50/p95, so the generic BC head largely ignored the added
signal. Full inference was 7.135/7.388/8.249 ms p50/p95/p99. The final split
stayed sealed and no RL ran. See the
[execution record](experiments/2026-09-22-pose-conditioned-gru-policy.md).

The selected full-data world policy remains the strongest preserved Gazebo
controller: 2/20 full insertions, 9 partials, mean official score 45.14. Its
learned dynamics is worse than persistence, so predictive imagination remains
disabled.

The September 20 Isaac continuation resolved the earlier joint/reset diagnosis.
The expert joint convention is usable; the original supported YAML had placed
the port about 552 mm from the expert pose. A second silent issue then showed
that the first generated scene variations retained a joint-state reset and
ignored their requested Cartesian starts. The corrected reset uses 6D damped
IK and reproduces requested 4--10 mm axial and approximately 0.7 mm lateral
starts within the declared tolerances.

On those corrected starts, the privileged adaptive guide achieved 9/9 strict
insertions in its first pass and 7/8 in its second. The world-policy replay path
stores exact 24D four-command transitions, 384D frozen causal features, guide
labels, and commands actually executed. The trainable 333,024-parameter head
receives nonzero gradients; its frozen perception/world trunk has 22,236,320
parameters.

The autonomous gate still fails. The round-one aggregate, round-two aggregate,
and varied-only heads each achieved **0/4 strict insertions** on heldout IK
starts. The varied-only head was the most consistent: all four trajectories
reached within 0.10 mm before the opening plane, but lateral offset at closest
approach remained 3.25--3.89 mm against the 0.5 mm success threshold, and force
reached the configured 35 N clip. The four reserved final starts remain
untouched. No Gazebo transfer or actor-Q update was started because the
supervised anchor does not insert. The diagnostic checkpoints, videos, and
machine-readable summaries are under
[`outputs/experiments/2026-09-20_isaac_world_rl/`](../outputs/experiments/2026-09-20_isaac_world_rl/).
See the [execution record](experiments/2026-09-20-isaac-world-policy-rl.md).

The privileged guide is an **Isaac CheatCode policy**, not a new Gazebo
implementation. It uses the same ground-truth plug-to-port rigid-transform idea
as the existing Gazebo CheatCode, but is currently a local near-port feedback
servo rather than the Gazebo policy's full approach, settle, minimum-jerk
insertion, and completion-event sequence. The 552 mm finding is a spatial scene
placement mismatch, not a purely lateral port-frame error or proof that every
cross-simulator asset transform agrees.

DAgger fractions apply to every command rather than to a fraction of timesteps:
50% means `0.5 * model + 0.5 * guide`, while 25% means
`0.75 * model + 0.25 * guide`. Replay keeps the unblended Isaac CheatCode chunk
as the BC label and the blended command as the action that produced the next
state. The detailed record also specifies the Isaac shaping reward and makes
clear that it is not an official Gazebo score.

The causal replay audit is complete. Earlier replay joins were unsafe at every
terminal because Isaac had already reset before returning `next_obs`; new
collection retains a separate pre-reset terminal snapshot and all requested
causal decision fields. The frozen 384D pose probe was evaluated on four unseen
episode-grouped configurations. Near the port it reached 0.730 mm median and
2.533 mm p95 lateral vector error with 93.5% correction-direction accuracy.
This fails the predeclared 0.25/0.5 mm resolution gate, so no conditioned policy
was trained. The bounded native 576×512 crop ablation is also complete. Its
learned RGB locator uses simulator masks only as training labels and selects
evaluation crops from RGB. On the matched 225-decision heldout set, near-port
lateral error improved only from 0.962/2.630 mm median/p95 to 0.851/2.154 mm,
while correction-direction accuracy fell from 84.8% to 80.4%. It therefore
failed the material-improvement and absolute resolution gates. Complete live
inference measured 2.96/3.20/182.50 ms p50/p95/p99, satisfying the 300 ms p95
limit. The policy comparison was correctly skipped. The reserved final starts,
actor-Q, Gazebo transfer, reward-model training, and imagination remain closed.
See the
[full execution record](experiments/2026-09-20-isaac-world-policy-rl.md#causal-replay-audit-and-frozen-feature-probe).
The record's [plain-language interpretation](experiments/2026-09-20-isaac-world-policy-rl.md#plain-language-interpretation-of-the-crop-experiment)
explains the crop stages, median/p95 errors, label limitation, recent robotics
alternatives, and why pose prediction alone does not perform insertion.

The crop failure was subsequently traced more precisely. The old labels were the
SFP module centroid and nearest NIC-card mask pixel, not the plug tip and port
opening. An oracle-coordinate ceiling using those labels still failed at 1.132
mm near-port lateral median and 3.141 mm p95, so further fitting to them was
stopped. The Isaac camera projection path was repaired and visually checked on
all three views. The replacement 1,057-decision train and 225-decision held-out
collections are complete. Exact-keypoint triangulation reached 0.124 mm median
and 0.235 mm p95 near-port lateral error, proving the geometric ceiling is
sufficient. The deployable calibration-selected RGB locator reached only 2.380
mm median and 6.947 mm p95, with 82.6% correction-direction accuracy. It failed
the 0.25/0.5 mm and 90% gate. Complete inference p95 was 3.311 ms. Accordingly,
that stage did not train a pose-conditioned policy or RL. A later frozen-pose
diagnostic is reported above; the four final starts remain sealed. See the
[decision record](experiments/2026-09-20-perception-supervised-rl.md#executed-result).

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

- Calibrated Isaac placement across task families and scene variations. One SFP
  port-1/card-0 near-opening scene now has stable reset, correct action direction,
  guide insertion, and terminal validation; broader coverage is unproven.
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

The SFP pose-refitting and Isaac snag-reproduction branches remain parked. The
active continuation begins from the failure modes actually observed in the
post-fix Gazebo evaluator:

1. **Collect compact repeated exact replays.** Repeat the released three-trial
   YAML with event-driven telemetry rather than multi-gigabyte full-rate TF
   bags. Retain plug/port pose, TCP command/state, force, insertion event, named
   contact pairs, and low-rate cameras. Estimate the intermittent SC axial-block
   rate without tuning on a reserved subset.
2. **Name the blocking contact.** Add diagnostic Gazebo contact logging for the
   SC plug, port rim/latch, gripper, cards, board, and cable links. Reproduce the
   near-aligned partial and determine which collision prevents the final 3--5 mm.
3. **Create short recovery starts from real failures.** Snapshot states before
   the observed axial block and larger approach-tracking failure. Validate
   reset identity and cable state; do not synthesize a cable snag label that the
   production run did not exhibit.
4. **Compare two recovery scales.** For local axial blocks, test a 10--15 mm
   measured-path retreat followed by coherent submillimeter/millimeter lateral
   search and reapproach. For the 8 mm approach miss, test a larger retreat or
   collision-safe transport replan. Keep unmodified CheatCode as the matched
   baseline.
5. **Train only after causal labels exist.** Add port-relative BC/recovery data,
   then critics and bounded online SERL, using observation-only RGB/state/force
   inputs and episode-grouped splits. Preserve the supervised anchor and require
   autonomous insertion before promotion.
6. **Use Isaac as a matched accelerator later.** Reopen Isaac only after its
   grasp, board, port, and collision contract reproduces one of the named Gazebo
   failures. A collision-disabled proxy is not training or promotion evidence.

World-model dynamics, SEER, reward-model training, imagination, and the reserved
final split remain parked. If explicit corrective control fails, return to the
perception restart checklist in the
[cable experiment](experiments/2026-09-22-cable-visibility-perception.md#parking-decision-and-restart-checklist).

Preserve actor-only evaluation for the June experiment lineage. Any privileged
guide or action override belongs to a separately labeled diagnostic experiment.
Report metrics for the same timestep/episode together: independent best depth,
lateral error, and orientation values cannot establish insertion.
