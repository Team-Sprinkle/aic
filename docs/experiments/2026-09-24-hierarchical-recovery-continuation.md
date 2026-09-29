# Hierarchical recovery continuation: execution plan and log

Date: 2026-09-24
Status: one constructed five-card SC mechanics scene repaired and probed;
camera/perception, broader mechanics, BC/RL, and evaluation gates remain open

Follow-up: the [Isaac SC fidelity repair](2026-09-24-isaac-sc-fidelity-fix.md)
replaced the wrong-frame port colliders, calibrated the Gazebo SC grasp and
first cable links, and retained normal scene collisions. A scripted full-start
five-card probe reached 0.073 mm axial / 0.054 mm lateral error; two near-port
offset probes produced 54/92 N plug/port blockage. The old Isaac gripper/card
contact disappeared on the centered route after the repair, so that old trace
is not an accepted failure label. New [corrected-asset routes](2026-09-24-isaac-sc-contact-routes.md)
deliberately crossing cards reproduced named gripper/card and plug/card
blockages. Cable links contacted a card only during a gripper-led jam; an
independent cable snag remains unproven. Their high forces exclude them as
training demonstrations.
The original camera attempt failed; a later container-local driver-check
override passed a short [three-camera SC smoke test](2026-09-24-isaac-sc-fidelity-fix.md#camera-follow-up-september-24).
One scene and four static frames cannot pass the 0–5-card SC mechanics,
moving-cable appearance, or autonomous-policy gates.

The [core controller design](../HIERARCHICAL_RECOVERY.md) explains the safety
supervisor, checkpoint escalation, learned actor boundary, and prior art.
The [September 23 ranking](2026-09-23-ranked-failure-scenarios.md) and
[five-card route evidence](2026-09-23-fixed-five-card-route-probe.md) are
frozen starting evidence. This plan uses a maximum of four concurrent GPUs,
rootless Docker, episode-grouped development splits, no reserved-final tuning,
and complete inference p95 below 300 ms. As of the user's 2026-09-24 update,
the former one-hour-per-issue diagnosis cap is removed: investigate and fix
blockers iteratively when useful. An unresolved gate is recorded explicitly and cannot
be silently bypassed by privileged or collision-disabled data.

## What is known about the abrupt five-card stop

The normal-collision fixed Gazebo scene comes from request seed 51500, one SC
port and five NIC cards. The diagnostic policy uses privileged TF only to
execute a predeclared route; it is **not an autonomous policy**. The selected
route crosses the card row at TCP lane `x=-0.4776 m`, then hands off to stock
CheatCode. Its outside-left control uses `x=-0.5686 m`. The saved exact scene
YAML and route plans are under
`artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/`;
the command and rootless image digest are in the [route record](2026-09-23-fixed-five-card-route-probe.md).
One of three original across-card runs scored no insertion, with cable link 5
only 0.9 mm from a main PCB center-box approximation and virtually stationary
later. In seven subsequent 20 Hz wide-view repeats of the same route, six
scored partial and one no insertion. The latter reached 52.9 N peak raw wrist
force norm, stopped 38.7 mm before the opening with 4.45 mm lateral error, and
ended with 111.6 mm TCP command error. Its cable lay across card tops, but its
link 5 was **not** stationary like the first candidate. Named contact pairs
were not logged; cable snag, gripper/card collision, and misaligned approach
cannot yet be separated. [Official physics notes](https://discourse.openrobotics.org/t/note-to-participants-gazebo-bug-fix-changes-to-friction-properties-of-the-cable/54098)
and [open CheatCode issue #396](https://github.com/intrinsic-dev/aic/issues/396)
do not establish this run's cause. The official evaluation image is already
the post-friction version; the ghost-collider bug concerned accumulated trials.

The 20 Hz [combined failure video](../../artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/smooth_wide_repeat_06/visuals/all_views_20fps.mp4),
[overhead view](../../artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/smooth_wide_repeat_06/visuals/overhead_20fps.mp4),
[side view](../../artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/smooth_wide_repeat_06/visuals/side_20fps.mp4),
and [overhead contact sheet](../../artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/smooth_wide_repeat_06/visuals/overhead_20fps_contact_sheet.jpg)
are durable compact artifacts. Bulk JPEG frames and MCAP bags remain under
`/var/tmp/chmin_aic_cable_route_smooth_20260923/`. The historical failed
VLM/MoveIt five-card bag was not found in the checked S3 prefixes; only its
written note and an accepted dataset/video survive.

## Ranked failure and simulator matrix

Likelihood is a qualitative priority from saved rollouts, **not** a measured
production frequency. Reproducibility refers to a valid, retained incident.

| Priority | Observed class and evidence | Gazebo | Isaac | Training/recovery choice |
| --- | --- | --- | --- | --- |
| 1A | Learned lateral/orientation drift; several SFP videos miss opening by 5–18 mm | Bounded start offsets available | Existing near-port starts | BC alignment and cautious approach; no backtrack without blockage. |
| 1B | SC axial/port-lip partials; exact Trial 3 and five-card rail 1 | Reproducible, official scores | Zero-card local correction worked; full SC contract still suspect | Local measured-pose retreat, lateral/rotation retry, reinsert. |
| 2 | Large approach/tracking block; five-card SC rail 0 and route-probe no-insertion | Reproducible, cause unresolved | Gripper/card hit reproduced, but grasp/collision fidelity gate fails | Retain as ambiguous obstruction; route-level checkpoint and detour after contact audit. |
| 3 | Cable/card trap or tension; one across-card route-sensitive candidate and historical note | Visually reproducible candidate, no named contact proof | Not yet collision-faithful for five-card SC | Instrument named contact/tension; use only reviewed incidents, not presumed snag labels. |
| 4 | Reset/stale-command infrastructure faults | Historically observed | Reset identity measurable | Reject invalid episodes and repair engine; never train actor to compensate. |

The Isaac column above records the **pre-repair ranking**. The corrected
five-card scene supports item 1B local port blockage and a deliberately
constructed item-2 gripper/card blockage as physics diagnostics. The old
gripper/card incident did vanish in the centered full-start probe; a new
across-card route reproduces a genuine named scene contact. A high return route
also jams the plug against a card. Item 3 remains unproven as an **independent
cable snag** because sampled cable/card contact occurs during a gripper jam.
See the [repair metrics](2026-09-24-isaac-sc-fidelity-summary.json) and
[route controls](2026-09-24-isaac-sc-contact-routes.md).

If neither simulator reproduces a plausible failure within the bounded search,
record a **coverage gap** and its likely evaluation relevance. Train generic
force/stall detection and safe retreat on similar observed contacts, but do
not claim cable-specific recovery. Gazebo-only incidents may seed slow Gazebo
RL or supervised recovery; Isaac-only incidents are development hypotheses
until Gazebo verifies the failure mechanism. A simulator mismatch must not be
fixed by suppressing collisions in data used for a promoted policy.

## Ordered execution steps and gates

1. **Freeze and index evidence.** Preserve exact scene YAML, official image
   digest, commands, bags, force/TCP/plug geometry, videos, scores, source
   hashes, and invalid attempts. Add a causal contact-pair recorder if the
   current bag lacks it. Pass: repeatable clean single-trial run and explicit
   failure classification or `unknown contact` label.
2. **Collect development failures by class.** Start with Gazebo's reproducible
   1A/1B/2 scenarios and the five-card route candidate. Capture full episodes,
   terminal observations, commands versus measured motion, force, camera views,
   and recovery mode. Build short pre-contact resets only if robot **and cable**
   state can be restored faithfully; otherwise use nearby seeded full resets.
   Split by parent episode/layout/seed. Pass: matched held-out incidents for
   each trainable class; preserve unsuccessful collection.
3. **Repair the Isaac SC mechanics contract.** Compare Gazebo/Isaac grasp,
   reversed cable colliders, joint transforms, contact and physical insertion
   depth across 0–5 cards. Pass: collision-faithful scripted normal SC
   insertion plus valid reset/terminal identity. If this gate fails, use
   Gazebo-only data and do not train mislabeled Isaac SC RL.
4. **Audit existing SC supervision and perception.** Verify the 21 canonical
   SC episodes' official scores, camera/action timing, complete target chunks,
   port-frame conversion, and terminal events. Audit additional successful
   historical episodes separately. Measure held-out SC pose and orientation
   error by card count and occlusion. Pass: compatible SC training labels and
   observation-only pose estimates adequate for control; no GT actor input.
5. **Train a full-action SC BC warm start.** Use recorded teacher target
   trajectories converted into the selected port frame; predict complete
   actions, not residual corrections. Evaluate SC-only and jointly
   task-conditioned SFP+SC on identical episode-grouped development starts.
   Use existing verified data first; add successful routes around cards only
   for missing coverage. Pass: autonomous full-depth SC insertion with safe
   force on new starts, including multi-card cases. Preserve all failed runs.
6. **Implement hierarchical recovery.** Store measured SE(3) checkpoints and
   route phase; trigger on force plus stalled progress; retrace safely, release
   contact, let the learned actor choose a distinct local retry. Escalate to
   an earlier route checkpoint after repeated same-corridor failure. Track
   counts, tried directions, force, clearance, and actor ownership. Pass:
   deterministic unit and live incident tests with no false retreat on normal
   insertion; identical implementation semantics for Isaac and Gazebo.
7. **Matched frozen-actor ablation.** Compare no recovery, current local
   recovery, and hierarchy on the *same* BC checkpoint, incident starts,
   budgets, and development scenes. Report full insertion, incident release,
   peak/integrated force, path/time, retry counts, and p95 latency. A hierarchy
   helps only if it improves held-out full insertion without force regression.
8. **Offline critic gate.** Build prior replay from verified successes,
   failures, and scripted recoveries; keep terminal states and actor-ownership
   masks. Train critics while actor is frozen. Compare direct online start
   against offline critic initialization; use it only if held-out episode
   ranking/calibration improves. Offline SERL is optional, not assumed useful.
9. **Online SERL on reproducible short incidents.** Start from BC and the
   validated critic if it passes. Balanced prior/online replay, full-mixture
   behavior anchoring, and calibrated reward/force limits. Train matched no,
   local, and hierarchical recovery arms, including an actor-only correlated
   exploration reference. Keep updates, starts, and wall budgets equal. Pass:
   held-out recovery/full insertion improvement with no mode collapse.
10. **Scale only after the hierarchy gate.** Train task-conditioned BC on all
    compatible verified SFP and SC data (and more SC only where missing).
    Scale RL on failure starts by class and card count until learning and
    uncertainty stabilize; episode count is evidence-driven, not a fixed
    target. Recheck complete normal episodes and SFP regression.
11. **Isaac-to-Gazebo transfer.** Validate observation/action/force/reset
    parity. Use Gazebo supervised correction or replay first, then repair the
    existing Gazebo RL bridge as needed for valid terminal accounting and
    sequential online updates. Gazebo need not match Isaac throughput; use
    short incidents and validated critic initialization. Pass: Gazebo
    autonomous full insertion and safety on new development layouts.
12. **Official evaluation.** Freeze checkpoint, controller thresholds, and
    routing logic before running the released `eval_config.yaml` unchanged
    with all three trials. Report by SFP-to-NIC/SC-to-SC, full/partial/none,
    force, incidents, resets, and latency. Preserve exact image digest,
    scene config, artifacts and videos. Open reserved final configurations
    only after the development gate, once, with no tuning afterward.

After each step: if the pass condition holds, continue. Investigate a blocker
iteratively while there is a useful path to a fix; record the concrete failed gate,
artifacts, and next feasible independent work. Never treat a proxy, invalid
collision ablation, or a changed scene as a passed gate.

## Execution log

| UTC | Step | Result | Artifacts / next gate |
| --- | --- | --- | --- |
| 2026-09-24 00:34 | Plan and evidence review | In progress | This record and core design. |
| 2026-09-24 00:42 | Step 1 incident index and SC audit | Indexed 24 retained Gazebo episodes: 8 full, 11 partial, 5 none. Only the latter are generic failure candidates, not confirmed snags. Canonical SC experts: 21 full, 14 train/7 validation, NIC counts 1/2/3 = 11/6/4. No canonical SC NIC4/5 coverage. | `outputs/experiments/2026-09-24_hierarchical_recovery/incident_index.json`; reproducible command below. |
| 2026-09-24 00:55 | Step 1 causal trace review | Re-extracted all seven fixed-route 20 Hz Gazebo bags, including measured TCP SE(3), target movement, tracking error, and wrist force vector. Raw force norm has about 21 N resting load; subtracting its magnitude misleadingly gave **0/7** instantaneous triggers and **2/7** longer-window triggers. Correct vector taring gives **7/7** early instantaneous triggers and **7/7** later progress triggers, including six partial insertions. Neither rule distinguishes a cable obstruction. These repeats are **one scene group** and cannot calibrate generalization or identify contact. | [Trace extractor](../../scripts/extract_recovery_trace.py), [read-only diagnostic](../../scripts/analyze_recovery_traces.py), local `outputs/experiments/2026-09-24_hierarchical_recovery/wide_repeat_XX_trace.jsonl` and `trace_diagnostic.json`. |
| 2026-09-24 00:56 | Step 6 prototype, offline only | Implemented measured-SE(3) local/route checkpoint logic with force-release and budget aborts; four isolated tests pass. No Gazebo or Isaac deployment was changed. The trigger failed specificity on the retained trace test and is unsafe for deployment. | [Controller](../../aic_example_policies/aic_example_policies/ros/hierarchical_recovery.py), [tests](../../aic_example_policies/test/test_hierarchical_recovery.py). |
| 2026-09-24 01:00 | Step 2 cross-scene trace audit | Read ten additional targeted Gazebo bags and retained 17 scored 20 Hz traces across five scene/config groups. Peak TCP target-tracking error was 55.8–58.0 mm for six full, 61.2–72.6 mm for eight partial, and 89.2–126.8 mm for three no-insertion episodes. All three no-insertion runs crossed 80 mm at least 10 s before their terminal trace; none of the other 14 did. The threshold was chosen after inspecting outcomes and has no held-out calibration. High error identifies a control/physical obstruction candidate, **not** cable contact. | [Cohort summarizer](../../scripts/summarize_recovery_trace_cohorts.py), local `trace_cohorts.json` and `targeted_01..10_trace.jsonl`. |
| 2026-09-24 01:15–01:29 | Step 2 fresh-scene check | Ran three new five-card SC Gazebo scenes with normal collisions. Scenes 01/02 used the same privileged across-card route then stock CheatCode; scene 03 used stock CheatCode on a rail-0 layout shifted 7 mm in board X and 4 mm in Y from a prior no-insertion layout. Outcomes: one full (Tier 3 75, total 82.67) and two partial (Tier 3 39.67/39.00, totals 47.79/52.50). Peak tracking errors were 56.56, 71.49, and 75.53 mm; none crossed the post-hoc 80 mm line. All briefly had vector-tared force above 20 N without a force penalty. No new no-insertion case was obtained; sensitivity remains unvalidated. | [Scored scene summary and hashes](2026-09-24-hierarchical-recovery-fresh-scenes.json), [scene 01 video](../../artifacts/hierarchical_recovery/fresh_scene_01/all_views_20fps.mp4), [scene 02 video](../../artifacts/hierarchical_recovery/fresh_scene_02/all_views_20fps.mp4), [scene 03 video](../../artifacts/hierarchical_recovery/fresh_scene_03/all_views_20fps.mp4), [tracked scene YAMLs](../../configs/hierarchical_recovery/). |
| 2026-09-24 01:20 | Step 4 SC target-frame audit | Located all 21 canonical SC scored bags and retained their TF chains. Converted **13,349 recorded teacher TCP targets** into the selected SC receptacle base frame, preserving complete episode split and source scene hash. 13,016/13,286 possible four-command windows had consecutive indices and no gap over 250 ms; 84 command indices and nine longer simulation-time gaps remain. A valid window's first-to-fourth elapsed simulation time ranges 150–450 ms, so it is **not** automatically an exact four-by-50-ms RPDP chunk. Port-frame TCP targets are not connector-waypoint labels, and no observation-only SC port estimator has been validated. No SC BC training was started. | [Port TF manifest](2026-09-24-sc-expert-port-edges.json), [label audit](2026-09-24-sc-port-target-audit.json), local `sc_port_tcp_teacher_targets.jsonl`, [converter](../../scripts/audit_sc_port_targets.py). |

```bash
python scripts/build_recovery_incident_manifest.py \
  --output outputs/experiments/2026-09-24_hierarchical_recovery/incident_index.json
```

The incident index groups all seed-51500 routed repeats together; they are
**not** independent scenes for train/validation. The existing bags retain
commands and measured trajectories, but no named contact pair. The index is a
discovery manifest, not yet a causal transition replay. The canonical 21 SC
episodes do have official Tier 3 = 75 and reconstructed teacher target labels;
their narrow card-count and fixture coverage is the immediate BC data gap.
Portable copies of the small [incident index](2026-09-24-hierarchical-recovery-incident-index.json),
[trace diagnostic](2026-09-24-hierarchical-recovery-trace-diagnostic.json),
and [cross-scene trace metrics](2026-09-24-hierarchical-recovery-trace-cohorts.json)
are tracked beside this report. The raw MCAP and 20 Hz observations remain
under the stated bulk roots.

### Trace diagnostic command and interpretation

The raw MCAP reader runs inside the pinned official image with ROS sourced.
For each repeat, mount the read-only bulk bag root and run:

```bash
python3 /repo/scripts/extract_recovery_trace.py \
  /bulk/across_repeat_06/results/bag_trial_000001_20260923_234345_864 \
  /repo/outputs/experiments/2026-09-24_hierarchical_recovery/wide_repeat_06_trace.jsonl
python scripts/analyze_recovery_traces.py \
  outputs/experiments/2026-09-24_hierarchical_recovery/wide_repeat_{02..08}_trace.jsonl \
  --output outputs/experiments/2026-09-24_hierarchical_recovery/trace_diagnostic.json
```

The bag's first-five-second wrist-force norm is about 21 N, so the naive
untared 8 N threshold is invalid. Repeat 06's peak raw norm is 52.86 N;
subtracting the three-axis initial baseline before taking magnitude yields
72.3 N, consistent with the official scorer's 72.16 N maximum insertion
force and 2.03 s above 20 N. The scorer reports no **off-limit** contact,
which does not identify the ordinary contact pair. At 90–95 s it moved
only 3.9 mm while target commands accumulated 23.6 mm and tracking error grew
from 44 to 66 mm. At 95–100 s it moved 0.7 mm while error grew another 23 mm.
An earlier force pulse and this later divergent tracking are present, but
they are not unique to the failed run. With correct vector taring, the naive
two-sample trigger fires in **all seven** runs as early as 26–59 s; a 10 s
force latch plus 2 s progress window also fires in **all seven** at 73–98 s.
Six of these runs still achieved partial insertion. This is a failed
specificity test, not a successful obstruction detector. Subtracting force
*norms* instead of force vectors appeared to mark only repeat 03 and 06,
but that computation hides force direction and is physically wrong. Do not
wire this trigger into live control. Next, classify named contacts and
command/controller-limit events, then calibrate on independent scene groups
with full-insertion controls and explicit timing relative to the insertion
event. No controller ablation or RL replay from these candidate labels is
valid yet.

The ten targeted trial bags broaden the observation check. The 80 mm TCP
command-tracking-error line separated **3/3 no-insertion** from **14/14
full-or-partial** runs across the combined 17 traces, and crossed 10–20 s
before the end of each failure. This is a *post hoc* threshold on only five
scene/config groups. It can nominate a large-error incident for review, but
cannot distinguish cable trap, gripper/card collision, an unreachable target,
or poor insertion alignment. It cannot be promoted to a recovery trigger
until a new scene-grouped validation set and named contact/controller-limit
audit test false triggers during legitimate full-depth insertion.

Three new development scene variations kept five cards and used the same
pinned image, normal collision geometry, and 20 Hz wrist plus wide cameras.
The first two used the long across-card route with different board/SC-rail
positions; the third used stock CheatCode in a nearby rail-0 scene. All are
controls for the 80 mm diagnostic: one full and two partial, none falsely
crossed it. The [overhead sheets for scene 01](../../artifacts/hierarchical_recovery/fresh_scene_01/overhead_20fps_contact_sheet.jpg)
and [scene 02](../../artifacts/hierarchical_recovery/fresh_scene_02/overhead_20fps_contact_sheet.jpg)
show the cable draped across the cards in both. [Scene 03](../../artifacts/hierarchical_recovery/fresh_scene_03/overhead_20fps_contact_sheet.jpg)
shows the stock route. It became partial even though its nearby parent layout
was no-insertion, so the outcome is sensitive to small scene changes. These
do not identify a cable contact or test learned recovery. The exact config,
scored bag, trace, video
hash, and metrics are in the tracked fresh-scene JSON. Bulk bags and frames
remain under `/var/tmp/chmin_aic_hierarchical_recovery_20260924/`.
The rootless reproduction command for scene 03 is:

```bash
mkdir -p /var/tmp/chmin_aic_hierarchical_recovery_20260924/fresh_scene_03_replay
cp configs/hierarchical_recovery/fresh_scene_03.eval_config.yaml \
  /var/tmp/chmin_aic_hierarchical_recovery_20260924/fresh_scene_03_replay/eval_config.yaml
cp configs/hierarchical_recovery/{world_audit.sdf,bridge_audit.yaml} \
  /var/tmp/chmin_aic_hierarchical_recovery_20260924/fresh_scene_03_replay/
AIC_AUDIT_SMOOTH_CAPTURE=1 AIC_AUDIT_DEADLINE_SEC=360 \
  bash scripts/run_recovery_gazebo_diagnostic.sh \
  /var/tmp/chmin_aic_hierarchical_recovery_20260924/fresh_scene_03_replay stock
```

It uses GPU 1, the pinned official image, and no `sudo`. The source YAML is
tracked at `configs/hierarchical_recovery/fresh_scene_03.eval_config.yaml`;
the wide-camera world and bridge files are tracked beside it. Use a new output
directory when repeating so the preserved run is not overwritten.

```bash
python scripts/summarize_recovery_trace_cohorts.py \
  --index outputs/experiments/2026-09-24_hierarchical_recovery/incident_index.json \
  --trace-dir outputs/experiments/2026-09-24_hierarchical_recovery \
  --output outputs/experiments/2026-09-24_hierarchical_recovery/trace_cohorts.json
```

### Gate at the SC learning branch

The already retained [Isaac SC mechanics audit](2026-09-23-sc-mechanics-and-routing.md)
shows the source reversed weld targets `ati/tool_link`, while the prepared USD
weld targets a finger. The asset is stable at its initial posture but not in a
near-port full-scene hold; the latter can generate enormous force or miss the
requested tip pose by 44.27 mm even under collision ablation. This is a
physical-contract failure, so those transitions cannot seed SC BC or SERL.
An unrelated long-running Isaac diagnostic process remains active and was not
interrupted. The canonical Gazebo SC set has only 21 successful NIC1–3 episodes,
and the current successful RPDP/BC/SERL pipeline was trained for SFP-to-NIC
near-port starts; it has no audited SC port-pose encoder or converted SC
port-frame dataset. A new SC policy checkpoint or RL result is therefore **not**
established. Gazebo-only SC collection and conversion remain legitimate after
the observation/action timing and scene split are proved; collision-disabled
Isaac proxy episodes do not satisfy that gate.

### SC teacher target-frame audit details

Every canonical SC expert has a retained scored MCAP. The port-frame audit
reads `world -> tabletop -> base_link` and
`aic_world -> task_board -> selected sc_port -> sc_port_base_link` from the
first five seconds of each bag, then converts each recorded **teacher TCP
target** from base coordinates into that port base frame. The scene's `world`
and `aic_world` roots are treated as coincident; the derived port location is
plausible against the teacher's final target but that alias still needs an
independent transform check before use as a metric pose label. The numerical
base→port→base target round trip is below `2e-16 m/rad`; it verifies the
converter algebra, not camera calibration or SC plug pose.

All 21 original episodes contribute 13,349 target-bearing image rows,
14 training episodes and seven validation episodes. There are 84 absent
command indices within their original spans and nine adjacent image gaps over
250 ms, three of which last about 5.1 s near the end. The converter masks
nonconsecutive four-command windows and records exact simulation timestamps.
Among otherwise valid four-command windows, the first-to-fourth target spacing
is 150–450 ms (median 200 ms), so four adjacent *recorded* commands cannot
silently be treated as four exact 50 ms microsteps. Exact time-grid labels
would need a causal join to bag commands and an explicit hold/interpolation
policy, then validation against executed motion. The [frame audit](2026-09-24-sc-port-target-audit.json)
and ignored JSONL are reproducible with:

```bash
# Run the first command inside the pinned official image with /repo and
# /var/tmp/chmin_aic_20260918_act mounted at /bulk.
python3 /repo/scripts/extract_sc_expert_port_edges.py \
  --manifest /repo/outputs/trajectory_datasets/expert_verified/manifest.json \
  --source-prefix /var/tmp/chmin_aic_20260918_act --mounted-prefix /bulk \
  --output /repo/outputs/experiments/2026-09-24_hierarchical_recovery/sc_expert_port_edges.json
.pixi/envs/default/bin/python scripts/audit_sc_port_targets.py \
  --manifest outputs/trajectory_datasets/expert_verified/manifest.json \
  --port-edges outputs/experiments/2026-09-24_hierarchical_recovery/sc_expert_port_edges.json \
  --output outputs/experiments/2026-09-24_hierarchical_recovery/sc_port_target_audit.json \
  --labels-output outputs/experiments/2026-09-24_hierarchical_recovery/sc_port_tcp_teacher_targets.jsonl
```

This proves availability of teacher TCP targets in a port-attached frame, not
readiness for SC RPDP. The connector tip is offset from the TCP, the selected
`sc_port_base_link` is not the `sc_port_base_link_entrance` opening, and the
actor must receive an **observation-only** estimate of that frame at test
time. Using the scored TF in autonomous evaluation would invalidate the
experiment.

**September 24 correction:** the original v1 audit interpreted the recorded
`[x,y,z,qx,qy,qz,qw]` teacher pose as if its quaternion were WXYZ. Its
round-trip check used the same mistaken convention in both directions and
therefore missed the error. Bag 251's terminal port-frame orientation changed
by 153.7 degrees after correction. The regenerated
[v2 audit](2026-09-24-sc-port-target-audit.json) explicitly uses XYZW.
Nothing was trained from the old v1 port-frame output. The new
[connector-waypoint labels](2026-09-24-sc-connector-waypoint-labels-held.json)
also apply the scored TCP-to-SC-tip transform and retain exact command times.

## Current gate disposition

| Step | State | Why it cannot be promoted yet |
| --- | --- | --- |
| 1. Freeze/index evidence | Complete for retained cases | Exact paths, scores, scene group, configs and reproducible extraction are saved; contact pair remains explicitly unknown. |
| 2. Collect incidents | Partial | Seventeen retained traces and three new full/partial controls exist. No new independent no-insertion start or verified cable-specific contact label. Short resets cannot preserve cable state. |
| 3. Isaac SC mechanics | Failed gate | The prior physical near-port hold misses by 44.27 mm after source-collider corrections; source weld frame and prepared USD differ. A long-running pre-existing Isaac diagnostic was left untouched. Collision-disabled proxy motion is ineligible for RL. |
| 4. SC labels and perception | Partial | All 21 verified teacher TCP targets were transformed to a selected port base frame and timing audited. Connector-waypoint/50 ms semantics, opening offset, `world`/`aic_world` frame alias and observation-only SC pose are not fully validated. |
| 5. SC BC | Not started | Its RPDP observation and action contracts are not yet valid for SC; the old mixed ACT 21-SC trial approached but inserted 0/4 development cases. Training another unchecked controller would not pass the autonomous full-insertion gate. |
| 6. Hierarchy | Prototype only | Four unit tests pass; the force/stall detector fires on legitimate partial insertions and cannot own live motion. |
| 7–12. Matched ablation, critic gate, online SERL, scale, Gazebo transfer, official eval | Not opened | They require an SC BC warm start and validated blockage labeling. No new RL checkpoint or production evaluation is claimed. |

The initial trigger-specificity and SC-mechanics investigations ran in the
00:34–01:33 UTC continuation window. The former one-hour-per-issue limit has
since been lifted; the unresolved gates remain unresolved until retested. The
next valid data action is a new scene-grouped set with named cable/gripper/port
contacts and controller-limit status, including full-insertion controls. The
next valid model action is an SC observation-only port estimator with measured
held-out error, followed by timestamp-checked connector-waypoint targets.
Only then does the prescribed SC BC → frozen-actor recovery ablation → optional
offline critic → online SERL sequence open. The reserved final configurations
remain sealed.

### September 24 corrected-asset contact continuation

Seven additional scripted route traces on the same constructed five-card
Isaac scene are archived in the [contact-route record](2026-09-24-isaac-sc-contact-routes.md).
The two across-card repeats produced sustained named gripper/card blockage;
an instrumented repeat also recorded secondary cable-link/card contact. A
nearby lane had no sampled card contact, and a high return route produced a
separate plug/card jam. The programmed reversal cleared the across-card
gripper contact, but this is not learned recovery and the hundreds-of-newtons
loads preclude directly using those commands as training demonstrations.
This upgrades item 2 from a suspected pre-repair artifact to a constructed,
named corrected-scene incident. Item 3 remains a coverage gap. Gazebo's saved
bridge only logs filtered off-limit contacts, not cable/card pairs. Isaac RGB
now passes a short corrected-scene smoke test; longer moving-cable visuals and
distal cable calibration remain gates for visual SC policy training.
