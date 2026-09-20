# World-model follow-up: visual fidelity, dynamics, and control

This is the next-work handoff for the September 18 Dreamer-style pilot. Read
the [current status](STATUS.md), [pilot record](experiments/2026-09-18-dreamer-pilot.md),
[dataset guide](DATASETS.md), and [geometry/scoring terms](scoring.md) before
starting. The frozen paired final and the first visual/dynamics follow-up are
now documented in the [completed follow-up record](experiments/2026-09-18-world-followup.md).
Preserve the frozen pilot and final-scene records.

The approved implementation is now tracked in the
[September 19 full-training record](experiments/2026-09-19-full-world-training.md).
It completed a fresh tokenizer on the canonical 250/39 split under a held-out
validation plateau rule, then trained dynamics and supervised control using
native observed endpoints and actual executed TCP-delta commands. The tokenizer
selected update 75,000 and substantially improved held-out reconstruction.
Dynamics still lost to persistence at every tested horizon, while the supervised
policy achieved 2/20 full and 9/20 partial insertions on the sealed final set.
Reward and imagination remain disabled. See the linked record for metrics and
artifacts.

## What the pilot established

- The selected six-view tokenizer reconstructs broad scene geometry, but the
  connector and port remain blurred. The added crops came from images already
  resized to 288×256, so they cannot restore detail removed by that resize.
- The expanded corrected 200 ms world rollout, conditioned on four *actually
  executed* commands, misses the future measured TCP position by 12.06 mm on
  average; predicting no movement misses by 2.29 mm across all 924 strict
  held-out windows. The earlier three-window probe reported 10.33/1.85 mm. This is a future-state prediction
  test, not controller target-tracking error. The longer-horizon fidelity gates
  also failed. See [metric explanation](STATUS.md#september-18-work).
- All 74 pilot expert bags have an official correct-port insertion event, but
  none of their native saved images/states occurs at or after that event. These
  are successful command demonstrations; they supply no synchronized post-event
  RGB/state example. All 74 raw bags do retain short post-event controller-state
  tails, but none contains camera/video messages (audit below).
- The canonical `expert_verified` dataset has 289 BC-eligible episodes
  (268 SFP, 21 SC). BC eligibility does not imply that every source has
  synchronized images and the four executed commands needed for a valid
  200 ms dynamics transition. In particular, the historical CheatCode source
  retains a wall-clock sampling limitation. Keep failed rollouts separate from
  successful BC labels.

## Selected dynamics: future latent and pose audit

The selected dynamics checkpoint is **step 1,000**, SHA256 `839a1e9aaf77045cfe2c3db5aa6c5c2ed019f2cc9b8d271f4954aea23d515f87`. This is the world-only checkpoint selected before BC, not the subsequently updated BC transformer. No training or final-scene tuning was performed for this audit.

The expanded audit evaluates **every strict held-out native transition window**, with actual executed future commands provided only for this dynamics diagnostic. The three horizons contain 924/158/54 windows from 14/9/5 held-out episodes. Each episode receives equal weight after averaging its windows. This extends the earlier three-window probes; sampling and random noise differ, so the earlier report remains preserved rather than overwritten.

| Horizon | Episodes / windows | TCP error: world / persistence / shuffled | Rotation error: world / persistence / shuffled |
|---|---:|---:|---:|
| 200 ms | 14 / 924 | 12.06 / 2.29 / 13.43 mm | 2.28 / 0.237 / 2.45° |
| 400 ms | 9 / 158 | 14.92 / 4.81 / 19.84 mm | 3.24 / 0.390 / 3.94° |
| 600 ms | 5 / 54 | 21.92 / 9.85 / 24.23 mm | 4.97 / 0.942 / 6.28° |

RMS Euclidean position errors are **13.64 / 16.88 / 25.21 mm** for the world model, **2.98 / 5.94 / 10.73 mm** for persistence, and **15.07 / 22.05 / 26.92 mm** for shuffled actions at 200/400/600 ms. This RMS is the square root of episode-balanced mean squared Euclidean distance, not per-axis RMSE; `rmse_summary.json` also records angular RMS.

**The dynamics gate still fails.** The actual-command model is sensitive to actions, but holding the last observed pose is substantially better overall. Shuffled controls use all four verified commands from another validation episode at its nearest elapsed time; source episode, times and native indices are preserved. Their elapsed-time match is approximate, not an exact phase-matched intervention.

### Motion and phase matter

For windows with less than 1 mm true TCP displacement, world errors are **13.54 / 18.33 / 29.19 mm**, versus persistence **0.247 / 0.366 / 0.513 mm** at 200/400/600 ms. Coverage is 14/6/1 episodes and 308/21/3 windows. For the 5–20 mm motion bin, world errors are **9.01 / 13.99 / 23.79 mm**, versus **5.99 / 7.47 / 11.30 mm** for persistence, across 14/7/5 episodes. There are **no ≥20 mm held-out windows** at these horizons.

The final-three-seconds recording proxy gives world TCP errors **11.96 / 14.05 / 29.19 mm**, versus persistence **0.292 / 0.185 / 0.513 mm**, with **13/2/1 episodes and 73/5/3 windows**. This proxy is not an authoritative contact phase. Early/middle/late time bins and true-rotation bins are also recorded. A small high-rotation subset shows better translation than persistence at 400/600 ms, but only 3/2 episodes support it, rotational prediction remains worse, and the shuffled control is better at 600 ms. It does not establish usable action-conditioned dynamics.

### Distance to the physical opening: completed

Recorded scoring TF was matched to 1,690 native observations across all 14
validation episodes; **1,675 passed** the geometry/timing checks. Fifteen early
observations lacked a complete TF graph, excluding 25 transition windows
(75 actual/persistence/shuffled records). No missing transforms were invented.
Accepted controller stamps differ by at most 4 ms, moving cable-tip TF by at
most 2 ms, and native/controller TCP positions by at most 0.265 mm.

The task board is explicitly static and publishes its static poses at 1 Hz.
Its port/entrance transforms were exactly constant across the sampled times in
every episode, so older fixed-assembly header timestamps are treated separately
from moving-tip timestamps. The audit uses the **explicit recorded entrance
frame**, 45.8 mm from the port reference, rather than treating reference distance
as standoff from the opening.

For windows starting **within 5 mm of the entrance**, world TCP prediction error
is **13.94 mm versus 0.570 mm for persistence at 200 ms** (13 episodes / 163
windows), and **20.80 versus 0.360 mm at 400 ms** (5 episodes / 12 windows).
There are **no valid 600 ms windows starting within 15 mm of the opening**.
These geometric bins therefore confirm poor near-opening prediction while
exposing sparse longer-horizon coverage. They are not authoritative contact
labels. Prediction error still compares predicted future measured TCP with the
actual future measured TCP, not a command target.

The [opening-distance report](../outputs/experiments/2026-09-18_dreamer60_pilot/world_opening_geometry_audit/report.json)
contains start/end distance bins, all denominators and exclusion reasons.
Its archive includes compressed full records, exact TF snapshots, timing/source
hashes and scripts. The first broad reader's timeout is preserved; the completed
reader excludes the duplicate ground-truth `/tf` relay and retains authoritative
`/scoring/tf`, static transforms and observations near each requested timestamp.
It runs in a CPU-only rootless container; no policy or model was changed.

### Separate tokenizer loss from future prediction error

Two representative windows were selected using recorded motion and time, before examining prediction error:

- [Largest recorded 600 ms motion: all three cameras](../outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/largest_true_600ms_motion_full_cameras.png): episode 139, start 2.15 s, 18.42 mm displacement.
- [Latest valid 600 ms window in a different episode: all three cameras](../outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/latest_600ms_other_episode_full_cameras.png): episode 193, start 21.5 s, 0.511 mm displacement; its endpoint is still 2.85 s before the last native recording.

Each sheet shows recorded RGB, decoded true latent, decoded actual-action future, decoded persistence, and decoded shuffled-action future at 200/400/600 ms. Additional sheets show all three contact crops. Decoding shares the same true observed prefix; predicted decoders receive no true future latents. Both full-camera sheets were visually reviewed.

True-latent reconstruction already blurs connector/port detail. Predicted latents add drift as the horizon grows, particularly in the near-static example. At 600 ms, the moving example has RGB MSE **0.00425** for true-latent decoding, **0.01026** for world prediction, and **0.02021** for persistence. The near-static example gives **0.00219 / 0.00822 / 0.00229** respectively. These are selected examples across six views, not population-level image metrics. Gross image motion can improve over persistence while precise physical-state prediction remains poor.

Artifacts: `outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/`. `plan.json` pins the sampling/bins before inference; `report.json` includes all records, phase/motion denominators and source indices. The script ran read-only on physical GPU 3 in 18.94 seconds, without altering checkpoint or runtime sources.

### Implication for the next experiment

First require correct near-static pose behavior and meaningful motion prediction on fully recorded short trajectories. A state-change prediction objective with an explicit persistence baseline is a concrete candidate, but its benefit must be measured. Collect enough independent examples for contact and longer horizons before imagination RL. Better reconstruction alone would not repair the observed pose-prediction failure, and lower aggregate MSE alone would not demonstrate control readiness.

## Raw-bag post-event coverage: completed for the strict 74

All 74 original bag metadata hashes were checked against the prior event audit, including 11 archived bags. **None contains any camera/image/video topic.** Every bag has controller-state records after the first official correct-port insertion event, with **0.134–0.922 seconds** between the event's upper time bracket and the last controller stamp. Their controller, command and scoring-TF topics are retained. The native synchronized image/state dataset still has **zero post-event examples**.

This distinction corrects an overly broad reading of “no post-insertion states”: native RGB/state training examples are absent, while raw state-only tails exist. A timestamp after an event does not by itself establish sustained insertion. Matching scoring geometry and events could support a separately audited state-only reward or outcome model; it cannot recover missing RGB frames or justify assigning success to the last saved image. No reward labels, split or training run was changed.

The [all-verified metadata extension](../outputs/experiments/2026-09-18_world_followup/verified_bag_metadata_audit.json)
locates an unambiguous ROS bag for each of the **149 aligned** verified episodes,
including 23 archived bags. All 149 have insertion-event and controller-state
topics, and **zero have camera/image/video topics**. The other 140 historical
episodes link to LeRobot video datasets, but the canonical manifest does not
link them to ROS bags or event timestamps. Metadata alone does not determine
post-event state-tail duration in the additional 75 aligned bags, and it
does not establish historical video frames after insertion.

Evidence is in [the raw-bag coverage report](../outputs/experiments/2026-09-18_dreamer60_pilot/post_event_raw_bag_audit/report.json), with all 74 metadata files and the hashed controller/event reports. This conclusion is scoped to the strict 74, not automatically to all 289 expert episodes.

## Completion and remaining work

- **Completed:** both frozen final sets are archived (20 valid trials each, zero full insertions); all 40 scored rollouts have post-run plug-tip/port-opening geometry; the 289-episode visual provenance and leakage-safe split audit, matched 74-versus-expanded tokenizer comparison, contact-resolution ablation, strict held-out future-latent decoding at 200/400/600 ms, pose/rotation and persistence/shuffled controls, native near-opening TF bins, visual review, and strict-74 raw-bag topic coverage are recorded. See the [follow-up record](experiments/2026-09-18-world-followup.md).
- **Completed full-data continuation:** the fresh tokenizer reached the declared plateau at 78,000 updates and selected update 75,000. Dynamics and supervised policy also reached their declared plateau rules. The sealed final20 evaluation completed 20/20 valid scenes with 2 full insertions, 9 partials, mean 45.14, and 80.77 ms pooled live p95 latency. Dynamics remained worse than persistence, so the planning gate is closed. See the [full-training record](experiments/2026-09-19-full-world-training.md).
- **Remaining for dynamics diagnosis:** authoritative contact-phase labels and enough near-opening long-horizon transitions are unavailable: no valid 600 ms native window starts within 15 mm of the opening. If state-only outcomes become useful, audit event-relative tails in the additional 75 aligned bags; establish the historical videos' timing before treating them as post-event observations.
- **Separate supervised comparison completed:** both same-architecture arms finished 2,500 BC updates and 4/4 valid new development scenes; neither achieved a full insertion. The selected-world arm had one official partial and modestly lower held-out command error, but its fresh-world control used a different seeded random draw from the original world-pretraining ancestor. See the [comparison record](experiments/2026-09-18-world-supervised-init-ablation.md) and [paired live scores](../outputs/experiments/2026-09-18_world_followup/supervised_world_development_archive/paired_summary.json). A stronger repeat should use the exact preserved ancestral random initialization and multiple seeds. Improved near-static state-change prediction and complete near-contact/failure trajectories are still needed; these diagnostics do not authorize imagination RL.

## Work sequence

1. **Close the current comparison and localize failure.** Finish and archive
   the already frozen paired ACT/world evaluation before changing either
   model. For each complete rollout, measure plug-tip position in port
   coordinates: axial distance to the entrance, lateral offset, orientation,
   measured TCP motion, published target, and insertion/contact events. Use
   official scores and inspect the one-second videos. A reported 4–5 cm
   plug-to-port-reference distance alone does not mean the tip is 4–5 cm in
   front of the opening: the SFP entrance is 45.8 mm from that reference.
   Identify approach, alignment, and seating failures separately. Do not tune
   on the frozen final scenes; make a new development split for follow-ups.

2. **Audit the larger visual dataset without another simulator run.** Use all
   eligible expert images for a tokenizer experiment after checking source
   provenance, grouped scene splits, RGB/BGR convention, resizing, camera
   order, and near-port frame coverage. Keep the original 74-episode tokenizer
   and held-out scenes as a comparison. Compare 74 versus all eligible images
   with the same architecture, resolution, update budget, and selection rule.
   Keep source/task-balanced sampling visible so the 140 historical SFP
   episodes do not silently dominate the 21 SC episodes.

3. **Measure tokenizer reconstruction directly.** On held-out early, approach,
   near-opening, and last recorded frames, save side-by-side original/decoded
   images for each full camera and contact crop. Report whole-image and
   plug/port-region pixel MSE or RMSE, plus a local edge/detail measure.
   Identify crops where the target is absent. Inspect the actual images;
   whole-image MSE can improve while the small insertion feature stays blurred.
   Try a native-resolution contact crop or higher-resolution encoder input as
   a separate ablation if more-data-only training still blurs the port. Check
   the inference size and live latency after any resolution change.

4. **Separate decoder and dynamics errors.** First decode a *true* held-out
   next-frame latent to show the tokenizer's best available reconstruction.
   Then start from the same real observation, feed the recorded executed
   commands into the world model, decode its predicted next latent, and place
   both decodes beside the actual next image. Repeat at strict 200, 400, and
   600 ms horizons only where native observations and complete action windows
   exist; report episode counts per horizon. Compare predicted and observed
   TCP position with Euclidean error in mm and RMSE, orientation with angular
   error in degrees, and force/contact state where valid. Stratify by motion
   magnitude and distance to the opening. Include no-movement persistence and
   shuffled-action baselines. Pose error must compare predicted *future
   measured pose* with the actual future measured pose, never a command target
   with measured pose. A model useful for planning should outperform the
   simple baselines on held-out phases and respond correctly to actual actions.

5. **Test control value before scale-up.** Train the same policy architecture
   and action contract with and without world pretraining, using identical
   supervised data, split, runtime, and development scenes. Compare with the
   corrected TCP-delta ACT baseline. This isolates any benefit of pretraining
   from model size and policy-head differences. If dynamics still loses to
   persistence, keep imagination planning disabled even if supervised control
   improves. Consider predicting state change relative to the current measured
   state and explicitly modeling controller response/contact as a later
   dynamics ablation. Increase model size only after visual and data checks
   identify a capacity limit.

6. **Repair outcome coverage, then consider faster simulation.** Audit raw
   bags for post-event RGB/state before recollecting. If absent, collect a
   short timestamped observation tail after success and failure, with every
   applied command tied to its reference observation. Prefer targeted
   near-opening attempts over repeating long full Gazebo episodes. Isaac can
   later supply parallel expert or RL experience, but first demonstrate
   stable resets/zero-action behavior, matching TCP-delta action and camera
   semantics, and expert insertion on shared scenes. Its reward is not the
   official Gazebo score. Evaluate frozen Isaac-trained policies in Gazebo and
   use Gazebo experience to measure and close the simulator gap.

For each new run, record the exact source revision/diff, data manifest,
checkpoint, scene split, number of native transitions per horizon, reconstruction
sheets, pose-error report, simulator videos, official scores, and real ROS
latency. Keep the existing under-300 ms inference requirement. The pilot's
reward and imagination heads should remain disabled until post-event labels
and held-out dynamics fidelity support them.
