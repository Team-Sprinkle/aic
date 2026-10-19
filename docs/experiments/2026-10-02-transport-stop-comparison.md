# October 2 transport stopping comparison

This is a stopping-signal diagnostic, separate from insertion BC. The
transport actor brings the measured TCP near the selected SC pre-insertion
standoff. A stop signal may then **continue transport**, **flag uncertainty
for a future measured-pose hold**, or **latch handoff**. The uncertainty state
here is an offline advisory label; no replayed command was changed. The
frozen transport-only head latches when its observation-only probability is
at least 0.95 on two decisions after 15 s. The current visual stop latches
after three accepted near-standoff visual fits. The latter estimates the
selected SC port from the three wrist RGB images, a fixed camera calibration,
and measured TCP pose. Neither stop reads a simulator world pose. All
distances and rotations below are **measured TCP to the intended transport
standoff**, scored after deciding; they are not plug-tip insertion errors.
The public task code selects port 0 or 1. A selected-port index in a trace is
not itself proof that the predicted physical opening was correct.

## Frozen SC development traces

The seven complete frozen trajectories represent five distinct development
episode configurations: both ports, one/three/four/five cards, five correct
routes, and two documented wrong-route repeats. Each row compares both stop
signals on **the same images, measured TCP states, task code, clock, and
motion history**. The visual signal is the recorded causal controller trace;
the transport-only frozen head was replayed on saved RGB with its original
long-history input contract. On an independent logged-head trace, the replay
matched 203 logged probabilities with correlation 0.99991 and identical last
probability 0.92969, supporting replay fidelity. Saved JPEG decoding is
still a replay approximation. The original transport action weights were
unchanged by the frozen head training.

| Saved route | Cards / port | First valid TCP arrival | Transport-only head | Visual stop | Interpretation |
| --- | --- | ---: | --- | --- | --- |
| `llb_0009_v11` | 5 / 1 | 42.15 s | 42.85 s, 0.69 mm / 2.07° | 43.10 s, 0.64 mm / 2.07° | Both valid; head 0.25 s earlier |
| `llb_0184_v1` | 5 / 0 | 55.20 s | Missed | 55.75 s, 1.83 mm / 1.52° | Visual recovers missed handoff |
| `llb_0122_v4` | 3 / 0 | 41.40 s | Missed | 43.70 s, 2.05 mm / 0.28° | Visual valid; 2.30 s delay |
| `llb_0135_v2` | 1 / 1 | 42.70 s | 43.60 s, 1.14 mm / 1.06° | 44.10 s, 0.96 mm / 0.64° | Both valid; head 0.50 s earlier |
| `llb_0193_v1` | 4 / 1 | 42.50 s | Missed | 43.70 s, 0.64 mm / 0.78° | Visual recovers missed handoff |
| `llb_0009_v7` wrong route | 5 / 1 | Never | False: 80.39 mm / 14.39° | **False: 1.02 mm / 6.71°** | Visual selected the right position but accepted bad orientation and unsafe route |
| `llb_0122_v2` wrong route | 3 / 0 | Never | False: 95.10 mm / 6.54° | No latch | Visual avoids this false handoff |

Thus the transport head made **2 valid handoffs, 2 false handoffs, and missed
3 arrivals**; visual stopping made **5 valid handoffs, 1 false handoff, and
missed 0 arrivals** on these paths. The five valid visual delays after first
measured 10 mm / 3° arrival were 0.55 to 2.30 s. The wrong-route visual
failure is material: its 1.02 mm position error alone looks excellent, but
orientation was 6.71° and the route had exceeded 50 mm from the teacher path.
No SC visual signal at the latch claimed the opposite task port, although
the selected-port field is task coded and cannot independently certify image
identity. All seven official score files report no disallowed contact and no
excessive force. Raw wrist force includes tool gravity, so raw norm alone is
not called collision force. Complete recorded inference p95 ranged from
175.7 to 287.9 ms across the seven runs, below the 300 ms limit; p50/p95/p99
per episode are preserved in the machine record. The head-only replay timing
is not a deployable end-to-end latency measurement.

These are **stop decisions conditioned on trajectories already driven by the
visual standoff controller**. They show that visual verification can improve
the stop decision on the same path. They do not prove that switching the stop
rule improves route generation or autonomous insertion.

## Small frozen live check and promotion decision

One serial same-scene, same-controller pair used the independent four-card
SC `llb_0193` development snapshot. The action transformer and visual servo
weights were identical; only the stop selector was switched. Both kept the
same task, cameras, calibration, 60 mm command cap, and visual servo gains.
The transport-head run never entered the measured 10 mm / 3° gate and ended
101.32 mm from the standoff, with teacher-route p95 116.40 mm. Its head
probability stayed below the threshold, so it timed out at 70 s. The visual
run entered the gate and stopped at 0.99 mm terminal position error, with
route p95 26.93 mm. Its frozen RGB trace replay shows that the transport
head would have **missed** that same valid arrival, while visual stopping
latched. Complete live p50/p95/p99 inference was 136/195/264 ms for the
head run and 121/185/205 ms for the visual run. Both score files report no
bad contact or excessive force.

The two actual trajectories diverged long before the stop, as earlier Gazebo
repeats have. Therefore their terminal gap cannot be credited causally to
the stop rule. The same-trace replay favors visual stopping, but its saved
wrong-route false handoff fails a safe general promotion standard. **Decision:
retain visual stopping as a development-only verifier; do not promote it to
a general autonomous handoff or enable insertion after it.** The next repair
is a bounded orientation and route-quality veto tested against this exact
seven-trace set plus new grouped development routes; it must remove the
`llb_0009_v7` false handoff without losing valid arrivals, stay below 300 ms
p95, and then pass a stable matched live route. A future actor switch must
preload its insertion model, command a measured-pose hold, and avoid the
previous model-loading drift. Reserved final scenes remain sealed; no RL was
run.

## Separate SFP identity and stop feasibility

The current SFP 480 px/160 px original-resolution RGB cascade was replayed
on two existing three-card development episodes, 100 sampled near-port frames
each. In the **card-0-absent, card-1/port-1-selected** episode, only 2/100
frames passed its geometric fit; **both** estimates were nearest a different
physical port, about 48 mm from the selected opening, despite three-view
reprojection residual below 1 px. The five-estimate temporal lock did not
latch within these samples. In the other episode, 71/100 estimates passed,
2 selected another physical port, and the first five-estimate lock at 13.9 s
was correctly associated with card 4/port 0 at 0.61 mm position error.
This replay covers sampled frames, not every possible later frame. A stable
wrong-port consensus remains possible, and the SFP transport path has no
validated observation-only stop. **Do not deploy an SFP perception-based
handoff from these results.**

## Reproduction and artifacts

- [SC same-trace replay code](../../scripts/compare_frozen_sc_transport_stops.py), [seven-trace JSON](../../outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.json), [plot](../../outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.png), [compact decision](../../outputs/experiments/2026-10-02_two_family_perception/sc_sfp_stop_decision_v1.json). The JSON retains exact selected times, target/route errors, false/missed labels, force/contact fields, stop traces, inference quantiles, source paths, model SHA-256 hashes, and the 0.95/two-decision head rule.
- [SFP lock replay code](../../scripts/replay_sfp_stop_feasibility.py), [identity and lock JSON](../../outputs/experiments/2026-10-02_two_family_perception/sfp_stop_identity_feasibility_100frames_v1.json), and [source 100-frame cascade predictions](../../outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_development_cascade_baseline_100frames.json).
- [Matched live runner](../../scripts/run_sc_stop_matched_20261002.sh), [transport-head failure video](/home/nas/chmin/aic_archives/2026-10-02_transport_stop_comparison/llb_0193_transport_head_v1/five_cameras_10fps.mp4), [visual-run video](/home/nas/chmin/aic_archives/2026-10-02_transport_stop_comparison/llb_0193_visual_v1/five_cameras_10fps.mp4), [wrong-route visual false-stop video](../../outputs/experiments/2026-10-01_transport_reliability/live_sc_visual_standoff_llb_0009_v7_fastresize/five_cameras_10fps.mp4). Each audit retains exact scene snapshots, manifests, model/code hashes, frame streams, controller traces, and official scores.
- Exact replay command: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/compare_frozen_sc_transport_stops.py --audit <each of the seven audit paths in the JSON> --handoff outputs/experiments/2026-10-01_transport_reliability/frozen_actor_handoff/best.pt --actor outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt --imagenet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth --output outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.json`.
