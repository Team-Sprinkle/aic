# Native TCP-delta correction for the strict 60/14 pilot

Status: corrected training and frozen paired final complete on 2026-09-18.
The user identified a mismatch:
the expert label is a six-value TCP-frame relative command, while the first
ACT60 and world-model pilots trained policies to predict absolute base-link
targets. Preserve those results as historical absolute-target baselines.

## Verified action contract

For each of the 32,183 native observations in the same 60 train / 14 validation
episodes, the source record stores the observed TCP pose, a relative command,
and the teacher target. Composing the observed pose with the recorded command
reconstructs the teacher target with maximum errors **3.32e-8 m** and
**3.33e-8 rad**. The model should output the relative command. At execution,
the runtime may compose it with the observed TCP pose to supply the physical
target required by the simulator API. It must not accumulate against a later
controller target. See `outputs/experiments/2026-09-18_dreamer60_pilot/act60_delta_contract_audit.json`.

The old ACT cache causally held labels across gaps in native recording times,
expanding 32,183 observations to 41,575 rows. That representation is unsafe
for repeated relative commands, so the brief first delta restart was stopped
after roughly 150 updates. Its output is retained under
`/var/tmp/chmin_aic_20260918_dreamer60/act60_tcp_delta_v1/` and is not a
selection candidate. The replacement cache copies only rows whose timestamps
exactly match native observations, with no duplicates, and audits each selected
state and command against the source `frames.jsonl`. Its ready record is
`/var/tmp/chmin_aic_20260918_dreamer60/act_cache_native_delta_v1/READY.json`.

## ACT60 restart

Fresh ImageNet ResNet18 ACT training ran from 18:20 to 18:55 UTC on physical GPU 4.
The 60/14 episode split, task input, image size, optimizer rates, batch 128,
terminal sampling, state masking, and 6,000-update cap match the prior pilot.
Action representation is now `delta_pose`, command frame `gripper/tcp`, and
the runtime delta reference is the current camera observation. Chunk length
and execution horizon are **one**: native inter-observation gaps vary from
50 ms upward, so longer fixed-rate delta chunks would require unknown future
labels. The runner and selection rule are
`outputs/experiments/2026-09-18_dreamer60_pilot/run_paired_act60_delta.py`
and `act60_delta_selection_rules.json`. The completed and failed absolute-pose
results remain in their original folders. The corrected run completed all
**6,000** requested updates, 768,000 sampled training examples, without a
deadline or signal stop. The fixed held-out rule selected update 6,000:
first-command **2.96 mm** and final-three-second **2.63 mm** TCP-delta
translation error, summed **5.59 mm**. These are command imitation errors,
not insertion scores. The selected checkpoint, normalizer, CUDA TorchScript,
metadata, validation, and contract audit are archived in
`act60_tcp_delta_selected_archive/`; the runnable selection and freeze records
are `act60_tcp_delta_selected.json` and
`act60_tcp_delta_runtime_freeze.json`. Four paired development scenes ran from
18:56 to 19:09 UTC, each with a fresh simulator, initial-state audit, full
90-second scoring, and 1 fps video. All four were valid; **0/4 full insertions**,
mean official score **27.73**. Individual scores were 36.69, 36.35, 1.00,
and 36.90; final plug-to-port distances were 5, 5, 33, and 5 cm. The
third scene's start/end frames show the target leaving camera view as the
policy drifts. Results, score files, audits, and videos are archived in
`act60_tcp_delta_development_archive/`. These development scenes did not
change checkpoint selection. The earlier absolute-target ACT60 on these
same four scenes scored 34.65, 31.33, 31.94, and 25.28 (mean 30.80), also
0/4 insertions. With four trials, the corrected delta policy has not shown
a reliable live improvement despite lower offline command error. An early
update-1,000 runtime diagnostic
did complete on one declared development scene: valid fresh start and
90-second official scoring, **0/1 insertion**, total **12.60**, final
plug-to-port distance reported as 2 cm. Its 1 fps video, contact sheet,
official score, and start audit are in `act60_delta_early_runtime/`. This early
checkpoint was not used for final selection.

## World-model restart

The visual tokenizer did not consume actions and may be reused after its
weight/input audit; its fine-contact reconstruction limitation remains.
Action-conditioned dynamics must use deltas derived from each recorded
observed TCP pose and executed target. Raw relative commands are teacher BC
labels. Missing 20 Hz observations do not become repeated relative commands.
The prior action-conditioned world and BC runs are superseded. Corrected
short-transition dynamics trained from 18:28 to 18:45 UTC on GPUs 2 and 3,
then stopped after 3,191 updates because the fixed held-out prediction gates
failed. The selected update-1,000 model's one-, two-, and three-step TCP
translation errors were 10.33, 14.80, and 19.56 mm, versus persistence at
1.85, 4.47, and 9.03 mm. These checks used 14, 9, and 5 eligible episodes,
respectively. A fresh supervised policy with exact future-label masks ran from
18:46 to 19:20 UTC on GPUs 2 and 3. It stopped at update 4,509 under a
plateau rule recorded during the run; fixed held-out selection chose saved
update 4,000: **1.82 mm** ordinary and **2.51 mm** terminal first-command
translation error, **2.16 mm** combined. This is supervised command imitation,
not a successful dynamics or insertion result. A trained one-GPU 1,000-call
inference benchmark measured **28.22 ms p95** on already-sized images. The
first live startup exposed a camera-size mismatch before any policy command;
the runtime now matches the recorder's `cv2.INTER_AREA` resize to 288×256.
With actual raw camera dimensions and resize included, trained inference
measured **32.76 ms p95**, and the full four-command callback loop measured
**33.79 ms p95** in isolation. In a live simulator diagnostic on the same
GPU as Gazebo, 168 published commands had **464.67 ms p95** observation-to-
publication latency, above the 300 ms requirement. Phase logs put most delay
inside the model call under live contention. The failed startup and this
latency diagnostic are retained as ineligible. Separating rendering on GPU 2
from policy inference on GPU 3 did not resolve the ROS-process delay: its
one-scene diagnostic still measured 464.85 ms p95. A private process worker
then isolated inference from the ROS callback process. Four verified native
observations produced byte-identical 4×6 commands with the same executed
history; reset, timeout, error propagation, and cleanup checks passed. A
fresh full-scene diagnostic passed at **76.12 ms p95** live latency. The
subsequent four-scene development set was **4/4 valid, 1/4 full insertions**,
mean official score **52.14**, with individual scores 49.49, 36.25, 86.46,
and 36.36. Its pooled 1,782 decision latencies measured **81.00 ms p95**,
95.48 ms p99, maximum 133.11 ms, and none exceeded 200 ms. Scene 3 received
full Tier 3 insertion credit; its force log recorded a brief 28.01 N peak
for 0.02 seconds, below the penalty duration threshold. The model checkpoint
and worker runtime were frozen before the paired 20-scene final evaluations
started at 20:11 UTC. Insertion reward remains unavailable for training:
all 74 expert success events occur after the last saved observation.

Both corrected policies used the same already declared development and paired
final scenes. Each final set completed 20/20 eligible 90-second scenes:
ACT **0/20 full insertions**, one official partial, mean score **22.69**;
world **0/20 full insertions**, two official partials, mean score **32.32**.
World observation-processing-to-publication p95 was **77.01 ms** across
8,908 actual decisions. The paired result and hash-pinned media are in
`paired_tcp_delta_final_summary/`, `act60_tcp_delta_final_archive/`, and
`paired_world_archives/world_tcp_delta_final_worker_v1/`. ACT scene 17's
first attempt failed at model lifecycle activation before any scored rollout;
the original attempt was preserved and the same scene was retried unchanged,
then ran the full 90 simulated seconds. No final scene was used to choose a
checkpoint. The post-run [follow-up](2026-09-18-world-followup.md) audits
actual port-opening geometry, dataset provenance and visual fidelity.
Their control horizons differ:
ACT predicts one delta and replans at each fresh observation, while the world
policy predicts four 20 Hz microcommands per visual decision. A live score
difference would therefore compare complete policies, not isolate the effect
of world-model pretraining.
