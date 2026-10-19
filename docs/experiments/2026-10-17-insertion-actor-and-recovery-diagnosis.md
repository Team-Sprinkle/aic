# Evaluation videos and why insertion fails

## Videos

These are full Gazebo evaluation rollouts with two diagnostic overview cameras and the actor's three wrist cameras, H.264/20fps. The overview views are recorded for inspection; they are not policy inputs.

- [SC: approaches but loses alignment](/home/nas/chmin/aic_archives/2026-10-17_insertion_failure_diagnosis_v1/SC_alignment_drift.mp4). Insertion takes over around simulation39.65s.
- [SC: attempted retreat aborts](/home/nas/chmin/aic_archives/2026-10-17_insertion_failure_diagnosis_v1/SC_retreat_aborted.mp4). Insertion takes over around41.04s; retreat aborts51.05s.
- [SFP: oscillates far from opening](/home/nas/chmin/aic_archives/2026-10-17_insertion_failure_diagnosis_v1/SFP_stalled_approach.mp4). Insertion takes over17.35s.
- [SFP: limited approach progress](/home/nas/chmin/aic_archives/2026-10-17_insertion_failure_diagnosis_v1/SFP_limited_approach.mp4). Insertion takes over19.90s.

Prediction-free raw-view contact sheets were generated from each recording and visually inspected. They show the SC wrist/plug changing attitude and approaching the board, whereas SFP stays at a large stand-off with modest oscillations. Images alone cannot identify the exact contacting parts; contact penalties and geometric distances below come from independent posthoc scoring.

## Is transport or insertion failing?

For these selected episodes, transport finds the requested target and reaches the approximately150mm holding position. Fresh vision then locks the port and hands control to the trained insertion actor. The failures develop afterward.

Across the37actor-executing scenario episodes, opening-position error p95 is **0.146mmSC /0.251mmSFP**, and opening-orientation error p95 is **0.272degreesSC /0.539degreesSFP**. The terminal gripper-to-tip offset error p95 is0.991mmSC/0.116mmSFP. Thus a grossly misplaced visual target is not supported as the main explanation, although these checks do not certify every runtime transform or remove smaller grasp errors.

Transport is not universally solved: only37/40scored scenarios reached actor execution. The separate transport-only batch achieved121/130, including2/6exposed official cases. My earlier wording that transport was working fine was too broad.

### SC: approaches while alignment deteriorates

In the one-card port0video, the estimated initial plug position is154.24mm before the opening with1.49mm lateral offset. Its first four predicted target depths are151.24/150.42/149.59/148.73mm before the opening, so it initially requests forward progress. Within about1s, however, its next chunk targets171–172mm before the opening and requests a15.1degree orientation change from its current attitude. It later approaches while misaligned. Independent terminal measurements show **21.21mm lateral error /22.13degrees orientation error**, and scored harm. The no-NIC-card SC example ends26.79mm lateral/31.20degrees. A positive axial coordinate merely means crossing the opening plane; it does not mean entering the port.

This locates the failure in the insertion/control stack. It does not yet establish whether the underlying cause is BC state coverage, action decoding/history/timing, controller interaction or a combination. Do not label this a proven architecture limitation or restart training blindly.

### SFP: no sustained approach

In the stalled SFP example the plug starts146.49mm before the opening and ends143.58mm before it, after about108s of actor execution. Sampled chunks repeatedly aim behind the current estimated plug position, and the robot oscillates rather than advancing to insertion. The second example starts145.18mm away and ends122.90mm away. These are not small final alignment errors; most of the150mm approach remains undone.

Recorded commands are compliant-controller equilibrium targets. A target behind the measured plug is not, by itself, proof of a wrong label: load compensation can offset equilibrium and measured position. Compare with the recorded successful teacher at matched states/controller loads before declaring the action targets incorrect.

## Is hierarchical backtracking enabled?

**Yes.** The evaluated arm is `visual_warm__hierarchical`, using the shared BC actor plus a rule supervisor. No new RL was trained.

It triggers on **both** continued motion demand with little measured movement **and** elevated baseline-corrected force: at least8N, little translation (<=0.15mm per check), for two checks. It does not trigger merely because the actor is making no useful axial progress. The stalled SFP example's load change peaks at only3.14N, so it never triggers. Nor will every sideways sliding/misaligned motion satisfy the stall condition.

A full trace review corrects the earlier preliminary count: **10actor-executing episodes enter backtrack**, not just the eight with explicit recovery-abort events. None completes a retreat/lateral retry:

- Six end with `retreat_timeout`.
- Two end with `force_rising_during_retreat`.
- Two run out of the whole-task budget during backtracking.

There is therefore no evidence yet that hierarchical backtracking improves this evaluated pipeline.

## Concrete recovery-execution gap

The retreat implementation tries to reach each measured waypoint within1.5mm/2degrees, using a commanded target at most3mm from the current TCP position. In the six timeout cases, final TCP tracking error to the last commanded retreat target is **2.18–3.00mm** (orientation error0.002–0.117degrees). Hundreds of commands do not complete the path. In one SFP example,416commands repeat an exactly unchanged absolute target; in one SC example395commands shift the target only0.318mm from first to last. The last20SCtargets differ by at most0.0023mm.

This supports a controller/waypoint-progress problem: repeatedly requesting a tiny pose offset can settle under compliant load without reaching the waypoint tolerance. It is not proof of cable snag and does not justify raising force limits. Exact contact causes still need independent inspection.

## Targeted next checks

1. Repair the existing terminal-geometry audit mismatch and preserve the frozen evaluation before reporting the remaining scores.
2. Audit first-seconds insertion actions at matched teacher/live states, including all four predicted commands, frame round trips, history and compliant-load compensation. Compare the fresh and visual-warm models; do not infer generalization from teacher-state validation alone.
3. Test a separately versioned retreat controller with measured-path lookahead and bounded tracking-bias compensation, retaining force aborts and genuine checkpoint arrival. The earlier transport retreat repair is a reference, not evidence that this insertion retreat already works. Do not change the frozen evaluation midstream.
4. Separately test an observation-only no-progress recovery trigger so low-force SFP stalls can be detected without treating harmless pauses as blockage.

Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_insertion_failure_diagnosis_v1/diagnosis.json`, `analyze.py`, four `*_review.jpg`contact sheets and four linked videos. No model, policy, dataset or evaluation threshold changed for this analysis. Reserved final scenes remain sealed.
