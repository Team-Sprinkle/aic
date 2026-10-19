# Explicit controller versus BC insertion actor: existing comparisons

**Yes, we have preliminary comparisons. They favor explicit control for SFP, but we have not completed a broad, strictly matched comparison across all settings.**

## Same SFP development starts

Two reused development repeats request SFP insertion into port0 of one NIC card:

| Insertion method | Passes physical insertion and toolkit completion |
|---|---:|
| Earlier full-connector BC actor with initial controller repair |1/2|
| Continued BC actor, controller-feedback channels masked |0/2|
| Continued BC actor, controller-feedback channels included |0/2|
| Explicit image-based controller |**2/2**|

The explicit result was also reproduced in another two-run controller version. These are repeated exposed development scenes, not independent broad coverage or evidence of a100%success rate.

The two explicit-controller insertions reached about46.76–46.77mm depth,0.31–0.35mm lateral error and0.34–0.37degree orientation error. The continued BC policies stopped11–16mm before the opening with roughly2.6–4.8mm lateral error. Thus the difference is actual movement/alignment, not merely toolkit score acknowledgement.

These use the same requested development cases and frozen perception components. However, the surrounding controller/reference handling also differs between versions. This is an engineering comparison of insertion pipelines, not an experiment isolating network-versus-equations as the sole changed variable.

## Complete original public sample

| Insertion pipeline | Score /300 | Physical + toolkit successes |
|---|---:|---:|
| Earlier learned BC actor |165.42|0/3|
| Explicit controller, shorter seating hold |152.61|1/3|
| Explicit controller, repaired seating hold |**176.85**|**2/3**|

All use the original three-trial public sample: two SFP trials and one SC trial. Each pipeline has only one complete run here. Later explicit-controller versions contain control/reference changes, so this is not a perfectly isolated actor-only ablation. Toolkit score can reward shallow/partial insertion, making the physical checks essential.

**SC does not yet have a successful broad comparison.** Both the learned actor and the initial explicit controller fail it. A later local-recovery explicit-controller pilot reaches15.43mm depth but still fails our unchanged depth/orientation gate. It uses extra recovery/calibration changes and must not be counted as a matched win over BC.

## Conclusion

- **SFP:** explicit image-based control is the stronger current engineering baseline.
- **SC:** neither approach has demonstrated reliable strict insertion.
- **Learned policy conclusion:** these results show the current BC actor is underperforming; they do not prove learned insertion is inherently worse.

A conclusive next comparison would freeze identical perception, starts, near-port approach, command limits, force guards, seating acknowledgement and recovery settings, then change only the insertion action-selection method. Start with no recovery in both arms and report geometry, harm, duration and insertion rate separately. That matched experiment has not yet been completed across diverse settings.

Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/{runs,context_live_current_pose_v1,visual_servo_current_pose_v2,visual_servo_current_pose_v3}/` and `/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1/score_comparison.json`.
