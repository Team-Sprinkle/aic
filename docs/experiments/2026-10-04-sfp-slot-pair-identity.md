# SFP slot and paired-port identity repair

## Frozen question and limits

The 2026-10-04 fresh ten-scene shadow test found 46 accepted wrong-port predictions among four frozen heads. The requested opening was inside an observation-selected crop in 335/346 triplets, so this attempt tests whether full-image rail-slot context and joint port-pair supervision can resolve identity. The recent ten scenes are development evidence, not independent final validation. The four reserved final IK scenes remain sealed.

The failure list is frozen at `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/frozen_failures/`. Training labels and exact image paths are frozen at `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/labels/`. The fit split has 59 complete reset groups; ten other complete reset groups are used for checkpoint selection. One synthetic scene close to a validation scene was excluded. Simulator geometry supplies training and posthoc scoring labels only. Autonomous acceptance uses RGB, task code, camera calibration, predicted rail slots, predicted paired ports, and fixed CAD spacing.

## Predeclared attempt

Run one model on GPU 1 for at most 2,000 updates or 120 minutes, whichever occurs first. Initialize ResNet18 from the previous structured SFP checkpoint; train its later layers and a new head that predicts all five numbered rail slots and both ports on each card from original RGB. Losses cover projected points, in-frame flags, and each port pair's relative vector. Select a checkpoint on ten old development reset groups. Do not tune thresholds using new scenes.

Frozen acceptance rule: all three cameras must vote in-frame for both ports of the requested pair; triangulated reprojection residual must be ≤5 px; predicted rail spacing 30–50 mm, rail order residual ≤10 mm; port-pair spacing 15–35 mm with correct orientation; selected port must lie ≤20 mm along the rail axis from its selected slot. These are fixed geometry consistency tests on **predictions**, with no true port transform in runtime logic.

An independently generated test of at most 20 nonfinal complete resets follows only after model and rule freeze. Promotion requires zero accepted wrong ports, at least 18/20 anchors by 3 s with every request code represented, and accepted coarse 3D p95 ≤20 mm. Report abstentions. Failure on old development data with coherent neighbor swaps ends this model attempt and triggers one bounded observation-only view/history diagnostic. No transport, PoseInsert, or RL promotion follows a failed identity gate.

## Commands and results

The first smoke fit was stopped before its first 400-update checkpoint because the dataset loader inadvertently supervised the nominal coordinates of *absent* card slots as image detections. Those coordinates are useful for physical indexing but have no visible object in RGB. The partial log and empty `fit_v1/` are preserved. `fit_v2` masks absent slots and retains the same architecture, split, budget, and acceptance rule. This is a data-label correction, not a result-driven threshold change.

`fit_v2` was stopped after two minutes without a single training step: worker processes forked after CUDA initialization and waited indefinitely. `fit_v3` uses a main-process loader for this small dataset and prints a progress record every 100 updates. The second stopped log is preserved too.

Command: `bash scripts/run_sfp_slot_pair_identity.sh`. Append metrics and decisions after the bounded fit and evaluation.

## Fit result and root cause

`fit_v3` finished the complete 2,000-update budget in about 451 s on GPU 1. Its loss fell from 0.155 at update 100 to 0.044 at update 2,000, but the frozen consistency rule accepted only **one of 136** old development observations at the best (update 400) checkpoint, with **zero of ten** episodes anchored by 3 s. Later checkpoints accepted none. Thus lower training loss did not deliver usable identity. The one accepted prediction was correct, 2.26 mm from the selected port; that single point is not evidence of generalization. The best checkpoint and every 400-update prediction record are retained in `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/fit_v3/`.

Matched on exactly the same 136 observations and ten reset hashes, the frozen selected-only head accepted **51/136**, gave **4/10** early first anchors and zero wrong accepted ports. The new paired-port model is clearly worse on coverage; it did not earn an independent new-scene test.

The dedicated read-only diagnosis of the best checkpoint found that 127/136 observations had high in-frame scores for the requested slot and both ports in all three cameras. Yet 125/136 failed the pair-detection/triangulation gate. On image labels used **only after inference**, median point error was 8.7 px for the slot, 9.0 px for port 0 and 8.5 px for port 1; their p95 errors were 95, 127 and 117 px respectively. The long error tail causes cross-camera geometry to disagree. This is mainly a localization and correspondence failure, not a network that simply labels every target absent. The original labels mark *projected in-frame*, which does not certify unobstructed visibility; absent card slots were masked in the corrected fit. Diagnosis records: `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/fit_diagnosis_v1/`.

Diagnosis entry point (run inside the same pinned rootless image as `run_sfp_slot_pair_identity.sh`, with `PYTHONPATH=scripts:aic_utils/lerobot_robot_aic`):

```bash
.pixi/envs/default/bin/python scripts/diagnose_sfp_slot_pair_fit.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/labels \
  --init /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --checkpoint /home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/fit_v3/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/fit_diagnosis_v1
```

## Bounded sparse-history diagnostic

Because the fit failed on existing development scenes, **no new Gazebo resets were opened**. A read-only diagnostic asked whether a sparse visual history and measured tool movement can reject persistent wrong identities on the ten frozen fresh scenes. For each of the four earlier frozen heads, it required two or three consecutive accepted 3D predictions to agree within 5 mm, optionally after the measured TCP had moved at least 10 mm. The true port was consulted only to score each resulting decision. Command:

```bash
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
  .pixi/envs/default/bin/python scripts/diagnose_sfp_sparse_identity_history.py \
  --source /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/sparse_history_diagnostic_v1
```

For the structured head, one frame gave 5/10 early first anchors with one wrong; two agreeing frames gave 3/10 early and still one wrong. Requiring 10 mm of measured movement removed that wrong first anchor, but only 3/10 acquired any anchor, also 3/10 early. The all-port head retained a wrong first anchor even with two or three frames and the movement rule. The selected-only head stayed safe but at 3/10 early. Therefore simple temporal agreement or a modest natural view change does not solve wrong-port identity. This diagnostic uses teacher motion and is **not** an autonomous active-view test. Full rules, first decisions and counts are in `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/sparse_history_diagnostic_v1/`.

## Decision

Stop this slot-pair architecture; do not change its thresholds and claim success. The 20-scene independent test, complete live latency, transport handoff, PoseInsert and RL gates were **not entered**. Final scenes remain sealed. The next bounded question is whether an explicit *safe side-on camera viewpoint* makes the requested card/port identity visually separable. Use no training initially: pick a few nonfinal reset configurations, hold a collision-checked standoff, record original RGB before and after a small measured lateral view move, and score whether a human and the existing predictors can distinguish neighboring card/port openings. Predeclare the motion envelope and stop if identity is still ambiguous or the move is unsafe. Only if that view demonstrably exposes stable identity should another model or controller change be attempted.
