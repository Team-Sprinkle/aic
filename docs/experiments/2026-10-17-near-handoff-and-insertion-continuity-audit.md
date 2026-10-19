# Closer handoff and insertion command continuity

## Revised responsibility

The user reviewed the recordings and requests transport to deliver the plug to **less than30mm before the opening**, rather than letting the learned actor own the entire150mm approach. The earlier150mm stopping point was intentional under the previous plan; it was not a false distance report. This revision transfers the intermediate approach to an explicit vision/proprioception controller.

Start with **25mm nominal**, leaving room for estimation/tracking error while remaining outside the port. Gate actual posthoc checkpoint geometry separately:20–30mm stand-off, lateral<=2mm, explicit orientation and force/contact reporting. Do not infer full insertion from reaching that checkpoint. Confirm on development scenes before a larger sweep; do not tune on released official settings or open reserved final scenes.

The old40/62-scored full rollout batch remains preserved, stopped on its independent terminal geometry mismatch. No checkpoint, policy or result in that batch is overwritten.

## What the actor actually is

The evaluated shared insertion model is a **deterministic ResNet18 + two-layer256D transformer**, approximately13.73Mparameters. It is not the earlier diffusion transport actor. It conditions on current three-camera RGB,32Drobot state,10Dpublic task code, camera-estimated selected-port pose, estimated plug-tip translation and four causal robot/relative-pose history samples at1.2/0.6/0.2/0seconds. It has no camera-image history. The action output is four complete port-frame control-point targets, retaining TCP orientation; the runtime converts these into TCP targets and body-frame deltas for the fixed-frame API. There is no residual action adapter or random action sampling in these BC tests.

## Decisive execution audit

Across37actor-executing scenario episodes:

- The model predicts **four50ms targets**, spanning200ms from the observation.
- The insertion runtime executes **all four**, not just the first. SC issued indices0/1/2/3 on1899/1899/1896/1894commands; SFP8612/8611/8608/8603. Small end-of-episode differences are expected.
- **Zero insertion-stage stale-member drops**. Earlier drop events belong to transport; counting the entire trace would misattribute them to insertion.
- Within-chunk command spacing is approximately50ms. Between chunks the median gap is100ms and p95150ms; inference produces extra hold time. Full timing is not an ideal uninterrupted20Hz action stream even though each action uses50ms training offsets.

The short horizon is a plausible contributor, but truncation/dropped insertion steps is **not** the established failure.

### Actual target discontinuities

These are changes in commanded absolute TCP pose between consecutive commands, not measured robot jumps:

| Task | Within-chunk translation p95 | Replan-boundary translation p95 | Within-chunk rotation p95 | Replan-boundary rotation p95 |
| --- | --- | --- | --- | --- |
| SC |4.52mm|**32.15mm**|1.94degrees|**17.21degrees**|
| SFP |1.29mm|**12.31mm**|0.126degrees|**1.96degrees**|

Boundary maxima are61.30mm/25.26degreesSC and35.67mm/5.46degreesSFP. Therefore the visible jerks are strongly associated with **successive plans disagreeing**, rather than merely too few commands from each plan. These measurements do not alone determine why the model replans badly.

## Training coverage: substantial, with important limitations

The corpus has67,039rows/500source episodes, with421training episodes and79validation episodes. Training uses102SC and319SFP episodes. All task codes are represented: SCport0/1 have44/58training episodes; each SFPcard/port request has19–55. This is not a two-episode BC fit or an entirely missing SC task code.

| Training phase | SC rows / episodes | SFP rows / episodes |
| --- | --- | --- |
|90–185mm before opening|4,136 /95|12,182 /309|
|30–90mm|1,415 /102|6,169 /319|
|0–30mm|6,626 /102|6,559 /319|
|Inside opening plane|2,738 /102|15,067 /319|

Episode-normalized, phase-weighted sampling gives the90–185mm phase only about**12%SC /14%SFP** of its family's training mass. Fine-control emphasis was intentional but does not prove sufficient approach learning. At runtime, an initial pose has a median24SC/26SFPtraining episodes with the same task code within10mm estimated tip position and5degreesTCPorientation. This is limited geometric coverage evidence, not proof of visual/layout or dynamic-load coverage.

**55% of SC0–30mm training chunks are nearly stationary**, defined as<0.05mm translation between the first and fourth targets. Some are legitimate contact settling; do not delete them blindly. SFP far-approach chunks are mostly moving, so pauses alone cannot explain SFP's lack of progress.

Both models finished30,000updates; the subsequent late-validation check found no practical improvement justifying more of the same fit. Nevertheless selected held-out first-target p95 errors remain roughly9–12mm over the broad task, far larger than the insertion corridor. A plateau is not evidence that the policy is accurate enough. The loss supervises individual chunks and their internal motion; it lacks an explicit continuity objective between predictions made from successive observations.

## Bounded25mm test now running

`RunSample25mmThenInsertion` is a separate diagnostic policy. It reuses the frozen sample-derived route and requested-port perception, locks port/body offset at150mm, and uses that static-fixture estimate plus measured TCP to approach25mm. It aligns before axial motion, slows for the last50mm, bounds integral tracking compensation, and aborts on force/no-progress/budget violations. Simulator geometry is posthoc assessment only. It does not require the near-port estimator to re-estimate a potentially occluded grasp at25mm.

The first SC/fresh pilot's handoff is independently confirmed from the raw bag at **24.20mm before the opening /0.547mm lateral /3.684degrees plug orientation error**, with no scored harm. The unchanged actor then ends at0.513mm axial depth /5.955mm lateral /4.395degrees. Official tier3is75but the physical full-insertion gate fails. This demonstrates one closer approach, not reliable insertion. Remaining matched pilots continue.

Four fixed development rollouts compare fresh/visual-warm actors in SC/SFP. **Recovery is off in this isolated diagnostic**. The old hierarchical results remain unchanged. Deployable near-range hierarchy must preserve the150mm measured path/root and reacquire there after a large retreat; the old150mm lock-envelope check must not be silently reused at25mm. The diagnostic rejects recovery modes other than `none` until that integration is separately verified.

Command:

```bash
.pixi/envs/default/bin/python -u scripts/run_25mm_handoff_diagnostic.py \
  --output /home/nas/chmin/aic_archives/2026-10-17_25mm_handoff_diagnostic_v1
```

GPU5 only, rootless Docker, private domain98 and cooperative GPU lock. Approximately20–45minutes for four bounded tasks plus video processing. No active job interrupted, no new training, no official tuning.

## Next decisions

1. If the visual approach cannot reach the25mm gate, diagnose its measured tracking, visibility and force trace before changing the insertion model. Do not relax gates to disguise contact.
2. If25mm succeeds but both unchanged actors still fail, the long150mm approach is not the only actor problem. Compare exact live inputs/decoded targets with teacher labels, including cold-start history and compliant target compensation, before training.
3. Build a **copy** of recorded-target labels at50ms for longer horizons, preserving successful experts and complete-episode splits. Do not stretch four existing targets into eight or substitute future measured poses. Audit true contact holds separately from idle pauses.
4. Run matched near-range training: corrected/phase-balanced4-predict/4-execute baseline,8-predict/4-execute,12-predict/6-execute. These span200/400/600ms predictions and200/200/300ms executed segments. Add cross-observation overlap consistency only where timestamps/frames genuinely match. Retain all baselines and require plateau evidence plus autonomous insertion and command-smoothness improvement.
5. Repair measured-path retreat progress under compliant tracking and test low-force no-progress detection separately. Only then reconnect hierarchical recovery, including fresh reacquisition after large retreats. Do not rely on recovery to conceal an unstable actor.

## Artifacts

- Audit script/results: `/home/nas/chmin/aic_archives/2026-10-17_near_handoff_actor_audit_v1/`: `audit.py`, `audit.json`, `task_coverage.json`, `command_discontinuity.json`.
- Near-range pilots: `/home/nas/chmin/aic_archives/2026-10-17_25mm_handoff_diagnostic_v1/`: frozen plan/sources, commands, raw bags, causal terminal and handoff geometry, metrics and20fpsfive-view videos.
- Prior diagnosis: [Videos and insertion/recovery analysis](../answers/2026-10-17-evaluation-videos-and-insertion-failure-analysis.md).

## Additional orientation-input limitation

The actor receives estimated **port orientation** and measured **TCP orientation**, but the gripper-to-plug estimate is translation only. Both grasp helpers compute a diagnostic rotation; neither has an approved orientation output for this pipeline, and the actor masks tip-orientation inputs. It must infer actual plug attitude from images/nominal grasp instead of being explicitly conditioned on it. Startup grasp rotation can therefore matter even with an accurate port estimate. This can limit fine alignment but does not by itself explain the much larger20–50degreeSCdrift or SFP failing far from the opening. Audit existing diagnostic grasp rotations against held-out labels before enabling them; do not pass true grasp orientation to deployment.

Median replan-boundary command gaps are100ms(p95150ms), compared with50ms within chunks. Median inference is80.9msSC/100.3msSFP. The complete-chunk runtime avoids dropped members by starting execution after inference, but does not pause simulation while inference runs; the previous target is held longer. Preserve this timing limitation in any matched horizon comparison.

## Runtime frame parity and history follow-up

A saved-trace numerical audit checks42,080predicted target conversions. Reconstructing base-to-port from measured TCP and its logged relative pose differs from the locked visual frame by at most8.95e-6mm/2.92e-6degrees. Plug-point-to-TCP decoding differs from the expected rigid transform by at most7.46e-6mm. This rules out a large numerical frame-conversion error in those operations, not every possible training label or observation-time mismatch. Results: `runtime_frame_parity.json`.

A concrete follow-up input-distribution concern is cold-start history: visual acquisition clears robot history, and `RunLockedPortLocal.insert_cable` dispatches another `reset_for_task` that clears it again at actor start. Most training rows have preceding teacher-motion history. Empty history is causally padded with the earliest available state; it is not future data, but it may differ from teacher context. Test a separate actor wrapper that seeds only recent measured observation history from the approach (no old commands), and compare identical frozen inputs before changing training. Do not silently restore action chunks or use post-reset history. This is a hypothesis, not yet a measured explanation of failure.
