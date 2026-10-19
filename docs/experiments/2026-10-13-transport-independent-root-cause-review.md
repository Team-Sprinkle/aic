# Transport review: endpoint contract, lost terminal labels, and model priorities

**October 14 plan update:** Use the [consolidated transport and handoff plan](../../obsolete/docs/2026-10-19-retired-plans/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md) for subsequent execution. It retains these findings and adds an early bounded rule-based SC transport branch before broad retraining.

Date: 2026-10-13. Analysis only: existing checkpoints, code, recorded Gazebo traces, images, and primary research. No training, collection, simulation, or GPU inference was launched. Active processes were inspected and left untouched. Final scenes remain sealed. The reserved evaluation results and existing artifacts are unchanged.

## Main correction to the previous diagnosis

The recent SC actors have a real late-route/termination problem, but the previous evaluation overstated failure to learn the route. It compared **transport-only training labels** with a closer approach band reached by a **different teacher stage**. It also called same-clock teacher/actor separation an early route departure. These are different measurements.

This review supersedes the recommendation to immediately collect SC corrections around 6–9 seconds. First repair the transport/approach contract, terminal supervision, and stopping test. Do not promote a checkpoint merely by changing the metric.

## 1. Training endpoint versus evaluation endpoint

I streamed all 69,681 SC rows in the active union's `sc_labels.jsonl`, covering **83 training / 21 validation episodes**. Every row is labeled `transport`. None of these 104 transport prefixes enters the old TCP axial band of −100 to −70 mm from the selected opening.

The last retained transport labels have:

- TCP axial coordinate: median **−167.97 mm**, range **−175.79 to −162.72 mm**.
- Measured plug-tip axial coordinate: median **−148.10 mm**, range **−156.22 to −141.23 mm**.

These are signed distances along the opening's axis, not insertion depth. The difference between TCP and tip is physical geometry. They must not be interchanged.

`materialize_moveit_bc_segments.py` deliberately separates transport at the explicit `local_preinsert_align_started` event. For the examined `llb_0009` teacher, transport ends at simulation time 45.6 s. Later local alignment and `cheatcode_handoff_started` at 50.966 s perform additional approach motion, followed by insertion. Full source episodes succeeded; **the actor was not trained on the complete source episode**.

Thus perfect imitation of the admitted transport prefix does not imply entry into the old approach band. The missing approach must either be owned by transport through an expanded, verified prefix, or by a downstream actor whose validated starting range includes this endpoint. We have not established the latter at a plug distance near 150 mm.

Evidence: [all-episode dataset contract](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/dataset_contract.json>), [teacher endpoint states](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/teacher_endpoints.json>), [tip/TCP audit](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/endpoint_review.json>).

## 2. Reassessing existing actors against the taught endpoint

I rescored saved proposals against each episode's **last admitted teacher transport state**, without running a policy. The following diagnostic asks whether the robot ever came within 20 mm and 5° of that state. The thresholds were chosen posthoc for diagnosis; this is neither a preregistered gate nor a deployed stopping rule.

| Frozen actor | Existing development episodes | Median closest endpoint error | Episodes visiting 20 mm / 5° |
| --- | ---: | ---: | ---: |
| 100 ms, dense70 history | 6 | 22.65 mm | 3/6 |
| 100 ms, fine70 history | Same 6 | **13.15 mm** | **5/6** |
| Intentional fit, stage 2 | Same 6 | 15.63 mm | 4/6 |
| Intentional fit, final plateau | 3-scene subset | 13.65 mm | 3/3 |
| Corrected 900 ms cadence | Same 3-scene subset | 11.50 mm | 3/3 |

The fine70 actor is therefore worth retaining: it improved this transport-endpoint metric on five of six matched scenes. That differs from the previous ranking by closest distance to the port opening. It is a warm-start continuation, so it does not isolate extra history from extra optimization. The three-scene arms cannot be ranked against six-scene arms as if they covered identical distributions.

The corrected cadence actor approached the three endpoints within **9.28 / 11.50 / 12.81 mm**, with **2.57 / 2.24 / 4.31°** orientation error at those visits. It subsequently drifted to **201 / 129 / 144 mm** endpoint error. Other actors likewise often passed near the endpoint and then left. This is evidence of an arrival/termination gap, not proof that stopping there would yield insertion.

The diffusion branch in `RunTemporalTransport.py` explicitly sets both `last_handoff_probability` and `last_stage5_probability` to zero. `RouteDiffusion` has a future-motion head but **no trained stop head**. It does not inherit the older transformer's handoff prediction. Those runs were intentionally transport-only diagnostics. Their zero insertion counts cannot compare complete insertion systems.

Evidence: [all rescored traces and summaries](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/endpoint_comparison.json>).

## 3. Early time lag is not necessarily an incorrect route

The previous scorer finds teacher TCP at the actor's absolute simulation timestamp. Teacher first commands occurred at 3.8–3.86 s; the actor first proposals at 2.7–3.25 s. Even aligning starts cannot remove speed differences or teacher pauses.

For the corrected cadence actor, I additionally computed distance to the nearest **measured teacher transport path point**, independent of time. This is an optimistic geometric lower bound: nearest points may jump between route stages and do not enforce monotone progress or orientation. It cannot certify a safe route.

| Scene | Previous same-clock >20 mm onset | Nearest-path >20 mm onset, sustained three proposals |
| --- | ---: | ---: |
| `llb_0005` | 6.20 s | 39.70 s |
| `llb_0074` | 6.50 s | 14.00 s |
| `llb_0009` | 6.25 s | 26.95 s |

The plot shows the actor follows much of the bypass topology before late drift. Five-card offsets still matter. Saved five-view video samples at 50 and 60 s agree with the late motion change; they do not establish every possible cable contact. Official scores say no scored contact, which is not proof of zero cable force.

[Path plots](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/path_alignment.png>) · [per-proposal path and orientation data](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/path_alignment.json>) · [50 s frame](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/actor_llb0009_video_50s.jpg>) · [60 s frame](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/actor_llb0009_video_60s.jpg>).

## 4. Long-horizon filtering removes important late supervision

The 100 ms cache contains 4,356 SC training rows, but requiring all 12 target slots leaves **2,740**. Among cached rows carrying nominal final-stage index 5 (`return_toward_port`), **388 become 104**. **29/83 training episodes have no surviving stage-5 row**. Validation similarly has 7/21 episodes without valid stage-5 labels. Stage names here are inherited plan-time annotations and need checking against executed events; they are not independent measurements of route completion.

The last valid target anchor precedes the last admitted transport observation by a median **1.97 s**, p95 **12.23 s**, maximum **24.66 s** over all 104 episodes. At least the final 1.1 s is lost by construction when requiring 12 slots at 100 ms with no terminal mask. Some episodes lose much more. All **61 cached handoff-positive rows** are outside this fully valid action subset. They cannot train a stop head if its dataset is restricted to that subset.

Two code mechanisms explain this selection loss:

1. `build_extended_route_chunks.py` walks successive **cleaned rows** and requires approximately 50 ms increments. Removing quiet rows therefore removes potential future labels and history observations, not merely their action-loss weight. A command can remain valid while its loss anchor has been downsampled.
2. It requires SC `command_indices` to be exactly consecutive. However, `materialize_moveit_bc_segments.py` correctly selects the most recent command at each requested timestamp. A slightly delayed command produces repeated indices representing a valid held target. Example: `llb_0010`, t=38.91 s has indices `[694,694,695,696]` and source offsets `[0,0,.054,.110]` for slots `[0,.05,.10,.15]`. The extended builder rejects the anchor. This is not evidence of corrupt teacher motion.

The conservative filter prevented uncertain interpolation, but it is too restrictive for this purpose. Rebuild future labels from the untouched command stream using timestamps and bounded command age; retain genuine held targets. Separate **observation retention**, **loss-anchor weighting**, and **future-label availability**. Use an explicit per-slot terminal mask; never fabricate post-reset targets or unobserved terminal holds.

This predicts why extra late-phase sampling did little: it cannot recover labels already filtered out. It remains a hypothesis about the magnitude of the live benefit until a controlled repaired-label fit is evaluated.

## 5. Architecture and code assessment

The current actor is a custom diffusion transformer: three-layer 384D history encoder, six-layer 384D denoiser, roughly **14.29M trainable parameters**, plus a shared frozen **11.18M ResNet18**. It predicts full target poses relative to the decision-time TCP, not residual corrections on another actor. The task vector contains family, requested port, requested SFP card, and card-valid flag. SC card identity fields are zero by design; obstacle count/layout must be inferred from cameras. Controller-error state indices 13–18 are masked, and current SC fits hide previous teacher commands entirely.

The checked body-frame conversions and v-prediction/DDIM equations are consistent. Teacher and actor Cartesian stiffness/damping match in the examined streams. Runtime and dataset quaternion canonicalization agree. None of these checks proves every controller effect is identical. A teacher-command replay through the exact chunk executor remains useful.

Confirmed limitations:

- **Timing:** six 100 ms commands are followed by a synchronous inference gap, yielding roughly 0.9 s between decisions. The 0.9 s cache changes observation spacing but the teacher continues receiving intermediate commands while the actor holds its last target. It does not fully match execution.
- **Selection metric:** checkpoints and plateau stops use teacher-row translation error on a sparse selected subset. The diffusion evaluator does not report orientation error or terminal success. A plateau here is not optimal closed-loop transport. The 0.49 mm intentional-fit result is sampled training-row action error, not all-state control accuracy.
- **History:** adding requested ages cannot restore discarded frames; some nominal 38-frame contexts have only 14 distinct observations. Avoid clock-based phase assumptions, especially around stalls. Distance/rotation-spaced older memory plus dense recent samples is a reasonable later ablation.
- **Vision:** all three images are globally pooled before temporal fusion. Spatial relationships are compressed. The denoiser receives a single history summary rather than attending directly to spatial camera/history tokens. This may limit precision, but the current endpoint visits do not justify claiming insufficient capacity as the primary failure.
- **Rotation input:** canonical quaternions still have a discontinuity near 180°, common in these initial postures. A continuous 6D rotation input is worth a bounded sensitivity check, not an asserted root cause. Action rotations already use a 6D representation.
- **Coverage:** rotated-board SC LLB around the released −1.8 rad anchor is absent; a prior planning-only attempt failed. Current good endpoint proximity on familiar development layouts does not close that coverage gap or establish SFP improvement.

## 6. Research implications, with limits

- [Diffusion Policy](https://arxiv.org/abs/2303.04137) supports receding-horizon action diffusion. Its [author clarification](https://github.com/real-stanford/diffusion_policy/issues/9) says end-to-end visual experiments used the robomimic ResNet18 with spatial softmax; global pooled pretrained encoders were a separate experiment. Our frozen pooled encoder is not the full visual recipe. Retaining spatial features is a stronger next visual hypothesis than simply increasing denoiser width.
- [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) documents patch features. The local single-frame linear probe did not show an advantage, but it does not test a temporal spatial policy. Compare ResNet spatial features and DINO-S patches with the same trainable head, data and stopping logic. Do not conclude DINO is useless, or default to it yet.
- [Real-Time Chunking](https://arxiv.org/abs/2506.07339) addresses inference delay and consistency between overlapping chunks. This motivates a buffered executor after a teacher-replay diagnostic. Our actions use a changing TCP anchor, so overlapping targets must first be transformed into a common/new anchor. The method is not a drop-in call or a reason to revive the old previous-teacher-command shortcut.
- [DAgger](https://arxiv.org/abs/1011.0686) addresses states visited by the learner. It remains useful if errors persist after terminal/timing repair. It requires a valid expert continuation from the actual cable/robot state; an old route at the same clock time is not that label. Previous SFP continuation attempts failed safety/causal-label checks and must not be counted as successful DAgger.
- [Automatic Waypoint Extraction](https://proceedings.mlr.press/v229/shi23b.html) reduces imitation horizon by preserving approximable trajectory segments. A fallback should preserve the LLB turns and measured cable clearance. Earlier one-second and 20 mm SFP waypoint probes already failed actor-boundary checks; repeating those definitions would not be new evidence.
- [Continuous rotation representations](https://arxiv.org/abs/1812.07035) motivate testing a continuous state orientation encoding. This addresses representation smoothness, not the missing terminal actions.

## 7. Ranked next experiments — proposed, not started

1. **Agree the component boundary from evidence.** Inspect selected-port coarse/refiner predictions on saved teacher and actor endpoint windows, both requested SC ports, plus the excluded safe local approach. Separate arrival, stable stop, correct-port visual approval, and downstream insertion-start compatibility. Budget: one CPU/offline session, roughly 2–4 hours. If perception cannot work at the taught endpoint, extend transport through a verified part of the already recorded approach; do not merely relax the gate.
2. **Repair terminal data construction.** Use original timestamped commands, preserve history, downweight quiet action anchors, and independently admit stop labels. Audit all 104 episodes, manually inspect at least 12 windows spanning tilt, return, stop, both ports and card-count extremes. Gate: every verified episode has its genuine final stage represented or an explicit exclusion reason; no valid terminal labels disappear solely for lack of a full future horizon. No large new collection is needed.
3. **Test the executor without learning.** Later, run at most three existing nonfinal resets with teacher targets through the actor's exact anchor conversion/scheduler, compared with the original teacher. Track commanded/measured pose and progress separately. A proposed diagnostic budget is six rollouts total. If it cannot reproduce the taught endpoint within 10 mm/3° without harmful blockage, repair execution before training. If it passes, stop treating controller mismatch as the leading explanation.
4. **One controlled repaired-data actor comparison.** Preserve the current fine70 baseline. Same architecture, episode split and update budget; compare old versus repaired labels with a stop/proposal head trained on appropriate observations in both arms. Select on phase-specific translation, orientation and terminal metrics. Start with one GPU, up to four GPU-hours per arm, a minimum training interval and declared plateau rule; a budget stop is not convergence. First test the six reused diagnostic scenes, then freeze for six new grouped SC development starts if promising. Proposed continuation gate: at least 5/6 reach the validated acquisition region and stop without wrong-port approval or sustained blockage. This is a development screen, not production reliability.
5. **Spatial vision only if a visual gap remains.** Matched ResNet spatial pooling versus DINOv2-S patch attention, keeping state/task/history and repaired labels fixed. Two bounded arms, same update budget, at most four GPU-hours each initially. Continue only if both held-out late-route pose/action measures and live arrival/stop improve; a linear probe win alone is insufficient. Recheck complete latency after route correctness, consistent with the user's temporary relaxation of the latency constraint.
6. **Targeted corrections or route-progress architecture if needed.** Identify actual stage/progress errors, collect at most three causal safe continuation pilots first, and proceed only if at least two are valid. Harmless cable/card touch is not automatic rejection; preserve official prohibited-contact and sustained blockage/force checks. If expert labels are unavailable, test stage-conditioned short waypoints using measured route progress and explicit transition completion. Keep complete action prediction, not a residual adapter.
7. **Broader coverage and integration after the above gate.** Repair the rotated SC LLB route, retain SFP CheatCode-consistent data, then evaluate a shared actor or task-specific baselines on new grouped starts. Connect coarse approval, refiner and insertion only across an empirically validated starting envelope. Recovery/RL and reserved final scenes remain gated. No claim that SC conclusions transfer automatically to SFP.

The immediate next direction is **terminal supervision and handoff repair**, followed by execution parity. Another unconditional history/capacity sweep or broad corrective collection is not justified by the new evidence.

## Reproduction and artifacts

All new lightweight artifacts and exact analysis scripts are under `/home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/`. `artifact_sha256.json` hashes the JSON analyses. `commands.md` records the CPU commands. The archived scripts read existing data and write reports; they do not run simulation or modify datasets. They use fixed historical paths, so rerun into a new output directory to preserve this audit.

Source inputs: active SC labels `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/sc_labels.jsonl`; cache `/home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2/`; original scored streams `outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams/`; rollout roots are embedded in `endpoint_comparison.json`'s accompanying script and the earlier experiment records. No checkpoint was changed or promoted.
