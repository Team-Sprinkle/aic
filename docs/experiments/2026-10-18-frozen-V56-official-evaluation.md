# Frozen V56 evaluation on the released qualification configuration

## Scope and frozen method

The user requested the best configuration on the official `eval_config.yaml` after the public-sample repair loop reached **290.192549/300**, with3/3strict insertions and passing used-stage p95. This is one complete evaluation of all three released qualification trials, in original order from the configured robot home. It is not a live cloud submission, the default public sample, or an untouched test: these released scenes were previously exposed. The reserved internal final split remains sealed.

Authoritative configuration: [official toolkit eval_config.yaml](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml). The runner downloads the authoritative bytes and checks them against the existing exact released scene template; the robot xacro home must match every configured joint. It records SHA-256, full YAML and task order before launch. No scene values are fed into route rules, priors or learning.

Selected controller: `RunVisualConnectorServoQuietContextV56`. Rule transport, requested-port coarse/native refinement, RGB held-plug pose, and explicit visual insertion are unchanged. No BCactor actions or RL. SC local-backoff mode remains selected; it must not be called hierarchical recovery. All selected source/model/calibration hashes are checked before and after the run. The SCclose-pose model is still unqualified globally; the user's explicit request authorizes this one retrospective diagnostic extension. Original approval files and their false qualification flags are preserved.

## Evaluation plan

1. Verify no conflicting active jobs; keep rootless Docker, GPU4/domain143 and cooperative locks. Leave the legacy GPU0job untouched.
2. Copy the selected frozen artifacts; verify against authoritative released YAML and configured robot home. Preserve all models, gains, gates and recovery settings.
3. Run the three official trials once, sequentially, recording native camera views,20fps review frames, robot/command streams, scoring bags and offline oracle contacts. Oracle data is not a policy observation.
4. Report the official300point score and per-round outcome. Independently measure depth, lateral error and orientation from the retained terminal observation before reset. Preserve acquisition and insertion failures.
5. Review recorded images and used-stage p50/p95/p99. Save all commands, source/model hashes and results. Stop after this evaluation; do not tune or retrain from its outcomes.

## Commands and artifacts

Artifact root: `/home/nas/chmin/aic_archives/2026-10-18_V56_official_qualification_evaluation_v1/`.

```bash
.pixi/envs/default/bin/python scripts/run_frozen_candidate_official_evaluation.py \
  --output /home/nas/chmin/aic_archives/2026-10-18_V56_official_qualification_evaluation_v1 \
  --gpu 4 --domain 143
```

`plan.json`, `evaluation_only_scope.json`, `selected_candidate.json` and `command.json` document scope, input checks and the actual container environment. `scene/eval_config.yaml` is the unchanged released YAML. `rollout/results/scoring.yaml` is the toolkit score. `analysis_summary.json`, `per_trial/*/physical_terminal.json`, `gap_analysis.json`, review images and five-camera videos hold post-hoc evidence. `state.json` is current execution state. No result is available at launch; append measured outcomes below when complete.

## Completed result

**98.826833/300; 1/3 strict physical insertions.** This is the released official qualification YAML's three trials, not the default sample that scored290.192549. The selected V56 source and model hashes match before and after; the released YAML matches the upstream downloaded bytes. All three tasks ran in their original order. No threshold, waypoint, model or scene change was made after observing the score.

| Official round | Task and scene | Toolkit /100 | Physical depth mm | Lateral mm | Orientation degrees | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SFP, target NIC card2 / port0 | 96.826833 | 46.762860 | 0.198830 | 0.231596 | Inserted; toolkit75 completion |
| 2 | SFP, target NIC card4 / port1, intervening card3 | 1.000000 | -74.756392 | 137.182556 | 2.470114 | Force stop before port acquisition |
| 3 | SC, three obstacle NIC cards, target SCport | 1.000000 | -109.563536 | 268.592650 | 81.787115 | Visual-acquisition timeout, route retreat, then controller wall budget |

Negative depth means the plug remained outside the opening in the independent, pre-return geometry audit. Each failed round receives only the toolkit's one validation point. The official scoring log records no excessive-force **penalty**: trial2's >20N excursion lasted0.19s (max27.63N) and trial3's lasted0.04s (max69.04N), both shorter than its1s penalty duration. That does not mean contact was harmless; the controller has a separate immediate guard.

### What failed and how it was checked

- **Trial1:** one visual handoff, then explicit RGB insertion. The pre-return bag measures46.763mm depth,0.199mm lateral,0.232° orientation. Toolkit awards75 insertion points. The recorded camera sheet agrees with a visibly seated NIC module.
- **Trial2:** there is no visual port lock, near handoff or insertion attempt. At simulation time28.5s, the controller's raw force guard reads42.05N and stops. Oracle contacts, consulted **after** execution, join cable link6 to `nic_card_mount_3` near the stop. The target is card4. Recorded overhead/side/camera views show the cable bent across the intervening card while the plug remains above and away from its port. This supports a cable/card transport obstruction; it does not identify every force component.
- **Trial3:** no requested-port lock, near handoff or insertion occurs. The route reaches its tilt step around50.8s, then **visual acquisition times out** near86.15s. The immediately preceding `sample_rule_progress` and `rule_terminal` events both report `sample_visual_acquisition_budget`. The route-recovery code erroneously treated that timeout as blockage; its `force_or_measured_blockage` event reason was misleading. The offline contact window includes cable link1 against the right gripper finger, but does not establish that contact as the retreat trigger. The controller walks322 measured path waypoints backward and completes retreat at130.9s; the return itself takes roughly44.75simulation seconds. It then starts another route but reaches the controller's **175s wall-clock guard** at simulation time141.45s. The official task limit is180simulation seconds; the wall guard was the observed termination and is a controller limitation. The terminal video shows the arm still among the cards, away from the selected SC opening. This is a visual-acquisition failure followed by an unnecessary full-route reset, not evidence that a card or cable blockage required that reset.

These are transport/recovery failures in this run. The accurate insertion seen in the default sample did not get a chance to operate on rounds2/3. No BC actor actions, RL updates or privileged geometry entered runtime. The enabled SC route retreat was triggered once; it did not complete a successful recovery on this setting. Final internal scenes stayed sealed.

### Post-run recovery-trigger correction

After freezing the official result, inspection of the causal trace found `sample_visual_acquisition_budget` immediately before the SC `transport_route_backtrack_requested` event. The transport code grouped that timeout with true force stop and measured no-progress, then emitted the generic `force_or_measured_blockage` label. This overstates the evidence for physical blockage. The live source was corrected so a visual-acquisition timeout ends the transport attempt at its conservative hold rather than reversing the whole route. A later matched released-YAML diagnostic confirmed zero route backtracks but no insertion improvement; see the [October 19 rerun](2026-10-19-visual-timeout-route-retreat-fix.md). The original frozen controller, score, trace and video remain unchanged.

| Actually used inference stage | Calls | p50 / p95 / p99 ms |
| --- | ---: | --- |
| Transport proposal | 325 | 48.57 / 164.75 / 201.69 |
| Fresh target/grasp acquisition | 3 | 193.44 / 203.25 / 204.12 |
| Explicit servo | 85 | 5.82 / 7.40 / 8.30 |
| Stationary scene check | 1 | 87.90 / 87.90 / 87.90 |

All measured used-stage p95 values are below300ms. There is no close-grasp timing sample because neither failed task reaches that phase. The three fresh calls and one scene check are too few to establish a broad latency distribution.

### Recorded artifacts and decision

The raw score is `rollout/results/scoring.yaml`; physical metrics and causal per-trial views are in `analysis_summary.json` and `per_trial/trial_*/physical_terminal.json`. `official_evaluation_metrics.json` contains the latency summary; `manual_recorded_image_review.json` names actually viewed terminal and route images. `rollout/oracle_contacts.jsonl` is strictly an offline diagnostic. Source/model/input hashes, exact launch environment and official scene copy are preserved in `plan.json`, `command.json` and `frozen/`.

Five-camera20fps videos:

- `per_trial/trial_1/five_cameras_20fps_grid.mp4`
- `per_trial/trial_2/five_cameras_20fps_grid.mp4`
- `per_trial/trial_3/five_cameras_20fps_grid.mp4`

**Decision:** freeze and report this evaluation. No repair or repeat is selected from these released scenes. Any further transport/recovery development should use independently declared pre-release-specification development cases and a separate held-out split, then evaluate the frozen revision once. The earlier public-sample result is valid for its three easier cases but does not predict success on this released qualification configuration.
