# No-Guard Training Plan Update - 2026-05-29

## Latest diagnosis

The guarded action/servo stack improves lateral centering and can hold the tip near the port axis, but it may be blocking axial insertion and confusing credit assignment because executed actions differ from actor actions. The preserved guarded diagnostic rollout is:

- Run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_09-46-02_eval_v456_guarded_diagnostic_v455_ckpt600_40x10_fullvideo_step1200`
- Checkpoint: `v455/checkpoints/checkpoint_000600.pt`
- Start: 40 mm axial / 10 mm lateral near-gate
- Evidence: center/left/right full-episode videos, step snapshots, metrics JSON/CSV, command, config, git status/diff, and `guarded_diagnostic_note.md`
- Result: no strict success

At the best-depth frame, the guarded rollout reached tip `s=-9.323 mm`, `r=0.073 mm`, and `theta=0.08021 rad`, while module consistency was still poor (`module_s=-32.898 mm`, `module_r=1.838 mm`, final axial consistency error `55.073 mm`). The guard was applied and predicted-r rejection/orientation recovery/module lateral alignment were active, but the selected axial command was outward/backoff. This supports treating the guard as diagnostic rather than the main learning path.

## Updated direction

Start a separate no-guard reward-only branch:

- Disable hard action override paths by config: insertion action guard, target-tip servo, final two-stage servo, module lateral alignment override, orientation recovery/backoff, contact retreat/recovery, predicted-r rejection/action replacement, prelip/offgate clamps, and final hard orientation holds.
- Keep standard policy action clipping and strict termination/logging.
- Keep strict success and post-step evaluation thresholds unchanged.
- Keep the multiplicative insertion gate: `G_insert = G_lateral * G_orientation * G_action_axis`.
- Increase gated axial-progress and action-axis incentives, while preserving module consistency and bypass penalties so tip-only depth cannot dominate.

## Initial experiment

Launch a short smoke training run from the best guarded partial-alignment checkpoint:

- Base checkpoint: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_09-33-51_train_v455_v453_isolate_offgate_theta100_from_ckpt600_40x10_policy1400_bs8/checkpoints/checkpoint_000600.pt`
- Start: 40 mm axial / 10 mm lateral episode config
- Run name: `train_v499_noguard_rewardonly_from_v455_ckpt600_40x10_smoke800_bs8`
- Training: 800 steps, actor updates enabled after warmup
- Videos: disabled during training; policy-only rollout with center/left/right videos will be saved only after a meaningful checkpoint or about 30 minutes.

## Decision rule

- If no-guard improves axial `s` while worsening `r/theta`, tune reward gates and curriculum first.
- If no-guard bypasses or destabilizes immediately, fall back only to soft safety: action clipping, failure termination, and logging.
- Do not re-enable hard corrective servo unless a controller/metric blocker requires it.

## Results so far

Strict success has not occurred in this branch. The main observations are:

| Iteration | Run | Config focus | Best usable post-step metrics | Decision |
|---|---|---|---|---|
| v499/v500 | `train_v499...40x10...`, `eval_v500...40x10...` | no-guard 40/10 from guarded checkpoint | 40/10 policy-only failed approach; final/best states stayed far outside lateral gate | Reject as too hard for immediate no-guard training |
| v501/v502 | `train_v501...20x4...`, `train_v502...20x4...` | no-guard 20/4, higher axial reward | best aligned state around `s=-28.2 mm`, `r=0.43-0.53 mm`, `theta=0.043 rad`; no axial progress | Diagnose axial-credit / reset-boundary issue |
| v503 | `train_v503...preinsert_axial_hook...` | added gated pre-entry axial reward hook | best aligned state `s=-28.30 mm`, `r=0.306 mm`, `theta=0.0435 rad`; no strict success | Hook active, but still no axial unlock |
| v504 | `train_v504...actor_axis_credit...` | action-manager action-axis credit and higher actor Q | same plateau: best aligned `s=-28.41 mm`, `r=0.400 mm`, `theta=0.0457 rad` | Reject; not an action-axis credit issue alone |
| v505 | `train_v505...soft_train_termination...` | delayed training-only offgate/lateral-bypass activation | axial improved to `s=-20.09 mm`, but `r=1.62 mm`, `theta=0.130 rad`; later lateral drift exploded | Confirms premature termination was blocking exploration, but reward gates too loose |
| v506 | `train_v506...tight_reward_gates...` | tighter r/theta reward gates after v505 | invalid best depth `s=-18.63 mm`, `r=38.6 mm`; aligned state still near `s=-26.24 mm`, `theta=0.085 rad` | Reject; 20/4 cannot yet combine approach and insertion |
| v507 | `train_v507...shallow2x0_orimodule...` | shallow 2/0 orientation and module refinement from v483 | best depth collapsed to `s=2.48 mm`, `r=0.168 mm`, `theta=0.0554 rad`; best theta `0.0487 rad` | Reject; orientation/module reward over-regularized axial progress |
| v508 | `train_v508...lowtheta40x10...` | old low-theta 40/10 reset set | post-step theta was not low in current evaluator (`theta=0.0677 rad` at step 1) and drifted to `0.354 rad` | Reject; old low-theta reset metadata is incompatible with current setup |
| v509-v512 | `resetcheck_v509...`, `resetcheck_v510...`, `resetcheck_v511...`, `resetcheck_v512...` | current-date reset theta calibration and rotvec orientation sweeps | best post-step reset remained around `theta=0.049 rad`; no variant met strict `theta < 0.030 rad` | Reject reset-quaternion-only calibration |
| v513 | `train_v513_noguard_mixed_oridepth_from_v485_ckpt400_policy1200_bs8` | mixed 40/20/10/2 curriculum from v485 ckpt400, moderate orientation/module reward | checkpoint 400 best depth `s=8.41 mm`, `r=0.223 mm`, `theta=0.0522 rad`, module `s=-15.22 mm`; worse than v485 depth and not strict | Reject; mixed curriculum destabilizes before solving final orientation/module |
| v514 | `train_v514_noguard_shallow2x0_soft_oridepth_from_v483_ckpt400_policy900_bs8` | shallow 2/0 continuation from v483 ckpt400, softer than v507 | best depth `s=7.87 mm`, `r=0.321 mm`, `theta=0.0201 rad`, but module `s=-15.78 mm` and final axial consistency error `37.96 mm` | Keep as a useful orientation-depth partial; not strict because module consistency is far behind |
| v515 | `train_v515_noguard_10x2_bridge_from_v485_ckpt400_policy1000_bs8` | 10/2 bridge with normal failure termination | best depth only `s=-4.43 mm`, `r=0.793 mm`, `theta=0.0488 rad`; repeated 20-step terminations | Reject; termination gates too early for bridge learning |
| v516 | `train_v516_noguard_10x2_softterm_from_v485_ckpt400_policy800_bs8` | 10/2 bridge with bypass/offgate/centered-sweep terminations disabled | apparent depth `s=28.14 mm` came with `r=57.3 mm`, `theta=0.115 rad`, module `r=54.8 mm`; reward near `-3892` | Reject; no hard termination is unstable and produces tip-depth false positives |
| v517 | `train_v517_noguard_10x2_delayedterm_from_v485_ckpt400_policy800_bs8` | 10/2 bridge with delayed 80-step failure termination | checkpoint 200 best depth `s=5.70 mm` occurred with `r=42.6 mm`, `theta=0.0782 rad`; clear lateral bypass | Reject; delayed termination remains too permissive |
| v519 | `train_v519_noguard_shallow2x0_actorhist_from_v514_ckpt200_policy600_bs8` | actor state history plus existing ConvNeXt/history critic from v514 | stopped after checkpoint 300; best depth only `s=-0.57 mm`, `r=0.494 mm`, `theta=0.0513 rad`, module `s=-24.19 mm`; strict false | Reject; actor history reinitializes the adapter input layer and regresses the useful v514 insertion transient |
| v520 | `train_v520_noguard_shallow2x0_contact_module_from_v514_ckpt200_policy700_bs8` | stronger force penalty, stronger module consistency/bypass penalty, smaller action clips | stopped after checkpoint 200; force spikes reduced, but best depth only `s=-0.56 mm`, `r=0.687 mm`, `theta=0.0510 rad`, module `s=-24.19 mm`; strict false | Reject; contact/module penalties suppress the only axial transient instead of making it module-consistent |
| v521 | `train_v521_noguard_shallow2x0_resumefull_from_v514_ckpt200_policy500_bs8` | new opt-in full online state resume; same reward/action settings as v514 | online critics, target critics, and actor/critic optimizers restored with no shape warnings; stopped after checkpoint 200; best depth only `s=-0.57 mm`, `r=0.533 mm`, `theta=0.0504 rad`, module `s=-24.20 mm`; strict false | Keep the resume feature, reject this run; v514's positive-depth event was not recovered by optimizer/critic continuity alone |
| v522 | `train_v522_noguard_shallow2x0_collect_replay_from_v514_ckpt400_step180` | no-update rollout from v514 ckpt400 with new replay save path | replay saved 180 transitions to `replay_buffer.pt`; best depth only `s=-0.57 mm`, `r=0.494 mm`, `theta=0.0497 rad`, module `s=-24.20 mm`; strict false | Keep replay-save feature, reject this replay as useful insertion data because it contains no positive-depth/near-success transition |
| v523 | `train_v523_noguard_shallow2x0_loadreplay_v522_from_v514_ckpt400_policy220` | full-state resume plus preloaded v522 replay | replay preload worked (`loaded=180`, replay size began at 181); run drifted to a false positive at best depth `s=4.97 mm`, `r=41.7 mm`, `theta=0.0717 rad`, module `r=40.0 mm`; strict false | Reject v522 replay; it is not a useful teacher and can destabilize into lateral bypass |
| v524 | `train_v524_noguard_shallow2x0_collect_filtered_replay_from_v514_ckpt400_step90` | no-update rollout from v514 ckpt400 with metadata-backed filtered replay save | saved 60 of 90 transitions with `centered_high_theta_or_positive`; best depth only `s=-0.57 mm`, `r=0.662 mm`, `theta=0.0515 rad`, module `s=-24.20 mm`; strict false | Keep filtered replay plumbing, but treat the dataset as centered pre-insertion/high-theta only |
| v525 | `train_v525_noguard_shallow2x0_filteredreplay_v524_from_v514_ckpt400_policy220` | full-state resume plus filtered v524 replay preload | loaded 60 transitions; stable training, but best depth stayed `s=-0.57 mm`, `r=0.648 mm`, `theta=0.0512 rad`, module `s=-24.20 mm`; strict false | Reject as an insertion-improvement path; filtered alignment replay alone did not restore axial progress |
| v526 | `train_v526_noguard_shallow2x0_reproduce_v514_collect_positive_replay` | rerun v514 recipe from v483 ckpt400 with positive-centered replay capture | did not reproduce the v514 positive-depth transient; best `s=-0.566 mm`, `r=0.633 mm`, `theta=0.0553 rad`, module `s=-24.19 mm`; positive-centered replay saved 0 rows | Reject; the useful v514 event was rare/non-reproducible under the current deterministic rerun and cannot yet seed replay |
| v527 | `train_v527_noguard_shallow2x0_guideimitation_noexecute_from_v483_ckpt400` | privileged guide action used only as actor imitation loss; `collect_blend=0`, no guard/action override | guide loss was active at step 320 (`target_action_guide_loss=2.03e-4`, weighted `8.14e-6`), but best `s=-0.566 mm`, `r=0.272 mm`, `theta=0.0510 rad`, module `s=-24.19 mm`; strict false | Reject low-weight no-execute guide imitation as configured; it is stable but too weak to unlock axial progress |
| v528 | `train_v528_noguard_shallow2x0_strongguide_noexecute_from_v483_ckpt400` | 10x stronger no-execute guide imitation; lower actor-Q pressure | guide loss was active and 10x stronger (`target_action_guide_loss=2.25e-4`, weighted `8.99e-5`), but best `s=-0.843 mm`, `r=0.769 mm`, `theta=0.0476 rad`, module `s=-24.47 mm`; strict false | Reject stronger always-on guide imitation; it slightly improved theta but worsened lateral/depth and still did not insert |

## Current diagnosis

The no-guard branch is training, not just evaluating, but it has exposed two separate blockers:

1. On 20/4 starts, strict failure terminations were activating near the reset boundary and preventing approach learning. Delaying those training-only terminations allowed axial exploration, but the policy then exploited axial movement with lateral/orientation regression.
2. On shallow starts, existing policies can produce some tip depth with good lateral alignment, but semantic tip orientation remains around `0.048-0.055 rad`, above the strict `0.030 rad` threshold, and module depth consistency remains far behind the tip.

The next bounded step should be a reset/orientation calibration pass in the current depth-corrected setup, not another high-weight reward sweep. Specifically: generate or repair current-date 2/0, 10/2, 20/4, and 40/10 episode configs whose post-step reset theta is actually below `0.030 rad` under the strict checker, then rerun no-guard training from the best shallow checkpoint and re-expand outward.

## Update after v513-v517

The reset/orientation calibration pass was run before v513: the port-index bug in `validate_serl_reset_settle.py` was fixed, then current 20/4 reset validation and small world/body reset-orientation sweeps were tested. None produced strict post-step theta; the unmodified reset remained best at about `0.049 rad`. Small reset quaternion perturbations are therefore exhausted for this blocker.

The subsequent no-guard training variants confirm that the current policy/reward/curriculum cannot yet bridge from 10/2 to insertion without either early termination or lateral bypass. Hard termination protects against bypass but prevents learning; removing or delaying it gives axial motion only as false-positive lateral bypass. One useful partial result remains v514: it combined nontrivial depth with strict tip orientation, but module consistency was still far behind, so it is not success.

The next code change should not be another scalar reward sweep. The most likely missing piece is a curriculum/evaluator representation change that exposes a learnable module-consistent final-insertion target, for example offline residual targets or a supervised auxiliary loss from the best shallow/guarded trajectories, while keeping executed actions policy-owned during RL.

## Update after v519-v520

Two bounded structural probes were run after the reset-randomization audit:

- `v519` added `--actor_state_history_steps=4` while keeping v514's ConvNeXt/history critic. It was stable but worse than v514: no positive insertion depth appeared by checkpoint 300. This suggests actor history is not a drop-in fix when resumed from older adapter checkpoints because the wider adapter input layer starts partly fresh.
- `v520` preserved the no-guard policy-owned action path and increased force/module/bypass penalties with smaller action clips. It lowered force/contact excursions, but it also removed axial progress. This confirms that simply penalizing the v514 contact spike is too blunt; the policy needs a constructive module-consistent insertion target rather than stronger suppression.

A residual-target audit was written under `outputs/agentic_reward_curriculum_20260529/residual_target_audits/v520_guarded_v483_v514_v519/`. Across guarded v456 and no-guard v483/v514/v519 it found 1428 centered-high-theta rows, 304 contact-spike rows, 792 lateral-bypass rows, and zero safe centered-progress rows under strict gates. The best centered partial remains v483 step 385: `s=17.37 mm`, `r=0.445 mm`, `theta=0.0514 rad`, module `s=-6.25 mm`, module `r=0.842 mm`, still not strict.

Current recommendation: do not continue scalar penalty sweeps. The next implementation should turn these centered high-theta/module-lag states into an explicit supervised or auxiliary residual target, or add full online checkpoint/resume so rare v483/v514 insertion transients can be preserved instead of rediscovered from a fresh critic each run.

## Update after v521

`aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` now has an opt-in `--resume_online_state` flag. When enabled, online SERL checkpoints restore matching critic, target critic, actor optimizer, and critic optimizer state; incompatible state falls back to the previous fresh-state behavior with a diagnostic warning. The default remains actor-only resume to preserve old training paths.

The first v521 smoke run verified that the restoration path works, but it did not recover v514's rare positive-depth transient. This narrows the blocker: critic/optimizer discontinuity was a real engineering gap, but not the primary reason strict insertion is missing. The next useful implementation is a trajectory-level auxiliary target or dataset collection path that can repeatedly present the v483/v514 centered high-theta/module-lag states to the policy instead of hoping online exploration revisits them.

## Update after v522-v523

`aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` now also has opt-in replay persistence:

- `--save_replay_at_end` and `--save_replay_path` write the current replay buffer.
- `--load_replay_path` and `--load_replay_max_transitions` preload saved transitions before online updates.
- The load/save reports are written into `train_config.json`.

The feature was validated in Isaac. v522 wrote a 180-transition replay, and v523 loaded all 180 transitions before training. However, v522's replay did not contain a positive-depth or near-success event, and preloading it made v523 worse, with a lateral-bypass false positive at `s=4.97 mm`, `r=41.7 mm`, `theta=0.0717 rad`.

Current recommendation: keep replay persistence, but do not preload arbitrary shallow rollouts. The next useful change is a filtered replay/dataset collector that only saves transitions satisfying strict near-success gates or known useful partials, such as centered rows with `r <= 0.7 mm`, module `r <= 1.5 mm`, and either positive tip depth or explicit high-theta/module-lag labels. Without filtering, replay reuse amplifies bad resets and lateral-bypass transitions.

## Update after v524-v526

The replay path now stores post-step geometry metadata per transition and supports filtered saves. v524 verified this in Isaac by saving 60 centered/high-theta transitions from a no-update v514 checkpoint rollout. v525 verified filtered replay loading, but the loaded alignment replay did not produce axial progress. v526 then reran the v514 shallow no-guard recipe from the original v483 checkpoint with the same seed and a positive-centered replay filter; it remained near the entrance and saved zero positive-centered transitions.

This makes the current evidence sharper: the v514 `s=7.87 mm`, `r=0.321 mm`, `theta=0.0201 rad` partial is still the best low-theta depth event, but it is not reproducible enough to act as a replay source, and it was not module-consistent. The next training iteration should stop trying to replay entrance-hover data and instead add a constructive module-consistent target, such as an auxiliary residual/imitation term or depth-gated curriculum stage that explicitly teaches the module body to follow the tip through the port while keeping policy-owned actions.

## Update after v527

v527 tested the least invasive constructive target: the privileged target-tip/module guide was computed and used as a supervised actor loss, but not executed (`target_action_guide_collect_blend_effective=0.0`, `--no-target_action_guide_train_executed`, no insertion action guard). This preserved policy-owned rollouts and avoided hard action replacement. The run was stable and the guide loss was wired, but the loss was too small relative to the existing policy/reward dynamics and did not move the policy past the entrance-hover regime.

Current recommendation: do not continue with low-weight guide imitation. The next bounded branch should either increase the no-execute guide/imitation weight by an order of magnitude for a short smoke test, or make the guide target phase-specific so it supervises only the final centered/orientation-blocked window. If that still fails, the remaining blocker is likely not reward scale but the absence of repeatable, module-consistent final-insertion examples.

## Update after v528

The 10x guide-imitation variant was stable but still did not unlock insertion. It improved the best observed theta to `0.0476 rad` but stayed outside the lateral gate (`r=0.769 mm`) and did not advance axially (`s=-0.843 mm`). This rejects the idea that the privileged guide target can be used as a simple always-on supervised loss from this checkpoint.

Current recommendation: stop running always-on no-execute guide weight sweeps. The next constructive target must be phase-conditional: apply guide/imitation only to centered final-window rows, or first generate a guide-only/controller trajectory that reaches module-consistent positive depth and then imitate that data. Without that, the policy is only being taught to hover/trim near the entrance.

## Update after v529-v530

`train.py` now supports phase-gated no-execute guide distillation through `--target_action_guide_train_phase_filter` and related phase thresholds. Defaults preserve the historical behavior (`all`). The first smoke test, `v529`, used `centered_high_theta` samples only, repeated selected samples 4x, and kept policy execution unguarded. The implementation was validated with `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` and `python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` (`31 passed`).

`v529` selected most shallow centered rows and trained stably, but it still did not insert: best `s=-0.564 mm`, best `r=0.037 mm`, `theta=0.0506-0.0516 rad`, module `s=-24.19 mm`, strict false. This rejects phase-gated imitation from entrance-hover data as currently configured.

`v530` then removed guide imitation entirely and trained the policy with stronger gated axial/action-axis reward (`axial_progress_weight=1.80`, `action_forward_scale=5.5e-5`, stronger preinsert aligned axial and module progress). It was stopped after checkpoint 300 because post-step metrics remained in the same hover regime: best `s=-0.568 mm`, best `r=0.065 mm`, best `theta=0.0481 rad`, module `s=-24.19 mm`, strict false.

Current recommendation: continue policy training only after changing the data/curriculum state distribution. The repeated shallow 2/0 reset now reliably teaches entrance alignment but not axial/module insertion. The next bounded branch should generate or expose final-window module-consistent intervention data, or shift curriculum to repeatable shallow-positive/module-following starts. More scalar reward or guide loss on the same hover distribution is unlikely to produce full insertion.

## Update after randomized-curriculum pivot

The single-reset branch was stopped. Recent v529/v530 and earlier v519/v520/v524/v526 runs all used `full_depth_start2x0_v464_settle_centered_from_v462` with `num_envs=1`, so the policy repeatedly saw the same shallow entrance state. That is consistent with the observed behavior: good entrance hover/alignment, but no robust axial/module-consistent insertion.

I added `aic_utils/aic_isaac/scripts/build_randomized_near_gate_curriculum.py` to generate auditable randomized episode folders from the calibrated v464 reset. The first accepted mixed reset bucket is `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v543_interleaved_from_v464/train_mixed_50_30_20`, ordered so the first 8 envs include shallow/final, near-gate, and bridge samples. Its validation run `validate_v543_reset_train_mixed_50_30_20_interleaved` had no immediate terminations and produced post-step reset ranges:

| bucket | s min/mean/max mm | r min/mean/max mm | theta min/mean/max rad | module_s min/mean/max mm | module_r min/mean/max mm |
|---|---:|---:|---:|---:|---:|
| v543 mixed first 8 envs | -20.07 / -4.53 / +0.34 | 0.96 / 1.52 / 2.54 | 0.0509 / 0.0543 / 0.0574 | -43.69 / -28.15 / -23.28 | 0.83 / 1.06 / 1.78 |

Rejected reset/config attempts:

| config/run | decision | reason |
|---|---|---|
| v534 original randomized heldout 40x10 | reject | synthesized from shallow base produced lateral errors up to 30.9 mm; use existing fixed 40x10 dirs for held-out eval. |
| v511 world orientation sweep | reject | post-step lateral error was centimeter-scale (`r≈24-35 mm`) and theta was `0.071-0.095 rad`. |
| v512 body orientation sweep | reject | even worse theta (`0.105-0.149 rad`) and centimeter-scale lateral error. |
| v496 shallow settle2d | reject | consistent ~22 mm lateral offset. |
| v548 tip-preserving generator | reject | implementation bug normalized vector rotation and created impossible reset offsets. |
| v549/v550 tip-preserving generator | reject | numerically fixed but shifted effective reset distribution away from intended shallow/final-window samples. |

Training smoke results on randomized curriculum:

| run | checkpoint/source | config | best post-step metrics | strict |
|---|---|---|---|---|
| v544 | v483 checkpoint 400 | no-guard reward-only, v543 mixed, 8 envs, completed 400 steps | best s `1.95 mm`, r `1.41 mm`, theta `0.0580`, module_s `-21.68 mm`; best theta `0.0447` at s `0.78 mm`, r `0.175 mm`, module_s `-22.87 mm`; zero samples below `0.03 rad` | false |
| v551 | v544 checkpoint 200 | no-guard, v543 mixed, stronger orientation shaping, stopped after checkpoint 100 / step 139 | best s `1.95 mm`, r `1.41 mm`, theta `0.0580`, module_s `-21.68 mm`; best theta `0.0452`, zero samples below `0.03 rad` | false |

This validates the user's diagnosis: single-reset shallow training was insufficient, and the first randomized curriculum is physically plausible but still has a reset/theta floor around 0.05 rad. Reward-only training with stronger orientation shaping did not cross the strict orientation threshold and did not create module-consistent axial insertion.

Current recommendation: do not resume long training from v544/v551 without fixing the reset/orientation distribution. The next code change should create a calibrated low-theta randomized reset generator that preserves the semantic tip position without perturbing reset solver reference fields, then validate shallow/final, near-gate, and 40x10 held-out buckets before any long training. If that cannot produce post-step theta below 0.03 rad in reset/eval, the blocker is reset/controller geometry rather than SERL reward scale.
