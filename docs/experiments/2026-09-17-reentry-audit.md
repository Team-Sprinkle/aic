# 2026-09-17: repository and experiment reentry audit

Status: finished documentation/source audit. No new training or live simulator
evaluation. Code inspected at `7534090` on `feat/hybrid-train`.

Follow-up: the [evaluation/curriculum repair record](2026-09-17-evaluation-curriculum-fixes.md)
documents changes made after this audit. Findings below describe the inspected
base revision and historical artifacts; they are retained as evidence.

Subsequent documentation cleanup moved older reports and handoffs into
[`obsolete/`](../../obsolete/README.md). Original paths quoted as audit evidence
below refer to their locations at inspection time; navigation links follow the archive.

## Scope and sources

Reviewed selected source paths, historical reports, saved ACT metadata, the May
25 strict-success rows, Docker metadata, and June run logs. This is not a complete
audit of the large training loop or all saved runs.

The [June 17 tutorial](https://github.com/yoonjung0705/tutorials/blob/master/python/libraries/tutorial_aic.md#617-2026-hybrid-train-branch-model-training-especially-for-isaac)
was retrieved through the GitHub connection after unauthenticated requests
returned 404. File blob SHA: `1fbe5013f4b6831e5cf328824af4756b5dae4f86`.
Its proposed direct visual actor/DINOv2 direction and reported inserted-start
expert withdrawal are recorded in [status](../STATUS.md). The withdrawal report
was not reproduced here.

The supplied rootless guide was read at
`/home/chmin/code/ws_aic/ROOTLESS_DOCKER_GUIDE.MD`. The separate checkout under
that workspace is not the checkout reviewed here.

The July handoff file
`docs/agentic_axial_first_inserted_start_status_20260715_code20260614.md` was
already untracked at the start of this pass. Its content was preserved. If this
documentation is committed, include that existing note so the ledger link is
available in a fresh clone.

## Recovered June evidence

Source: stopped container `isaac-lab-base`, directory
`/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937/`.
Read using `docker cp`, without starting the container. Host copies now live in
`outputs/reentry_audit_20260917/june_run/` (ignored by Git).

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `events.jsonl` | 252312 | `0774bdf6133b1d41c5a3b08a2f0df8956b8d7d67dc918be87006b91a87157b6a` |
| `eval_cycle0200_level000_summary.json` | 1138 | `af5a7f5b25571075d9bdb8d3bf3d3d10df46692046c084e9e62fd49d54449e27` |
| `eval_cycle0200_level000_config.json` | 103990 | `e025efbfc1fa9e3bda8358467c0f2bc16988e00404c735ceedbece886053f136` |

Events cover June 14 14:59:37 UTC through June 19 13:32:34 UTC: 200
`eval_done` events, zero success flags, zero promotions, and 50
`assessment_due_no_promotion` events. Final level is 0 and nominal
`episodes_used` is 2000. The event log names the final checkpoint at:

```text
<source-run>/train_cycle0200_level000/2026-06-19_12-57-20_stateful_train_cycle0200_level000/checkpoint_latest.pt
```

The checkpoint bytes were not loaded or recovered in this pass. The final eval
config records `act_direct`, no guide/guard, settle steps 10, orientation
tolerance 0.04, and `target_reward_consistency_body=none`.

Final summary has 890 metric rows, `strict_success=false`, count 0, and zero
reported guide blend/guard/action override. Its independent best samples are:

| Sample | Signed depth mm | Lateral mm | Theta rad |
| --- | ---: | ---: | ---: |
| Best depth, post-step 889 | 42.599 | 36.655 | 0.24517 |
| Best lateral, pre-step 1 | −40.641 | 0.513 | 0.03806 |
| Best orientation, post-step 890 | −41.433 | 2.649 | 0.03243 |

These do not describe one successful pose. The zero success count is a result
under that run's own checker, which differs from earlier module-consistent
checks. It should not be described as a standardized held-out episode rate.

## Source findings

| Finding | Source and interpretation |
| --- | --- |
| Promotion threshold/demotion unused | [Axial config](../../configs/axial_first_inserted_start_curriculum.yaml) declares rate 0.8 and demotion after 3 failures. [Wrapper](../../tools/train_stateful_promotion_gated_insertion.sh) reads only episode budget, eval interval, policy, and level count from `progression`; its loop promotes on summary boolean `strict_success`. |
| Success samples are not unique episodes | Wrapper `summarize_metrics()` scans pre- and post-step geometry and increments for each true flag. It excludes post-step rows for terminated envs. Without episode IDs/terminal semantics this is not a success rate. It also skips malformed JSON lines and can produce a false/empty summary without flagging missing evidence. |
| “2,000 episodes” is nominal | Wrapper `run_segment()` is bounded by wall time/steps; the outer loop always adds `EVAL_EVERY` to `episodes_used`. `run_segment()` records a child exit status locally but accepts any surviving `checkpoint_latest.pt`; it does not require a clean training exit. |
| No-progress event does not intervene | Wrapper logs `assessment_due_no_promotion` and resets its assessment timer. There is no stop or hyperparameter update at that branch. |
| Strict criteria drift | Wrapper `COMMON_FLAGS` disables consistency-body checks, changes colliders, and uses 0.04 rad orientation. Historical May strict diagnostics used module consistency and 0.03 rad. Collider changes are explicit experiment assumptions, not validated simulator equivalence. |
| Actor still depends on ACT action context | [Offline actor](../../aic_utils/lerobot_robot_aic/lerobot_robot_aic/vision_offline_serl.py), `ACTAdapterSERLActor.delta_action()`, concatenates state encoding and base action. [Isaac actor](../../aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py), `IsaacACTAdapterActor.action_components()`, does the same even in `act_direct`. Direct mode changes output composition, not the head's visual inputs. |
| Silent state shape coercion | That Isaac method truncates or zero-pads `actor_state` to `adapter_state_dim`. This may be intentional compatibility behavior, but can conceal schema mismatch. Validate feature identities/order as well as dimensions before transfer. |
| Incorrect insertion label | [Score parser](../../aic_utils/gazebo_rl/gazebo_rl/score_parser.py), `parse_scoring_yaml()`, sets `insertion_success` from nonzero `tier_1`. [Official tier definition](../../aic_scoring/include/aic_scoring/TierScore.hh) defines Tier 1 as model validation. Its existing [test](../../aic_utils/gazebo_rl/test/test_score_parser.py) also expects the incorrect mapping, so that test passing would not establish correctness. |
| Runtime evaluator needs explicit checks | [Evaluator](../../scripts/evaluate_act_checkpoints_runtime.py) defaults to command mode `none`, returns 0 from `--once-existing` even for an empty checkpoint list, and skips any existing summary. It restarts the specified container, so use a dedicated one. |
| Retention is not archival | Wrapper defaults its output to `/tmp` and deletes training-cycle directories older than its recent window. Ignored host outputs and container-local checkpoints require an explicit backup. |

These findings are based on source inspection. The score-label issue also has a
small reproduction: a synthetic score file with Tier 1 = 1 and Tier 3 = 0 is
parsed with `insertion_success=true`. No scorer or trainer code was changed.

## Other evidence inspected

- ACT 175k export sidecar: state 82, action 6, chunk 8, three cameras,
  CUDA export device `cuda:0`; corresponding CPU export and pretrained
  normalizer files exist. Dataset `meta/info.json` exists. Model weights were
  not loaded for a policy rollout.
- ACT May 13 evaluation summary under
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/runtime_eval_official_compare_act175k_20260513/175000/`:
  `policy_ready=false`, engine return null, score path null; no `scoring.yaml`.
- [May 25 diagnostic](../../obsolete/docs/agentic_reward_curriculum_best_run_20260524.md): the
  saved `strict_success_rows.json` contains env 40 step 4 with s = 45.990 mm,
  r = 0.454 mm, theta = 0.02947 rad, consistency = 0.844. This is local
  privileged seating from 43.5 mm, not learned insertion from an uninserted pose.
- v865 artifacts and v1077 checkpoint/config exist. The v1077 config identifies
  `act_adapter` and `small_conv`; neither artifact's performance was rerun.
- `submission_handoff/artifacts/`, its referenced Gazebo eval directory, and
  `docker/aic_submission/Dockerfile` are absent. The historical submission
  commands therefore do not constitute a buildable bundle in this checkout.
- Docker reports `name=rootless`; the Isaac bind mount points to this checkout.
  Existing AIC/Isaac containers are stopped. Host GPUs are visible; GPU access
  inside a newly started container was not tested.

## Verification performed

- `python3 scripts/evaluate_act_checkpoints_runtime.py --help`: passed.
- Existing `.pixi` Python imports Torch and YAML successfully.
- Existing `audit_axial_first_inserted_start_reward.py`: exit 0, 13 named cases,
  64 random cases, no reported failures. This is formula evidence only.
- Checked referenced artifact existence and recovered JSON/event content without
  simulator startup. Small metadata files were copied to host storage; model
  artifacts were not removed or rewritten.
- New/changed documentation links and shell examples were checked locally.

Next work is the evaluation/reset audit in [STATUS.md](../STATUS.md), followed
by a scored baseline and bounded architecture comparison. The documentation
changes preserve older reports, add historical notices at common entry points,
and establish one maintained status/ledger/workflow/artifact map.
