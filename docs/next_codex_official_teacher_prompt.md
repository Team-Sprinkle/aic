# Next Codex Session Prompt: Official Teacher Reliability Work

You are working in the `Team-Sprinkle/aic` repo at `/home/ubuntu/ws_aic/src/aic`.

## Current Goal

Continue improving the official-container-compatible teacher trajectory pipeline.
Do not use or revive the fragile custom `aic_gym_gz` stepping environment. The
runtime path must remain official ROS/Gazebo + `aic_model.Policy` +
LeRobot/recording.

The current pipeline is:

1. Slow oracle planning run with GPT/VLM allowed.
2. Generate `PiecewiseTrajectory`.
3. Postprocess to `SmoothTrajectory`.
4. Replay through `aic_teacher_official.OfficialTeacherReplay` in official eval.
5. Record LeRobot dataset.
6. Build review bundle with sampled frames, observations, actions, score
   breakdown, and container scoring logs.
7. GPT-5 failure analysis feeds GPT-5-mini planner for the next loop.

## Important Current Status

There is a known high-score trajectory from earlier work:

- Base run: `trial9_2026_0425_205620`
- Best copied loop path:
  `outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_1`
- Score: about `97.02`
- Official score message: `Cable insertion successful.`

Recent repeated trials show the looped VLM-improvement process is not reliable:

- `trial11_2026_0426_010036_loop_1`: `96.770794556434552`
- `trial11_2026_0426_010036_loop_2`: `57.321630231652357`
- It stopped after loop 2 because loop 2 did not improve.

The likely regression was not missing scorer context. GPT-5 completed after
payload compaction and GPT-5-mini generated a plan. The generated loop 2 replay
physically plateaued with TCP around z `0.237m` even though the smooth target
descended to z `0.064m`, causing partial insertion.

Latest interrupted trial:

- Trial namespace: `trial12_2026_0426_011638`
- `loop_1`: `57.302546263419025`, partial insertion
- `loop_2`: `57.163913115585288`, partial insertion
- `loop_3`: `0`, task/model execution failed
- The run was interrupted and the process group was terminated. No ROS/eval
  processes should be running, but verify with `ps` before continuing.

## Recent Code Changes To Understand

Read the repo before editing. Key files:

- `aic_teacher_official/aic_teacher_official/review.py`
  - builds review bundles
  - compact GPT payload via `_compact_manifest_for_gpt`
  - includes container/eval scoring messages and log excerpts
  - includes derived per-sample geometry:
    - planned/recorded TCP-to-port vector and distance
    - recorded-vs-planned TCP error
    - TCP controller error norm
    - delta action norm
    - wrist force norm
  - generates visual aids:
    - XY path plot with port marker
    - XZ height plot with port marker

- `scripts/official_teacher_iterate.py`
  - orchestrates loops
  - has hard child-process GPT timeout
  - has `--stop-if-not-improved`
  - has conservative-success guard:
    - if a prior loop has official `tier_3` insertion success, geometry is
      locked from the best successful prior piecewise trajectory
    - GPT/VLM can be recorded in metadata, but cannot freely rewrite geometry
    - only tiny timing changes are allowed based on scorer messages

- `aic_teacher_official/aic_teacher_official/vlm_planner.py`
  - GPT-5-mini planner returns Cartesian delta waypoints in base_link
  - max budget is up to 20 calls, but currently one planner call is used

- `aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py`
  - replay policy defaults to `relative_delta_gripper_tcp`
  - this may be a source of controller/replay mismatch

## Verification Commands Already Passing

Run these after changes:

```bash
pixi run python -m pytest aic_teacher_official/test/test_official_teacher_pipeline.py -q
pixi run python -m pytest aic_model/test/test_policy_delta_pose.py aic_utils/lerobot_robot_aic/test/test_generate_trajectory_dataset.py -q
```

At last check they passed:

- official teacher tests: `22 passed`
- existing smoke tests: `13 passed`

## What To Do Next

1. Inspect the current code and artifacts. Do not assume this prompt is complete.
2. Verify no eval/ROS processes are running:

```bash
ps -eo pid,ppid,pgid,stat,etime,cmd | rg 'official_teacher_iterate.py|launch_policy_recording_per_trial|aic_policy_recorder|ros2 launch|gz sim' || true
```

3. Read the latest artifacts:

```bash
for i in 1 2 3; do
  p=outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial12_2026_0426_011638_loop_${i}/scores/trial_1_trial_000001/scoring.yaml
  [ -f "$p" ] && echo "loop_$i" && sed -n '1,100p' "$p"
done
```

4. Try a few full or partial trials until you find one where:
   - loop 1 is high but loop 2 drops, or
   - loop 1 is unexpectedly low, or
   - a later loop does not improve.

Use a fresh base run name each time, e.g.:

```bash
BASE_RUN_NAME=$(date -u +trial13_%Y_%m%d_%H%M%S)
pixi run python scripts/official_teacher_iterate.py \
  --root-dir outputs/trajectory_datasets \
  --base-run-name "$BASE_RUN_NAME" \
  --engine-config outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/oracle_engine_config.yaml \
  --context-json outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/piecewise_trajectory_from_official_v2.context.json \
  --seed-piecewise outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/piecewise_trajectory_from_official_v2_slow_final.json \
  --start-loop 1 \
  --max-loops 4 \
  --force-all-loops \
  --use-vlm \
  --use-gpt5-review \
  --review-samples 10 \
  --score-threshold 98 \
  --openai-timeout-sec 180 \
  --record
```

If you want to stop immediately after a regression, add:

```bash
--stop-if-not-improved
```

5. When a regression or low score is found, analyze before more blind trials:
   - Compare `scoring.yaml` tier messages.
   - Compare final recorded TCP vs port position from context JSON.
   - Compare planned smooth target vs recorded TCP at equidistant timestamps.
   - Check if replay/action mode is causing the controller to plateau.
   - Check whether `relative_delta_gripper_tcp` is producing deltas that move
     away from the intended absolute target in some phases.
   - Compare `absolute_cartesian_pose_base_link` against
     `relative_delta_gripper_tcp` on the same smooth trajectory if safe.
   - Inspect policy logs for insertion event timing and replay finish timing.

Useful quick analysis snippet:

```bash
pixi run python - <<'PY'
import json, pyarrow.parquet as pq, numpy as np
base = "outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/<BASE_RUN_NAME>_loop_{}"
context = json.load(open("outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/piecewise_trajectory_from_official_v2.context.json"))
port = np.array(context["port_position"])
for i in [1, 2, 3, 4]:
    root = base.format(i)
    data = f"{root}/raw_dataset/data/chunk-000/file-000.parquet"
    try:
        df = pq.read_table(data).to_pandas()
    except Exception:
        continue
    vals = []
    for idx, row in df.iterrows():
        st = np.array(list(row["observation.state"]), dtype=float)
        pos = st[:3]
        vals.append((float(row["timestamp"]), idx, np.linalg.norm(pos - port), pos.tolist()))
    print("loop", i, "rows", len(df), "best distance", min(vals, key=lambda x: x[2]), "final", vals[-1])
PY
```

## Current Hypotheses

1. The VLM planner is no longer the only problem. Even the seeded successful
   plan sometimes scores low in later runs, suggesting replay/controller
   nondeterminism, action-mode mismatch, or scene/state reset differences.

2. The `relative_delta_gripper_tcp` action mode may be problematic for long
   smooth absolute trajectories. In failed runs, recorded TCP can plateau far
   above the target while actions remain large. Test whether
   `absolute_cartesian_pose_base_link` is more reliable for replaying a fixed
   smooth trajectory.

3. GPT-5 feedback can over-optimize for smoothness/force and harm insertion.
   The conservative-success guard should prevent free geometry rewrites after a
   successful insertion, but it only helps once a successful prior exists in the
   current trial namespace.

4. Review context now includes enough scorer and geometry context. If GPT-5
   still gives poor advice, improve the prompt constraints rather than adding
   more raw context. Specifically tell it that official `tier_3: Cable insertion
   successful` is authoritative even if sampled final TCP distance looks
   imperfect because sampling and insertion-event timing may differ.

## Recommended Next Implementation Steps

1. Add a replay reliability experiment CLI or script that replays the same
   smooth trajectory in both action modes:
   - `relative_delta_gripper_tcp`
   - `absolute_cartesian_pose_base_link`

2. Compare final TCP-to-port distance, insertion success, duration, and whether
   recorded TCP plateaus.

3. If absolute pose replay is more reliable, switch official teacher replay
   default or make action mode chosen per trajectory metadata.

4. Strengthen the GPT review prompt:
   - Treat official `tier_3` success as authoritative.
   - For a successful insertion, do not recommend changing final geometry.
   - Recommend only small local timing changes targeting the weakest tier-2
     component.

5. If loop 1 with the known seed keeps alternating between ~97 and ~57, focus
   on deterministic replay first before more VLM planning changes.

## Do Not

- Do not delete existing trajectory datasets.
- Do not depend on custom `aic_gym_gz.step()`.
- Do not fallback silently if GPT-5 failure analysis times out. It is a critical
  component; fail clearly and inspect payload size/latency.
- Do not let VLM rewrite final insertion geometry after a successful official
  insertion.
