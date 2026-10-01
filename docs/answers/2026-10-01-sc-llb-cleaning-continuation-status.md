# SC dataset cleanup and training continuation — 2026-10-01

I preserved the original `sc_llb_score_verified` dataset and prepared a separate [cleaned-dataset specification](../../outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json) for all **104 officially successful SC episodes**. Each episode's score, bag, action trace, insertion events, and two deliberate route holds passed the metadata audit. The split is by complete scene and robot start: **83 train / 21 development validation**, with both selected ports and every card count represented in each partition. The original source manifest hash is unchanged.

The cleanup code is in `scripts/prepare_sc_llb_cleaned_dataset.py`, `scripts/filter_sc_llb_bc_chunks.py`, and `scripts/run_sc_llb_cleaned_materialization.py`. It drops only the interiors of identified long scripted command holds, keeps their transition edges, and preserves every recorded 200 ms target chunk it retains. The local filter is conservative: it removes only genuinely repeated low-motion, low-force windows; it does not erase slow insertion or contact responses wholesale. On an existing materialized SC example, the hold detector found both expected plateaus (171 and 111 repeated anchors), while the local rule deleted none of the moving/contact samples.

**The cleaned action dataset is not finished, and no BC or evaluation has started.** This managed session cannot access the rootless Docker socket (`permission denied`), exposes no NVIDIA driver, and lacks the ROS bag reader locally. The resumable extraction runner stopped at its Docker preflight, before writing partial action chunks. When run in the project's rootless Docker/GPU environment, the exact command is:

```bash
.pixi/envs/default/bin/python scripts/run_sc_llb_cleaned_materialization.py \
  --spec outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized
```

After that, audit the real raw-versus-cleaned action chunks, train matched clean-init transport BC with verified SFP and SC data, train the local actor with observation-only predicted pose, calibrate selected-port visibility/handoff, and run autonomous development evaluations. RL and final scenes remain gated. The [experiment record](../experiments/2026-10-01-sc-llb-cleaned-bc.md) lists the commands, validation completed, and remaining gates. `STATUS.md`, `DATASETS.md`, and `EXPERIMENTS.md` reflect this state.
