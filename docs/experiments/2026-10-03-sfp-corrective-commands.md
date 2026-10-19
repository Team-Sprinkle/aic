# Bounded SFP corrective continuation: command record

All commands ran from the repository root on October 3, 2026. The simulator launchers use the pinned image in their scripts and the user's rootless Docker socket. The frozen plan and every attempted simulator command, environment choice, return code, scene, and audit location are in the [collection progress](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_teacher_train20_v1/progress.json). Nine second-tranche attempts returned Docker exit 125 before simulation because the kernel session-key quota was 200/200. No launch was retried with changed scene/model settings.

```bash
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/collect_sfp_corrective_teacher_train.py \
  --balanced-train /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_teacher_train \
  --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/ordinary_bc_best_v1.pt \
  --output /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_teacher_train20_v1 \
  --limit 10 --gpu 1

# The same command with --limit 20 attempted the second tranche. Only one
# additional container ran. It was not admitted to the training set.
```

`scripts/extract_sfp_corrective_teacher_bags.py` decoded the first ten scored bags with `--expect 10 --follow`; `scripts/materialize_sfp_scored_transport.py --corrective-teacher --split train --episode-index-start 2000000` consumed their ten audit paths in frozen plan order. The exact admitted paths and controller-frame cutoff/causal-join metrics are in the [materialization manifest](/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/corrective_train10_materialized_v2/manifest.json). Its first partial attempts were preserved as `corrective_train10_materialized_v1` and `corrective_train10_materialized_v2` is the accepted version; `corrective_train10_materialized_v1` failed the former arbitrary TF-event lag check.

```bash
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/merge_sfp_corrective_transport_labels.py \
  --base /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/balanced_merged_sfp_v1/sfp_labels.jsonl \
  --corrective /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/corrective_train10_materialized_v2/transport_chunks.jsonl \
  --output /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/corrective_train10_merged_v2

# Offline cache and training used GPU 2 with the local Pixi environment because
# new rootless containers were blocked. Their input/output, data provenance,
# architecture, sampler mass, learning rate and update cap are frozen below.
cat /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_10ep_fullaction_v1.command.txt
cat /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_10ep_fullaction_v1/manifest.json

PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/run_sfp_corrective_bc_development.py \
  --collection /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_teacher_development \
  --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/corrective_10ep_fullaction_best_v1.pt \
  --output /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_10ep_autonomous_3dev_v1 \
  --gpu 1
```

The [live progress](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/corrective_10ep_autonomous_3dev_v1/progress.json) includes each exact rollout command and checkpoint path. Posthoc geometry was read by `scripts/score_sfp_autonomous_transport.py` from matched teacher and autonomous traces. `scripts/diagnose_sfp_balanced_route_divergence.py --candidate-progress ...` and `scripts/diagnose_sfp_corrective_direction.py` produced the route and command diagnostics. See the [experiment decision](2026-10-03-sfp-balanced-arrival-onpolicy-check.md#october-3-bounded-corrective-full-action-continuation-failed-live-gate).
