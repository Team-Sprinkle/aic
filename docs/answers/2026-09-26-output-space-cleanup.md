I cleaned up the old generated outputs. `/data1` went from **0 to 244 GB free**; the local `outputs` tree is now about **1.0 TB**.

| Action | Folder | Local space recovered |
| --- | --- | ---: |
| Deleted | `outputs/train/hf_smoke/` — old May smoke checkpoints | 13.9 GB |
| Deleted | `outputs/hf_converted/sfp2nic_target_card0_port0_randomized_smoke/` — derived smoke conversion | 3.4 GB |
| Deleted checkpoints; kept 926 config/metric files on NAS | `outputs/train/hf_sweeps/` — non-selected May parameter sweeps | 53.6 GB |
| Moved to NAS; old paths remain symlinks | `outputs/agentic_reward_curriculum_20260527/` and `...20260528/` | 191.6 GB |

The two moved trees passed recursive **content checksums and file-count checks** before their local copies were released: 26,164 and 3,372 files respectively. The NAS mount must be available for those old symlinked paths to work.

I kept the [canonical verified expert collection](/data1/chmin/yj/ws_aic/src/aic/outputs/trajectory_datasets/expert_verified/) (289 BC-eligible episodes), the original collection containing failed/non-insertion trajectories, the S3-derived sources, recent September results, and selected checkpoints. The failed trajectories remain useful for cable-snag analysis.

Other sizable candidates need a provenance check before deletion: the old ACT decoded-image cache (~45 GB), derived `hf_combined` datasets (~68 GB), older SFP-to-NIC training runs (~116 GB), and May 29 policy runs (~196 GB). I left them intact.

The [cleanup ledger](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-09-26-output-space-cleanup.md) links the exact [deletion record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-09-26-output-space-deletions.json) and [verified NAS migration record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-09-26-output-space-migration.json). I checked the protected paths, JSON, and `git diff --check`. No training was run.

Saved copy: [this answer](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-09-26-output-space-cleanup.md).
