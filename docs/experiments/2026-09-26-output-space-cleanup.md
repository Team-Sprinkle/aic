# Output storage cleanup, September 26, 2026

Scope: `/data1/chmin/yj/ws_aic/src/aic/outputs`. This cleanup is about old
generated artifacts, not the canonical expert collection. `/data1` was full
at the start. No active process using the candidate paths was found by a
process-command audit. Directory sizes below are `du -sx -B1` on this host.

## Removed locally

| Path | Measured bytes | Decision |
| --- | ---: | --- |
| `outputs/train/hf_smoke/` | 13,875,208,192 | Deleted old May smoke-training checkpoints and short test runs. No maintained-document reference or selected September model pointed here. |
| `outputs/hf_converted/sfp2nic_target_card0_port0_randomized_smoke/` | 3,389,894,656 | Deleted three smoke conversion variants. This was derived from old SFP-to-NIC data, not the canonical expert collection. |
| `outputs/train/hf_sweeps/` | 53,616,238,592 | Deleted May 8–9 non-selected parameter-sweep checkpoints. Before deletion, the 926 non-checkpoint files (configs, metrics, logs; about 24 MB) were copied to `/home/nas/chmin/aic_archive_20260926/metadata_only/hf_sweeps/` and verified with a recursive checksum dry-run. A local README points there. |

The first two directories were removed without a backup because they were
smoke-only derivative artifacts. The sweep checkpoints were removed after
retaining their small result metadata; the weights themselves were not
backed up. The measured total removed from these three trees was about
70.9 GB (decimal), subject to filesystem block accounting. The local free
space rose from zero to about 67 GB after these removals.
The exact paths, measured bytes, and backup disposition are in the
[machine-readable deletion record](2026-09-26-output-space-deletions.json).

## Backed up before local release

`outputs/agentic_reward_curriculum_20260527/` (71,646,138,368 measured bytes)
and `outputs/agentic_reward_curriculum_20260528/` (119,977,185,280 measured
bytes) are historical May policy-run trees, not source expert
demonstrations. They have no maintained-document path references in the
current audit. They were moved to
`/home/nas/chmin/aic_archive_20260926/old_policy_runs/`. May 27 passed a
complete recursive checksum comparison and equal count of 26,164 files;
May 28 passed the same comparison and equal count of 3,372 files. Both
original paths now resolve through NAS symlinks, releasing about 191.6 GB
decimal locally.
The migration script compares every file through a recursive checksum
dry-run and equal file counts before replacing a local tree with a symlink.
The exact completed manifest is the
[machine-readable migration record](2026-09-26-output-space-migration.json).
Neither local tree was removed before its complete verification passed.

## Protected or deferred

- Keep `outputs/trajectory_datasets/expert_verified/` (about 4.3 GB):
  canonical verified expert episodes for future supervised training.
- Keep `outputs/trajectory_datasets/clean_including_no_insert_trajs/`
  (about 65 GB) for now. It includes failed original trajectories, including
  examples used to investigate SC cable behavior. It is a candidate for a
  **verified NAS migration**, not an unbacked deletion.
- Keep `outputs/s3_clean/` (about 54 GB) until provenance and S3 recovery
  checks establish that all needed raw videos and bags have another copy.
- Keep September experiment records, selected checkpoints, recent SC route
  videos/bags, and the original source trajectories that produced them.
- Review `outputs/experiments/2026-09-17_act_verified_8h/cache_cheat130/`
  (about 45 GB of decoded image cache) separately. It is likely
  regenerable, but its split and historical ACT reproducibility must be
  checked before deletion.
- Review old `outputs/hf_combined/` (about 68 GB),
  `outputs/train/sfp_to_nic/` (about 116 GB), and May 29 policy runs
  (about 196 GB) separately. These contain derived data or historical
  models; do not delete entire trees without identifying selected weights
  and source provenance.

No training was run for this cleanup. The NAS mount has ample space but is a
separate failure domain, so hash verification precedes local release.
After all five cleanup actions, `/data1` showed **244 GB free** (`df -h`)
instead of zero at the start. Existing paths to the migrated May policy runs
still resolve through symlinks.
