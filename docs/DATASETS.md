# Expert trajectory locations

## Canonical expert collection (2026-09-18)

New ACT work uses `outputs/trajectory_datasets/expert_verified/`.
The original local collection has been renamed, without deleting data, to
`outputs/trajectory_datasets/clean_including_no_insert_trajs/`.
Historical experiment manifests retain their original paths; apply the prefix
mapping in `SOURCE_RELOCATION.json` at the archived root when reading them.
The S3 prefixes below were not renamed.

The current canonical collection contains **289 BC-eligible episodes**:
**268 SFP + 21 newly collected SC**, split into **250 training / 39 validation**
episodes and 169,710 cached frames. Cache v3 appended 17 SFP and 12 SC episodes
to v2's 260 episodes. Every previous record, array prefix, and split assignment
was preserved. Training and validation share no scene, and neither overlaps
the 59 reserved development/final evaluation scenes.

The initial filtered SFP collection contains **251 episodes**:
140 historical CheatCode episodes and 111 aligned expert/corrective episodes.
They comprise 141,556 cached frames, split into **223 training / 28 validation
episodes**. Exact scene/task configurations stay in the same split. No exact
scene/state/action/timestamp duplicates were found; image-level deduplication
was not performed. Seventeen additional aligned nominal expert episodes were
accepted after checking scores, labels, every image, and terminal frames. The
table below describes the **current 268-episode SFP portion**.

| NIC count | Eligible SFP episodes | Training | Validation |
| --- | ---: | ---: | ---: |
| 1 | 155 | 139 | 16 |
| 2 | 30 | 26 | 4 |
| 3 | 31 | 27 | 4 |
| 4 | 27 | 23 | 4 |
| 5 | 25 | 21 | 4 |

Target coverage is uneven: 258 episodes target card 0 / port 1. The ten
CheatCode smoke episodes target port 0, distributed across cards 0–4 as
2 / 2 / 2 / 1 / 3. Card-count coverage does not establish balanced target
coverage. The canonical task vector records family, target port, target card,
and card validity; it is now included explicitly in the ACT training cache.

### Newly verified SC coverage

The first aligned SC collection attempted 20 episodes and admitted nine; a
second collection admitted 12 of 16. The canonical SC total is therefore
**21 successes from 36 attempts**. Every admitted episode has official Tier 3
= 75, total score >=80,
no prohibited contact or force penalty, checked action/target reconstruction,
and inspected terminal camera frames. Fifteen failures remain outside the
canonical expert set. Five additional planned collection paths were never
attempted and are explicitly marked `unattempted_not_data`.

| Target SC port | SC receptacles | NIC cards | Training | Validation |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 4 | 2 |
| 0 | 1 | 2 | 4 | 2 |
| 0 | 2 | 3 | 2 | 2 |
| 1 | 1 | 1 | 4 | 1 |

These scenes use a fixed grasp, small board/rail variation, and fewer fixtures.
They do not cover SC port 1 with additional cards, or general difficult SC
layouts. The new append adds a held-out port-1 demonstration. New splits are
grouped by scene within task, target card/port, NIC-count and SC-count strata.
Sparse strata receive at least one validation scene when two independent
scenes exist, so SC alone is 14/7. Preserving old splits and small-stratum
holdouts gives the combined 250/39 split, rather than an exact 90/10 ratio.

The new cache is
`/var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3/`, also linked as
`outputs/experiments/2026-09-18_act_all_verified_6h50/cache_all_verified_sfp_sc_v3/`.
Read `expert_verified/manifest.json` → `training_cache` for the current version.
New SFP/SC source recordings are copied into the canonical folder; the NVMe
cache is derived data. Previous caches and canonical manifest versions remain
available. Append workflow: `scripts/append_verified_experts.py`; exact counts
and checks: `v3_append_result.json` and `v3_publication_audit.json` in the September 18
experiment. Initial SC evidence remains in `canonical_sfp_sc_membership_check.json`;
the second collection's terminal review is `sc_collection_gpu0_followup/visual_review.json`.

The [world visual follow-up](experiments/2026-09-18-world-followup.md) checked
all 289 episode ranges, 253 image shards, RGB/BGR provenance and scene
groups. Its matched tokenizer experiment keeps the older pilot's original
14 validation episodes sealed because 15 of the 74 pilot episodes have a
different assignment in this canonical split. The resampled image cache does
not establish how many of all 289 frames are geometrically near a port
opening; use synchronized plug/port transforms for that question.

The September 19 full tokenizer run uses all **250 canonical training
episodes** for optimization and all **39 canonical validation episodes** for
selection and reconstruction reporting. “All valid” therefore means the full
training side of the split, not training on validation episodes. Sampling gives
equal probability to historical SFP, aligned SFP, and SC, then samples an
episode uniformly within that category. Dynamics has a narrower eligibility
rule: an episode must provide native simulator timestamps, native observation
endpoints, and the actual four-command sequence for each 200 ms transition.
Visual eligibility alone does not make an episode dynamics-eligible. See the
[completed protocol and results](experiments/2026-09-19-full-world-training.md).

That strict audit retained **148/289 episodes** for dynamics (126 train, 22
validation): 127 SFP and 21 SC, with 68,884 native observations. The 140
historical video episodes have no native `frames.jsonl`; one aligned SFP episode
has frames but no actual command indices. They are excluded from dynamics
without synthesizing timestamps or commands, while their verified images remain
eligible for tokenizer training.
The prepared strict transition set contains 9,858 training and 1,486 validation
200 ms edges. The completed dynamics model used exactly this set and did not
outperform persistence at 200, 400, or 600 ms; visual eligibility must therefore
not be cited as evidence for planning-quality temporal coverage.

The **253** successful historical agent episodes (180 nominal SFP + 50 recovery
SFP + 23 SC) remain outside BC training under
`outputs/trajectory_datasets/successful_pending_label_repair/manifest.json`.
That manifest points to preserved archived data and records why each episode
is ineligible. The earlier conversational figure of 273 was an arithmetic
error. Sparse replay events and planned trajectories cannot recover exact
executed commands aligned to every recorded image; zeros occur during final
SC seating as well as initial joint motion. Actions were not invented from
observed motion. Separately, **212 scored non-insertions and 63 unresolved
episodes** are excluded from the verified folder.

`expert_verified/manifest.json` is the source of truth for membership, task,
score/source evidence, recording conventions, scene group, and split. The
folder contains complete verified CheatCode data/meta/videos and selected
aligned episode image/state/action directories. Unchanged source files use
hard links when on the same filesystem and byte-preserving copies otherwise;
consumers must treat them as immutable. The ACT cache reuses
audited image shards and converts their documented BGR/RGB conventions to RGB
when loading. Historical wall-clock timestamps remain a recorded limitation.
The small additional decoded cache is stored at
`/var/tmp/chmin_aic_20260918_act/cache_cheat10/` because `/data1` was nearly full;
it can be regenerated from the canonical recordings.

Creation command and distribution evidence:
`scripts/curate_verified_experts.py`,
`outputs/experiments/2026-09-18_act_all_verified_6h50/dataset_summary.json`,
and `source_relocation.json` in that experiment directory.

## Historical inventory and verification

Checked 2026-09-17. This page identifies collected demonstrations for the
ACT → offline SERL → online SERL workflow. The initial inventory fetched only
small S3 metadata files. The subsequent [eight-hour ACT experiment](experiments/2026-09-17-act-verified-8h.md)
audits scores, source arrays, and final camera frames and trains ACT locally.

## Main local collection

Use this root in the current checkout:

```text
/data1/chmin/yj/ws_aic/src/aic/outputs/trajectory_datasets/clean_including_no_insert_trajs/
```

There are **23 `accepted_dataset` directories, containing 668 stored episodes
and 539,480 frames**. Parquet row counts match each dataset's `meta/info.json`.
A first video frame from each of the three cameras in all 23 datasets decoded
successfully. This count is an availability inventory; training eligibility is
reported separately below. Cross-collection deduplication remains incomplete.

## Verified insertion episodes

**Historical camera/timing compatibility:** the 130-episode CheatCode source
below was recorded with the pre-fix collector, which converted camera RGB to
BGR before writing RGB video. Its cached pixels are preserved; models fitted
on those pixels require BGR live input, or a documented conversion to canonical
RGB during training. Checkpoint `image_channel_order` records that contract.
The collector also sampled on a wall-clock timer: parquet timestamps and video
PTS equal frame index / FPS, not original simulation time. Matching image PTS
does not establish a 20 Hz simulation trajectory. Do not assume these
conventions apply to later agent collections. See the [bounded ACT report](experiments/2026-09-17-act-verified-8h.md)
and its saved historical collector, image comparison, and clock audit.

The September 17 score/lineage audit found **393 episodes with official Tier 3
score 75 and accepted state/action arrays matching their raw source**. Another
212 episodes have scores indicating no insertion, and 63 have unresolved local
score/source mappings. Unresolved episodes are not counted as failures or used
in the initial ACT runs.

| Source family | Verified insertion episodes | Initial use |
| --- | ---: | --- |
| CheatCode SFP | 140 / 140 | Start with the 130-episode card 0 / port 1 collection. |
| Agent nominal SFP | 180 / 180 | Defer: many moving frames have zero recorded Cartesian actions. |
| Agent SFP recovery | 50 / 50 | Defer until action-label handling is addressed. |
| Agent SC | 23 / 298 | 212 near-gate episodes are not insertions; 63 mappings remain unresolved. |

For each source, first/middle/last episodes were sampled at the final frame and
one and two seconds before it, in all three cameras; CheatCode has additional
samples. Contact sheets were visually inspected. Some insertion views are
occluded, so images corroborate the official score rather than independently
proving microscopic seating. Raw/accepted end-frame checks for representative
CheatCode episodes agree within video-compression differences.

The 130-episode ACT cache also checks decoded frame timestamps across all three
cameras. It preserves the recorded full TCP-relative command labels and source
episode IDs, and canonicalizes quaternion signs to match the runtime. Of the
140 CheatCode episodes, only two frames combine a zero action with TCP speed
above 1 mm/s; the agent families have many such frames. Historical success and
raw-array equality still do not guarantee every action is an appropriate
imitation target: controller-state shortcuts and deployment timing are tested
in live rollouts.

Evidence: [episode audit](../outputs/experiments/2026-09-17_act_verified_8h/verification/episodes.json),
[source summaries](../outputs/experiments/2026-09-17_act_verified_8h/verification/sources.json),
and [experiment record](experiments/2026-09-17-act-verified-8h.md).

The paths below are relative to that root. Append `/accepted_dataset` to each
run path when loading LeRobot data. Wildcards describe multiple existing runs.

| Collection | Run path | Episodes |
| --- | --- | --- |
| CheatCode SFP smoke | `sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke` | 10 |
| CheatCode SFP card 0 / port 1 | `sfp_to_nic/cheatcode/nic_cards_mixed/n130__card0_port1_resume` | 130 |
| Agent/VLM nominal SFP, EC2 hard batch | `sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard` | 180, despite the `n400` target in its name |
| Agent/VLM SFP recovery, NIC counts 1–5 | `sfp_to_nic/agent/nic_cards_*/n10__sfp_nic*_start_near_gate_04mm_n10_recovery_gpt5_feedback` | 50 across five runs |
| Agent SC full-insertion collections, NIC counts 1–5 | `sc_to_sc/agent/sc_ports_2/n20__sc_ports2_nic*_full_insert_n20` | 71 across five runs: 7 / 11 / 18 / 20 / 15 |
| Agent SC explicitly stopped near gate | `sc_to_sc/agent/sc_ports_1/*stop_near_gate*` | 212 across four runs |
| Other agent SC collections | Remaining six SC `accepted_dataset` directories | 15 |

Exact paths/counts and inspection evidence:
[CSV inventory](../outputs/experiments/2026-09-17_trajectory_inventory/inventory.csv)
and [JSON inventory](../outputs/experiments/2026-09-17_trajectory_inventory/inventory.json).
These ignored artifacts are local, not supplied by a Git clone.

### What “agent/VLM” means here

The later `agent` collector uses GPT-5-mini for symbolic scene/approach strategy,
MoveIt for free-space motion, and ground-truth geometric alignment/insertion.
GPT-5 is used for failure critique in the recovery workflow. The SFP EC2 run's
`generation_config.json` records those strategy/analysis models. See the
[expert generator guide](expert_trajectory_generator.md) and the
[EC2 hard-batch request](../aic_utils/lerobot_robot_aic/config/data_generation_requests/ec2_02_sfp_hard.yaml).
This is not a VLM predicting all low-level actions directly.

A September 23 follow-up scanned 190 retained SC full-insertion
`agent_generation/replay_attempts` and replayed one low-score three-card
VLM/MoveIt joint trajectory. It also checked the one-port/five-card seed-51500
collection associated with an older cable-catch note. That collection's local
and checked S3 clean prefixes retain accepted metadata, but the original
failed route/video from the note is unavailable there. The retained score-1
three-card attempt failed on a later port-handoff miss; its transient force
peak should not be labeled a cable snag. See the
[route and replay audit](experiments/2026-09-23-ordinary-development-cable-audit.md#agentvlm-route-follow-up).

Earlier VLM experiments also exist under:

```text
outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/vlm_planner/
outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/vlm_planner_postprocessed/
```

These include individual `lerobot_dataset`/`raw_dataset` attempts. They are
separate from the 23 accepted collections counted above; their existence does
not establish success or justify adding them to ACT training.

## S3 location and current access

The original training orchestration script explicitly waited for a sync from:

```text
s3://aic-team-sprinkle/datasets/clean/
```

Evidence: [run_clean_act_serl_20260510.sh](../outputs/train/clean_sfp_sc/run_clean_act_serl_20260510.sh).
Collection manifests also retain the EC2 source prefix
`/home/ubuntu/ws_aic/src/aic/outputs/trajectory_datasets/`.

Read-only S3 access was tested. All **21 agent collections** have remotely
readable `accepted_dataset/meta/info.json` files under the same relative paths,
and those metadata files match the local bytes. Their roots are:

```text
s3://aic-team-sprinkle/datasets/clean/sfp_to_nic/agent/
s3://aic-team-sprinkle/datasets/clean/sc_to_sc/agent/
```

The older `vlm_planner/` and `vlm_planner_postprocessed/` prefixes also exist
remotely. Full data/video equality with S3 was not checked.

**The two CheatCode collections exist locally but their matching keys are
absent from the current S3 `datasets/clean/` prefix.** Do not treat their local
folder name as proof of a current remote backup. The attempted metadata reads
returned `NoSuchKey`. Preserve their local data, manifests, and score files.

The **10-episode CheatCode smoke collection was located under `dev/` instead**:

```text
s3://aic-team-sprinkle/datasets/dev/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset/
```

Its remotely read metadata matches the local bytes. The 130-episode collection
was not found at either matching `clean/` or `dev/trajectory_datasets/` key;
its remote backup location remains unconfirmed. No bulk S3 sync was performed.

## Correction to the earlier dataset audit

The [live-validation audit](experiments/2026-09-17-live-validation.md#demonstration-audit)
followed the old combined manifest's `outputs/s3_clean/` paths. That local mirror
is incomplete relative to the original `outputs/trajectory_datasets/clean/`
tree (now `clean_including_no_insert_trajs/`). Its two missing
CheatCode source paths were not evidence that the data were lost.

Both CheatCode sources were found in the broader local tree. Their selected
trial IDs map to **140 official scoring YAMLs, all with Tier 3 score 75**.
That initial lookup verified historical insertion outcomes. The subsequent
audit above additionally checks source arrays and image timestamps; it does not
rerun the expert. The previous raw audit
is retained as the result of its narrower lookup; this inventory corrects its
availability/provenance interpretation.

The earlier ACT/direct-actor dataset has **546 episodes / 389,907 frames** at:

```text
outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_h264
```

It is not the full collection. The broader source tree contains 122 additional
episodes: 71 SC full-insertion collection episodes, 50 SFP recovery episodes,
and one EC2 SC-easy episode. A later combined `*_raw32` directory also advertises
668 episodes, but its processed schema/media/training suitability were not
validated in this inventory.

## Before choosing ACT training data

- Start from explicit source runs and their accepted trial identities. Keep
  nominal, recovery, and stop-near-gate data identifiable. A directory named
  `clean`, an `n400` target, or an acceptance flag is not an insertion label.
- Counts in `generation_summary.json` can be stale. Three recovery summaries
  report fewer accepted episodes than their actual ten-episode datasets.
- The source-family audit above narrows the action-label concern: CheatCode is
  much cleaner than the agent collections, whose joint-space transport can
  produce zero Cartesian labels. Historical success alone does not validate
  every recorded action.
- The existing `prepare_clean_act_dataset.py` accepts explicit `--clean-root`
  arguments. Its default `outputs/s3_clean` would omit locally available data.
  Its generated `*_fake_scores` files are merge bookkeeping, not scoring evidence.
- Preserve original source IDs, use grouped/scene-balanced holdouts, and create
  a fresh derived dataset rather than replacing the historical one.


## New verified collections during the September ACT experiment

All paths below are under `outputs/experiments/2026-09-17_act_verified_8h/`.
These are additional local recordings, separate from the original 668-episode
inventory. No S3 upload has been performed.

| Source | Verified insertions | Eligibility and evidence |
| --- | --- | --- |
| `corrective_data_pilot`, `corrective_data_batch1`, `corrective_data_batch2` | 24 of 26 attempts | Bounded random expert-command perturbations; clean expert targets label observed states. Two partial episodes excluded. `cache_cheat130_corrective24` retains all lineage. |
| `nominal_absolute_data_pilot`, `nominal_absolute_data_batch1` | 19 of 21 attempts | Actual simulation timestamps, RGB, and recorded teacher target equal to executed absolute target. 15 training / 4 validation episodes in `cache_cheat130_nominal19`. |
| `student_corrective_data_pilot` | 1 of 1 attempt | Bounded ACT-influenced execution with privileged expert correction labels; total 92.55, Tier 3 75. This is a training collector, not a learned success. |
| `nominal_absolute20_batch1`–`batch4` | 20 of 20 attempts | Four independent five-scene batches; all accepted terminal frames inspected. Actual simulation timestamps and absolute targets. |
| `nominal_extra20_batch1`–`batch4` | 20 of 20 attempts | Additional independent clean scenes; full official insertion scores and all accepted terminal frames inspected. |
| `student_extra20_batch1`–`batch4` | 13 of 20 attempts | Bounded student interventions, clean expert targets; seven failed insertions excluded. All 13 accepted terminal sequences inspected. |
| `student_corrective20_batch1`–`batch4` | 14 of 20 attempts | Twelve episodes with recorded student interventions and two nominal episodes with no selected intervention. Six incomplete insertions excluded; every accepted episode's terminal frames inspected. |

Official scores, sampled terminal images and visual-review notes accompany the
collections. The clean and student pilot deliberately share one training scene
for control diagnostics; they are not independent reliability trials.

The latest complete combined cache, `cache_cheat130_aligned87`, contains 87 new
episodes: **70 train / 17 validation** (41,052 / 9,805 resampled frames). The
87-recording continuations used only these new training episodes for updates
and normalization. Their warm-start weights also inherit earlier training;
the 130 historical episodes remain in the cache as shared references. Their
overall validation logs additionally retain seven legacy NIC-1 episodes;
`corrective_holdout` reports recent validation alone. The final selected model
uses the preceding 74-recording stage (60 train / 14 recent validation), not
the newest checkpoint merely because it has more data. Incremental image shards use absolute paths in
`cache.json`: preserve referenced caches, or update shard roots after relocation.

These collection batches are complete. Consult the
[experiment report](experiments/2026-09-17-act-verified-8h.md) for training use. Successful outcomes and trustworthy action labels
are separate acceptance requirements.

### Strict 74-episode world-model pilot: event timing

The September 18 Dreamer pilot uses the original 60 training / 14 validation
aligned episodes, without the historical 130. Its scene audit found **73 unique
scenes in 74 episodes**: episodes 130 and 149 repeat one training scene, while
no scene crosses the training/validation split. The newly frozen paired
development and final scenes do not overlap these 73 scenes.

All 74 official bags contain a correct-port insertion event. The event occurs
**40–142 ms after the last saved observation** for the 60 training episodes and
**46–124 ms after** for the 14 validation episodes. Thus the episodes remain
verified successful command trajectories, but none contains an observed
post-insertion image/state. The aligned collector records an observation when
it sends a target on a fresh camera timestamp; the CheatCode policy then
continues that motion, checks the official insertion event, and exits after
success. The final saved image may show an approach or partial seating, but it
cannot by itself establish the exact insertion state. The later correct-port
event establishes that the expert **did** insert successfully; the missing
post-event frame is a recording-boundary problem for state/reward learning.
Do not label the last frame as a positive insertion state or train a
terminal-state reward from that proxy. The [strict-74 raw-bag audit](../outputs/experiments/2026-09-18_dreamer60_pilot/post_event_raw_bag_audit/report.json)
found short post-event controller-state tails in every bag but no camera or
video topics. The [all-verified metadata audit](../outputs/experiments/2026-09-18_world_followup/verified_bag_metadata_audit.json)
found no camera/image/video topics in any of the 149 linked aligned ROS bags;
the other 140 historical episodes link to LeRobot videos, with no event-linked
ROS bag in the canonical manifest. The additional aligned bags' state-tail
durations and historical videos' event timing remain unverified. Future
collection should save a short synchronized post-event observation tail and
explicit event timestamp. Evidence: [event audit](../outputs/experiments/2026-09-18_dreamer60_pilot/reward_gate_report.json)
and [paired scene manifest](../outputs/experiments/2026-09-18_dreamer60_pilot/paired_scenes/manifest.json).
