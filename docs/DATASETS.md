# Data and label contracts

The final demonstrations run learned perception with rule transport and explicit insertion. Historical expert action data were used for the ACT, diffusion, BC and RL experiments; they are not a claim that the final controller is a trained actor.

| Collection | What it contains | Important boundary |
| --- | --- | --- |
| `outputs/trajectory_datasets/expert_verified/` | Canonical 289 eligible episodes: 268 SFP, 21 SC; 250 train / 39 validation | Membership and score evidence come from `manifest.json`; SFP target distribution is skewed |
| `outputs/trajectory_datasets/clean_including_no_insert_trajs/` | Preserved original collection, including failures | Never treat this entire folder as successful experts |
| `outputs/trajectory_datasets/sc_llb_score_verified/` | 104 successful low-lateral-bypass SC teacher episodes from 200 attempts | Privileged teacher generation, not autonomous policy success |
| `outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/` | Separate cleaned SC transport derivative; 83 train / 21 validation episodes | Removed 19,570 quiet anchors; retained 69,681; original recordings and validation preserved |
| Native port-refiner training lineage | 72,857 RGB triplets from 497 original sources, then 20 targeted SFP acquisition episodes | Each triplet has three cameras; these are perception examples, not 517 verified BC demonstrations |
| Held-connector perception additions | Separate far/near image supervision, including near-25 mm SC captures | A perception reset need not be a successful insertion episode |

Later derivatives have their own manifests and grouped splits. Do not combine episode counts from these rows: collections overlap and serve different purposes. Exact counts, extensions and S3 lineage are in the [preserved dataset history](answers/archive/2026-10-19-documentation/datasets_full_history.md).

## Rules for reuse

- Split complete episodes and physical reset configurations. Keep reserved final configurations out of training and selection.
- Join images, measured robot state and plug transforms at the **observation/controller timestamp**. A command timestamp can be about 50 ms later; that mismatch previously changed labels by millimetres.
- Preserve actual recorded target commands. Transport actor labels use TCP-body delta poses; the historical PoseInsert-inspired actor uses complete connector targets expressed in the estimated port frame. Do not invent actions from measured motion or train on invalid zero labels.
- Account for the full connector-to-TCP transform, including grasp orientation. Gripper orientation alone is insufficient.
- Crop the original camera image before resizing. Check camera order, calibration, RGB/BGR, codec and task identity.
- Keep terminal observations before reset. Do not join post-reset geometry to a terminal transition.
- Preserve raw data, failed recordings and rejected labels; publish cleaned derivatives separately.

The historical S3 source is `s3://aic-team-sprinkle/datasets/clean/`; access and local relocation are documented in the archived history. Final media is a separate curated copy in [media/final_evaluation](media/final_evaluation/README.md), not a training dataset.
