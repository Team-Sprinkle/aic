# Fresh SFP anchor shadow check: ten scenes complete, identity gate failed

## Corrected all-channel crop replay (October 4)

The first crop-proposal script enumerated only physically present ports from simulator target labels. That leaked presence into the offline candidate list, although the network itself predicted all possible ports. I changed `scripts/audit_sfp_card_row_association.py` to enumerate **all ten SFP output channels** before observation-based crop selection, then replayed the same frozen models and ten scenes. The original 346-proposal counts below are preserved as the historical, presence-filtered run and must not be used as deployment evidence.

The corrected replay made **370** crop proposals. Posthoc labels place the requested opening in all three original-image crops on **362/370** proposals, with at least one such crop by 3 s in all ten episodes. The native crop refiner accepted **107/370**, including **12 wrong ports**; its accepted 3D p95 was **22.86 mm**, with an early accepted anchor in 6/10 episodes. The all-port, structured, and selected-only heads remained at 27, 8, and zero wrong accepts. Their four-head OR union had **47 wrong accepted model predictions**, 26 coaccepted disagreements above 10 mm, and early acquisition in 7/10 episodes. Thus the identity gate still fails. A causal selected-only anchor replay on corrected scores accepted 154 frames versus 101 selected-only frames, adding 53 later frames with zero wrong accepts on these scenes, yet still only 3/10 early episodes. Neither replay proves live safety or insertion.

Corrected artifacts are under `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/`: `all_ten_candidates_all_channels_v2.jsonl`, `all_ten_proposals_all_channels_v2/`, `all_ten_model_scores_all_channels_v2/`, `all_ten_union_all_channels_v2.json`, and `selected_anchor_tracking_all_channels_v2/`. The original crop refiner was trained on proposals generated with the presence-filtered script; its corrected replay checks that distribution shift, but a future refit must use all-channel training proposals too.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" .pixi/envs/default/bin/python scripts/audit_sfp_card_row_association.py --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_candidates_all_channels_v2.jsonl
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" .pixi/envs/default/bin/python scripts/audit_sfp_observed_rail_crop.py --candidates /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_candidates_all_channels_v2.jsonl --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_proposals_all_channels_v2 --source-reprojection-px 12
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" .pixi/envs/default/bin/python scripts/evaluate_sfp_fresh_anchor_models.py --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten --proposals /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_proposals_all_channels_v2 --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_model_scores_all_channels_v2
```

## October 4 continuation after quota increase

The administrator-side kernel quota changed while this work was in progress. `/proc/key-users` showed UID 1008 at `200/1024` **quota keys**, and `/proc/sys/kernel/keys/maxkeys` was 1024. An initial resume attempt still stopped because my runner mistakenly parsed the preceding `200/200` instantiated-key count. I fixed it to parse `qnkeys/maxkeys`, the fourth field after the UID as documented in [keyrings(7)](https://man7.org/linux/man-pages/man7/keyrings.7.html). The runner then completed the remaining nine fixed Gazebo scenes sequentially, with the active Isaac job untouched. After completion, quota usage was 193/1024. Host MCAP extraction worked after every scene. The initial key-quota section below records the earlier state and why execution paused; it is no longer the current blocker.

The **same ten reset configs** in `plan.json` yielded ten scored bags and **581** synchronized original three-camera samples at 0.5 s spacing. Nine ordinary CheatCode teachers fully inserted with tier-3 score 75. Card 2/port 1 instead reached partial insertion (tier-3 38.62, scored 0.04 m insertion). That episode remains in the fixed perception test; it was not replaced or used for fitting. The source agent actions were never used.

The frozen all-port candidate model made 346 observation-based crop proposals. In posthoc geometry scoring, **335/346** contained the requested opening in all three 480 px native crops. Every request had such a crop within 3 s of its first sampled image. The crop *proposal* stage therefore supplied an early view, but the learned port-refinement stage still failed identity and coverage:

| Frozen predictor | Accepted observations | Wrong accepted observations | Accepted 3D p95 | Early episodes / 10 |
| --- | ---: | ---: | ---: | ---: |
| Old all-port | 142/581 | 27 | 48.53 mm | 3 |
| Structured all-slot | 117/581 | 8 | 51.40 mm | 5 |
| Selected-only | 101/581 | 0 | 2.63 mm | 3 |
| Native crop refiner | 98/346 proposals | 11 | 22.87 mm | 4 |

The exploratory four-head OR union acquired an anchor by 3 s in **7/10** episodes and eventually in 10/10, but it made **46 accepted wrong-model predictions** and had **25** >10 mm coaccepted model disagreements. This fails both the zero-wrong and 9/10 early gates. The earlier 9/10, zero-wrong result on checkpoint-selection development scenes did not generalize. No autonomous actor, insertion handoff, or RL was promoted. Since accuracy already fails, the complete live latency gate was not run; the pilot offline image-stage p95 of 190.8 ms remains only a component measurement.

### Bounded diagnosis and one structural check

Posthoc labels show that wrong accepted results are coherent neighbor swaps, not merely noisy triangulation. All-port predicted card 2 instead of requested card 1 on **27** observations of card 1/port 0, typically about 47–50 mm away with reprojection residuals below 5 px. Structured all-slot made seven card-2→card-3 swaps and one card-1→card-2 swap. The crop refiner made eight port-1→port-0 swaps on card 0 and three on card 4, about 23 mm away. In these eleven crop failures, the correct port was **inside all three proposed native crops**; the refiner selected the neighboring aperture. I inspected representative original center RGB images for each failure type. Repeated, similar-looking port faces are visible, so three-view agreement alone cannot certify requested identity.

I ran one observation-only structural diagnostic after those failures: allow the frozen selected-only head; allow a crop estimate only when, for a same-card source pair, its 3D point lies within **10 mm** of that pair's image-predicted requested-port endpoint. Ten millimeters is less than half the roughly 23 mm same-card port spacing. No simulator label enters this decision. On **these same scenes**, it had zero wrong accepted predictions and 2.87 mm accepted 3D p95, but only **5/10** early episodes. It was designed after viewing these scenes and is **not an independent validation or deployable fix**. This bounded repair fails the 9/10 coverage gate; further threshold changes on the same ten scenes would overfit them. The next experiment needs a concrete, separately frozen hypothesis for requested-card/port identity and new non-final resets for its evaluation. The transport-to-insertion switch and RL remain disabled.

### Added commands and artifacts

```bash
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/run_sfp_fresh_anchor_shadow.py \
  --plan /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json \
  --skip-existing

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
  .pixi/envs/default/bin/python scripts/audit_sfp_card_row_association.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_candidates.jsonl

PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/audit_sfp_observed_rail_crop.py \
  --candidates /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_candidates.jsonl \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_proposals \
  --source-reprojection-px 12

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
  .pixi/envs/default/bin/python scripts/evaluate_sfp_fresh_anchor_models.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --proposals /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_proposals \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_model_scores
```

The exact successful launch log is `runner_resume_after_quota_v2.log`; `runner_resume_after_quota.log` preserves the parser-failure attempt. `progress.json` lists all ten official scores and paths. `labels_all_ten/`, `all_ten_candidates.jsonl`, `all_ten_proposals/`, `all_ten_model_scores/`, `all_ten_union.json`, and `observation_structural_gate_v1/` preserve every model prediction and the bounded repair. [Updated machine decision](2026-10-04-sfp-fresh-anchor-shadow.summary.json).

## Frozen design and data

The four perception checkpoints from the October 3–4 SFP work were frozen. The planned check has ten **new, non-final reset groups**, one for each requested card 0–4 / port 0–1 code. Nine configs are unused source-development YAMLs. Card 4/port 1 had no unused source-development YAML, so its scene was generated by an explicit board x +6 mm, y −4 mm and target-rail +3 mm perturbation of prior development `trial_000526`. [The immutable plan](/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json) records every scene, source trial, source split, task code, and reset hash. All ten hashes differ from the 50 training and ten old development resets, from each other, and from the source audit partition. No source action arrays were used. Final scenes remained sealed.

The intended frozen gate is: zero accepted wrong ports; early anchor in at least nine of ten requests, including middle cards, within 3 s of the first sampled image; accepted coarse 3D p95 ≤20 mm; and **complete live** p95 <300 ms. The predictor may use original RGB, measured TCP, fixed camera calibration, public task code and observation-selected crop proposals. Simulator port geometry is used only after prediction to score the result.

## First independent reset

The ordinary Gazebo CheatCode teacher completed `card0_port0` (`trial_000482`, five cards) with tier-3 score **75**, meaning full insertion by the teacher. This is a camera/perception test, **not autonomous actor insertion**. The scored bag gave 52 synchronized three-camera samples at 0.5 s spacing. The observed-rail rule proposed 52 crops; posthoc simulator projection found the requested opening inside all three original-image crops on all 52 samples. The four frozen models were evaluated on those images:

| Frozen model | Accepted observations / 52 | Accepted wrong ports | Accepted 3D p95 |
| --- | ---: | ---: | ---: |
| Old all-port | 36 | 0 | 4.13 mm |
| Structured all-slot | 33 | 0 | 2.30 mm |
| Selected-only | 26 | 0 | 1.58 mm |
| Native crop refiner | 25 | 0 | 1.78 mm |

The frozen union acquired the requested port on the **first** sampled image (time 2.75 s), with zero accepted wrong ports and zero coaccepted model pairs >10 mm apart. These are only one reset and correlated frames. They do not establish ten-request coverage, true occlusion handling, safe transport, or insertion. An offline GPU-1 profile of 20 saved triplets took **173.6/190.8/192.4 ms p50/p95/p99** for JPEG decode, resize, native crop, and four network forwards. It excludes ROS, camera transport, triangulation, actor, controller and scheduling, so the complete live latency gate remains untested.

## Rootless Docker stop and extraction repair

At the first Gazebo launch, UID 1008 had 199/200 kernel keys. The rollout succeeded, but a **separate** Docker extraction launch failed because the account was then at **200/200**. The key count remained 200/200 after the Gazebo container exited. No further Gazebo scenes were launched; the old active Isaac container was left intact. The previous answer that one sequential container was enough was based on a short probe that returned its key. This full Gazebo run showed that the assumption does not hold reliably.

I added a host-side MCAP reader fallback to `scripts/extract_moveit_training_stream.py`. It reads the same CDR bytes and uses the installed ROS interface packages to deserialize them, avoiding another Docker container. It successfully extracted the first bag: 459 commands and 514/515/514 center/left/right camera frames. Install the reader in a private target directory without sudo:

I also reran the host reader on an older development bag already extracted inside Docker. Its **405 commands, three message streams, 20 TF edge groups, scene/reset identity, score and timestamps matched exactly**. Only path strings differed (`/audit/...` inside Docker versus the host archive path). The comparison artifact is `/var/tmp/chmin_aic_sfp_host_extract_reference.json`.

```bash
pip install --target /var/tmp/chmin_aic_mcap_reader_py312 --python-version 3.12 --only-binary=:all: mcap==1.3.0
PYTHONPATH=/var/tmp/chmin_aic_mcap_reader_py312 \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
.pixi/envs/default/bin/python scripts/extract_moveit_training_stream.py \
  /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/teacher_card0_port0 \
  /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/teacher_card0_port0/scored_stream.json
```

The resumable sequential runner now extracts on the host after each Gazebo run and checks the UID key quota **before** attempting the next launch. Its attempted resume failed safely with `200/200; no new Gazebo container was launched`. A durable administrator quota increase or safe key cleanup is needed for the remaining nine starts. Do not revoke arbitrary user keyrings or interrupt the active Isaac job to force this check. Keep all four model weights and the ten reset configs frozen when resuming.

## Commands and artifacts

```bash
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/prepare_sfp_anchor_fresh_scenes.py \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1

PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/run_sfp_fresh_anchor_shadow.py \
  --plan /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json \
  --skip-existing
```

The archive contains `teacher_card0_port0/` with the original RGB frames, ROS bag, scored stream, score and logs; `labels_pilot_one/`; `pilot_one_candidates.jsonl`; `pilot_one_proposals/`; `pilot_one_model_scores_v3/`; `pilot_one_union_fresh.json`; and `pilot_one_offline_latency.json`. Failed evaluator attempts are preserved as `pilot_one_model_scores/` and `_v2/`: the first used the unwrapped all-port model; the second exposed a DataLoader fork deadlock after CUDA initialization. The corrected evaluation uses zero workers. [Machine summary](2026-10-04-sfp-fresh-anchor-shadow.summary.json).

When the quota is raised, finish the nine **already selected** scenes with the same checkpoints; rebuild labels over all ten; rerun frozen proposals, all four model predictions, union and full live timing. If a wrong-port acceptance occurs, stop promotion and diagnose its image and per-camera predictions. Do not tune on a final scene, train an actor, or start RL from this one-scene result.
