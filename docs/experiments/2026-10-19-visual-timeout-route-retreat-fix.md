# SC visual-timeout retreat fix and released-YAML rerun

## Change and scope

The prior frozen V56 run on the released `eval_config.yaml` scored **98.826833/300**. Its SC trial reversed 322 measured waypoints after a visual-acquisition timeout. The source had counted `sample_visual_acquisition_budget` as a physical blockage and emitted the misleading `force_or_measured_blockage` reason.

The user requested the fix and a rerun. In `RunSampleTransportRecovery.py`, only force stop or `sample_stage_no_measured_progress` can now request a full SC route retreat. A visual timeout ends the attempt at the conservative hold. The controller had already spent its 35-second acquisition budget searching bounded nearby viewpoints; this change does not pretend another identical search will find the target. The existing short near-port retreat and lateral retry were not changed. All 159 other selected inputs matched their previous hashes; `source_delta.json` records the one source hash change. There was no training, model change, scene change, or reserved-final access. The released qualification scenes are exposed, so this is a matched diagnostic, not fresh final evidence.

## Command and artifacts

```bash
.pixi/envs/default/bin/python scripts/run_corrected_transport_official_evaluation.py \
  --output /home/nas/chmin/aic_archives/2026-10-19_V56_visual_timeout_no_route_retreat_official_match_v1 \
  --gpu 4 --domain 143
```

The archive contains the exact released YAML, launcher and policy environment, frozen weights, corrected source, hashes, raw ROS bags, `rollout/runtime_trace.jsonl`, five-camera 20 fps videos, scoring YAML, `analysis_summary.json`, terminal geometry, `corrected_comparison.json`, and `manual_recorded_image_review.json`. The rootless run completed on GPU 4. Source syntax and whitespace checks passed. The comparison script requires exactly the intended source file to differ before launch and rechecks all selected inputs afterward.

## Results

| Trial | Previous score | Corrected score | Outcome |
| --- | ---: | ---: | --- |
| SFP target NIC card 2 / port 0 | 96.826833 | 96.856989 | Inserted; corrected physical depth 46.757 mm, lateral 0.190 mm, orientation 0.220° |
| SFP target NIC card 4 / port 1 | 1 | 1 | Force stop before port acquisition; short retreat not activated |
| SC target port 1, three NIC cards | 1 | 1 | Visual-acquisition timeout at 82.15 simulated seconds; no route retreat or insertion |
| **Total** | **98.826833/300** | **98.856989/300** | **1/3 strict insertions; no meaningful score improvement** |

The corrected SC trace has **zero** `transport_route_backtrack_requested`, `transport_measured_route_retreat_complete`, `measured_retreat_complete`, and `lateral_retry_complete` events. The SC terminal plug remains 163.917 mm outside the opening with 38.426 mm lateral error and 81.942° orientation error. These are post-hoc physical measurements, not runtime inputs. The previous 322-waypoint, 44.75-second reversal is gone. The tiny total-score difference is ordinary rollout variation in the successful SFP trial, not a recovery improvement.

SC still never obtains a valid requested-port pose. Its 114 visual inspections reject with `native_pose_geometry_rejected` (47), `native_refinement_exceeds_crop_margin` (38), or `requested_projection_not_in_all_views` (29). At 45 and 54 seconds in the saved five-camera video, blue SC port structures appear in the wrist views while the robot remains near the acquisition area. Those frames alone cannot establish which port the model identified; the causal rejection trace gives the controller's actual decision. The SFP second trial still has a force stop during transport, before any local insertion recovery.

## Decision

The **trigger bug is fixed and verified in one full released-YAML rerun**. It solves excessive backtracking on a missed visual lock, not the underlying SC acquisition failure or SFP cable/card obstruction. The corrected source is not a broadly qualified deployment candidate. Further pose/crop/viewpoint work should use new development starts, with a declared stop rule, rather than iterating on these exposed released scenes. A useful next bounded test is to compare the three SC rejection reasons and view coverage on episode-grouped development starts, then test one targeted perception or viewpoint repair only if it improves requested-port locks without wrong-port accepts. The reserved final split remains sealed.

Review frames and full videos: `/home/nas/chmin/aic_archives/2026-10-19_V56_visual_timeout_no_route_retreat_official_match_v1/per_trial/trial_{1,2,3}/five_cameras_20fps_grid.mp4` and `per_trial/trial_3/manual_review_{45s,54s}.jpg`.
