# SFP board rail-reference probe (October 5)

## Question and frozen scope

Previous direct requested-port and all-slot models often confused neighboring cards. The independent native-RGB slot audit mapped only 4/10 full occupancy patterns, and the wider-view test did not reveal an obvious number label. This bounded diagnostic asks whether the **fixed five-rail board pattern** provides a learnable reference: predict the first and last rail locations from three wrist images, interpolate the five rail positions using the known CAD spacing, and predict which rails are occupied. This is a perception prerequisite, not a camera-check handoff or insertion policy.

Use only saved nonfinal original 1152×1024 SFP images. Train on 70 complete reset groups (66 previous plus four successful one-/two-card groups); choose checkpoint on ten separate three-/four-/five-card development groups. Score three saved one-/two-card **autonomous actor** reset groups separately after checkpoint selection. No actor diagnostic, failed redirected CheatCode episode, or reserved final scene enters the fit. Run on GPU 1 only for at most **2,000 updates or 120 minutes**. The existing Isaac GPU 0 process remains untouched.

The model receives three original RGB images. It does **not** receive simulator board pose, rail projections, occupancy labels, port geometry, scene YAML, task code, or a geometry-chosen crop. Those fields are training/scoring labels only. The shared pretrained ResNet18 processes all cameras, a two-landmark heatmap head predicts rail 0/4 in each camera, and a fused head predicts five occupied/empty bits. The two endpoints impose a straight, evenly spaced CAD rail template instead of fifteen unrelated slot/port predictions. The checkpoint is chosen by grouped-development loss only. All saved failures and ablations stay in the archive.

The predeclared diagnostic gate is **at least 8/10 exact five-bit occupancy maps** on grouped development, at least **two correctly mapped sparse actor groups**, and **zero confident occupied predictions for truly empty rails** at the frozen 0.8 probability cutoff. The previous slot/paired-port model reached 0/10 early accepted requested targets; a constant train-scene occupancy baseline predicts all five rails occupied and scores 1/10 exact development maps and 0/3 sparse actor maps, with six and two confident empty-slot errors respectively. Endpoint pixel errors and early availability must also be reported. Passing this diagnostic would authorize only a later requested-port correspondence test on new nonfinal starts. It would **not** enable automatic handoff, PoseInsert or RL; complete p95 inference must later be measured below 300 ms.

## Label feasibility and visual check

[`scripts/audit_sfp_board_reference_projection.py`](../../scripts/audit_sfp_board_reference_projection.py) derives candidate board-outline and nominal rail-center projections from the board CAD, each saved scene YAML, measured TCP and fixed camera calibration. It discards camera/state pairs older than 10 ms and keeps reset-group separation. The complete outer board outline never fits in one wrist image in the 3,919 retained triplets, so a whole-board-corner model is unsupported by these observations. The **first and last rail** both project inside at least one camera early in **70/70 train and 13/13 development/actor groups**; 1,969/3,690 train triplets and 171/229 evaluation triplets have this property. Projection is only a necessary field-of-view check. [Posthoc labeled sample images](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v2/rail_projection_spotcheck>) show the five projected positions lining up with board grooves or installed cards in one-card, two-card and development scenes. Occupied-card anchors can be occluded, which is a remaining risk; no point is labeled truly visible from projection alone.

The strict [v3 rail-label derivative](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/summary.json>) contains 3,690 train, 119 grouped-development and 110 actor-diagnostic triplets. For every row, five scene-YAML occupancy bits match the recorded card count; no reset group crosses train/evaluation. The [constant baseline](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/constant_mean_baseline.json>) is scene weighted, not frame weighted. A one-update [smoke fit](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_smoke_v1/summary.json>) checked model forward/backward, checkpoint save, three-camera inference and grouped evaluation.

## Commands

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_board_reference_projection.py \
  --source /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1 \
  --source /home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_projection_v1 \
  --source outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --max-state-image-age-ms 10

CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_sfp_board_reference.py \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --init /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1 \
  --updates 2000 --minutes 120 --batch-size 4 --workers 4
```

## Completed bounded fit and decision

The full fit finished normally at the **2,000-update cap** in 689 s, using GPU 1. Lowest grouped-development loss selected the 2,000-update checkpoint. It was not stopped early for saturation. The saved [history](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/history.json>), [checkpoint](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/best.pt>), [summary](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/summary.json>), and prediction JSONL files preserve the comparison. The strongest intermediate exact-map score was 6/10 at update 1,800, but that checkpoint was **not** selected by the predeclared loss rule and still missed the occupancy gate.

| Complete reset groups | Constant occupancy baseline | RGB rail model, selected checkpoint | Gate |
| --- | ---: | ---: | ---: |
| Three-/four-/five-card grouped development | 1/10 exact maps; 6 confident empty-slot errors | **5/10 exact maps; 3 confident empty-slot errors** | ≥8/10; zero confident errors |
| One-/two-card saved autonomous actor routes | 0/3 exact maps; 2 confident errors | **3/3 exact maps; zero confident errors** | ≥2/3; zero confident errors |

The actor-route result is encouraging but is only three previously saved reset groups; it does not override the failed independent development gate. For example, one held-out three-card scene assigned 0.82 probability to an empty fourth rail. The model's nearly zero training occupancy loss versus five development mistakes is evidence of poor scene generalization, not evidence that more updates would fix it. The current camera-check deployment stays disabled.

The [independent endpoint scorer](../../scripts/score_sfp_board_reference_probe.py) reads frozen predictions and label manifests without changing the checkpoint. On 557 in-frame development endpoint observations, 2D pixel error was **12.0 px median / 26.4 px p90** at the original 1152×1024 resolution; the first three seconds were 12.5/26.0 px. On 554 sparse actor-route endpoint observations, error was **17.5/95.0 px**, with the first three seconds 9.9/23.8 px. Projected in-frame flag accuracy was 97.8% and 98.0%, respectively. These are errors against **CAD-projected rail centers**, not visually reviewed physical landmarks or 3D port-pose accuracy. [Development overlays](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/development_score>) and [actor-route overlays](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/actor_diagnostic_score>) show red RGB predictions and green posthoc projection labels. I inspected original-size center/left/right examples: the markers commonly fall near the same board groove/card, while some endpoints are clipped or occluded. The images do not show that the requested numbered port is identified.

```bash
.pixi/envs/default/bin/python scripts/score_sfp_board_reference_probe.py \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/development_predictions.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/development_score

.pixi/envs/default/bin/python scripts/score_sfp_board_reference_probe.py \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/actor_diagnostic_predictions.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/actor_diagnostic_score
```

**Decision:** The fixed-rail occupancy representation improves on an all-occupied baseline and maps the three sparse actor scenes, but fails the declared grouped-development gate. Do not connect it to automatic handoff or use it as requested-port identity. Stop this fit branch here. The next distinct diagnostic should test an observation-only *local* board/groove cue on the actual autonomous approach views, with manually reviewed requested-slot evidence and a predeclared wrong-port abstention gate. True target visibility and safe-pause labels remain missing. No new routine data collection, insertion actor, RL, or final-scene evaluation is authorized by this result. Final scenes remain sealed.
