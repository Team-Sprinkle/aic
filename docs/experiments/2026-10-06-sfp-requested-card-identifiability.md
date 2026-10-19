# SFP requested-card identifiability and opposite-side view

**Status:** bounded diagnostic on nonfinal scenes. No model trained or promoted.

## Frozen review

`scripts/prepare_sfp_requested_card_review.py` froze 40 unmarked three-camera original-RGB panels at `/home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/`: 30 teacher samples (10 requests at route, approach and near distances), six autonomous frames, and four known neighbor-swap frames. Each includes an earlier view after at least 30 mm of measured TCP displacement when available. `blind_manifest.json` gives only the requested code. Simulator truth is held in `sealed_mapping.json` and was not viewed before saving `blind_annotations_partial.json`.

The first **11 of 40** panels have been reviewed at full panel resolution. All 11 were marked uncertain for absolute numbered-card and requested-opening identity. Several show multiple visually similar faces; others show the backside or partial board. Earlier views often look nearly identical, and some early views have no 30 mm history. This is a partial qualitative review, **not a 40-item human accuracy score**. Earlier completed blind review found 4/10 correct absolute occupancy maps, with all five sparse boards unresolved; the +80 mm wide-view diagnostic exposed card faces but gave 0/4 confident requested numbered cards. These results do not establish that learning the identity is impossible. They do make another candidate association fit premature.

## Predeclared opposite-side pilot

Hypothesis: a view from the *opposite* side of the row may expose a board end or slot reference that the +80 mm view missed. Reuse the nonfinal card-1/port-0 reset from `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json`; the reserved final split stays sealed. At a privileged 180 mm plug-tip standoff, rise 40 mm, translate **−80 mm in base y**, hold, return along y, then descend. Use one rootless Gazebo container on GPU 1. The inherited guard limits increments to 2 mm per 50 ms; aborts at wrist-force rise >5 N, absolute force >35 N, stale oracle contacts, or any new forbidden contact pair; and requires measured pose within 5 mm at each hold and return. Preserve failed traces. The ordinary CheatCode continuation and its score are separate from autonomous performance.

**Pilot gates:** (1) safe full return with no guard abort; (2) in at least two unmarked cameras, the board-end reference, requested card number, and requested port side are identifiable without simulator projections. If either fails, do not gather routine opposite-view data or train an identity model from it. If both pass, new nonfinal reset groups are needed before any learned model promotion, with zero accepted wrong-card predictions, ≥9/10 early availability, accepted pose p95 ≤20 mm, and full live inference p95 <300 ms. This view is diagnostic privileged motion only, never an automatic crop or deployable action.

## Commands and artifacts

Preparation: `.pixi/envs/default/bin/python scripts/prepare_sfp_requested_card_review.py --output /home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1`.

The pilot uses [CheatCodeOppositeWideViewDiagnostic.py](../../aic_example_policies/aic_example_policies/ros/CheatCodeOppositeWideViewDiagnostic.py), `scripts/run_clean_transport_eval.sh`, the frozen scene above, and output `/home/nas/chmin/aic_archives/2026-10-06_sfp_opposite_wide_view_v1/card1_port0/`.

Exact launch (stdout/stderr in `/home/nas/chmin/aic_archives/2026-10-06_sfp_opposite_wide_view_v1.launch.log`):

```bash
DOCKER_HOST=unix:///run/user/1008/docker.sock AIC_TRANSPORT_EVAL_GPU=1 AIC_TRANSPORT_EVAL_POLICY_GPU=1 \
 AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.CheatCodeOppositeWideViewDiagnostic \
 AIC_TRANSPORT_EVAL_TEACHER_PREFIX=true AIC_TRANSPORT_EVAL_ORACLE_CONTACTS=true \
 AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06 AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=90 \
 AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC=250 \
 AIC_TRANSPORT_EVAL_SCENE_DIR=/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/scenes/card1_port0 \
 bash scripts/run_clean_transport_eval.sh opposite_wide_card1_port0_v1 \
 outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
 /home/nas/chmin/aic_archives/2026-10-06_sfp_opposite_wide_view_v1/card1_port0
```

Policy subclass SHA256 `dd865b81150538e2277fb974d10010a3aa5176d7b187219ea857a626709c8108`; frozen scene config SHA256 `15ec14cdb2a0b500413a32d055d21537c5dc93cfb1031d5045d992bfc6ffd681`. The launcher's snapshot and input hashes are preserved inside the audit directory.

## Completed pilot and decision

The pilot finished with **no guard abort**. At the opposite hold, measured TCP offset from baseline was +1.05/−77.78/+38.87 mm (base x/y/z); after returning it was +0.23/+0.11/+0.12 mm. Wrist force was 20.83 N at baseline and 20.59 N at the hold; the only oracle contact pair there was cable/enclosure. The privileged teacher subsequently scored tier 3 = 75. This demonstrates a safely executed diagnostic route on **one reused scene**, not policy success or general safety across configurations. See `pilot_summary.json` and `wide_view_events.json` in the pilot directory.

I inspected the unmarked original RGB panel `opposite_unmarked_baseline_vs_view.jpg` before opening any target projection and saved `opposite_blind_annotation.json`. The opposite move **reduced** useful row context: the center camera went from three visible card faces to one; the left and right cameras retained only partial card bodies. No view showed a reliable board-end/numbered-slot reference or unambiguous requested port side in two cameras. The predeclared visual identity gate failed. The opposite-view branch stops at this pilot. No opposite-view training set, association model, automatic handoff, or insertion test is justified by it.

The 40-panel blind set remains available for a second reviewer: only 11 panels were annotated, so no full-set human success rate is claimed. A testable next identity hypothesis would need a view that exposes a *fixed board reference* while preserving the full row, or a known-observation visual fiducial; a lateral wrist translation alone has failed from both sides. Avoid inferring numbered slots just by counting visible cards when slots can be empty.

Separately, transport is still 0/3 safe buffered arrivals on new sparse development starts, even though its matched teachers fully inserted. Its closest lateral offsets in the axial −100 to −70 mm band were 23.26/31.58/14.40 mm, all above the 10 mm requirement, and same-image task-code action diameter was only 0.554 mm median after paired-data training. That gives a concrete transport hypothesis: the actor weakly conditions its early route on the requested card, rather than being blocked at the insertion lip. Do not train another generic BC continuation without an observation-grounded target or a distinct matched diagnostic and stopping rule. [Transport evidence](2026-10-05-transport-route-handoff-continuation.md).

The earlier [spatial transport prerequisite](2026-10-05-spatial-transport-prerequisite.md) explicitly forbids fitting or launching a spatial actor until requested-target identity passes the hard grouped-scene gate. That gate is still failed (older hard replay only 3/10 early selected-target anchors). Therefore I did **not** run a spatial transport fit as part of this continuation. Reusing the three sparse actor routes or their final-scene equivalent as a fresh promotion set would violate that experiment's split/gate. The next transport work is a bounded target-identity repair or an independently declared diagnostic, not routine BC retraining.
