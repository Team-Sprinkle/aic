# SFP middle-card identity: original-RGB review and bounded structured locator

**October 4, 2026. Decision: do not promote.** The previous task-conditioned coarse anchor had excellent error on its accepted outer-card observations but no acquisition for cards 1–3. I reviewed the original RGB for those six failed development episodes, then trained exactly one bounded all-slot internal detector to test whether explicit neighboring-card supervision repairs the association. The frozen ten development resets and reserved final scenes were not used for fitting.

## What the camera review supports

The [six review sheets](/home/nas/chmin/aic_archives/2026-10-04_sfp_middle_card_visibility_v2) use 256×256 crops from the original 1152×1024 RGB, at first/middle/last sampled transport times for all three wrist cameras. Green crosses are **true projected requested-port locations for human review only**; red rings mark other present port projections. They never selected a model crop or entered inference. Five of six first observations show the marked port face apparently unobscured in all three views. The card-3/port-0 start is offscreen in all three, then appears later. By the middle observation all six show an apparently exposed marked face in all three. This is a single-reviewer exposure check *given the marker*, not an independently certified ability to identify the correct card index without it. It argues that pure occlusion is not the sole reason the prior locator missed middle cards. [Review manifest and limits](/home/nas/chmin/aic_archives/2026-10-04_sfp_middle_card_visibility_v2/review.json).

The first render script attempt stopped on that truly offscreen start after writing some sheets. It is preserved as `2026-10-04_sfp_middle_card_visibility_v1`; the completed corrected review is `v2`. The correction renders a labeled full-view thumbnail when the requested opening is offscreen.

## Matched supervised repair

The new internal detector starts from the frozen October 2 all-port ResNet18 locator and processes original JPEGs resized to **576×512**. Its training targets include **every present card's two openings**, rather than a randomly chosen single request per image. Each card/port channel has a spatial heatmap target, a coordinate target, and an in-frame flag; a small ranking loss penalizes a channel that puts its point closer to another opening than its own. At runtime, public task code selects **one requested output channel**. Input is RGB and camera identity, then measured TCP and fixed camera calibration triangulate the selected point. No true port transform, true selected crop, CheatCode action, or target projection enters inference.

The same **50 complete training resets / ten complete development resets**, 692/136 sampled three-camera observations, and fixed acceptance rule were used as the preceding probe: all three per-camera in-frame probabilities ≥0.5 and triangulation reprojection error ≤5 px. A single GPU (GPU 2), batch six, and **2,000-update cap** were used. The best recorded checkpoint is update 1,400, selected using zero wrong accepted identities, 3D accepted p95, and complete-episode acquisition within three seconds. Selection on these ten resets means the result is development evidence, not an untouched test.

| Frozen development comparison | Single selected-port head, October 3 | Frozen old all-port model | New structured all-slot head |
| --- | ---: | ---: | ---: |
| Accepted observations / 136 | 51 | 53 | **66** |
| Accepted wrong requested ports | 0 | 0 | **0** at selected checkpoint |
| Accepted 3D position p95 | 2.65 mm | 4.69 mm | **2.14 mm** |
| Complete episodes with any anchor / 10 | 4 | 8 | **8** |
| Complete episodes with anchor by 3 s / 10 | 4 | 7 | **7** |

The planned operational gate was **≥9/10 early episode acquisitions, including both requests for each middle card, zero wrong accepted identities, and accepted 3D p95 ≤20 mm**. The new checkpoint fails coverage. A different intermediate checkpoint at update 800 accepted one wrong identity; it was not selected. Thus the improvement in accepted count is real on the frozen development set, but it does **not** resolve transport coverage. No actor continuation, new live rollout, PoseInsert, RL, or world-model training followed.

The two never-acquired episodes show a specific error. For **card 2/port 0**, the selected channel's nearest posthoc true opening was card 1/port 0 in **11/13 center** and **9/13 right** views, while the left view was usually near the requested card. For **card 3/port 0**, center and left often chose card 4/port 0, while right chose card 3/port 0 in **11/14** views. This audit uses true geometry **after** RGB prediction only to name the wrong candidate. The views disagree by tens to hundreds of pixels, so the fixed three-view rule rejects them rather than silently inserting at the wrong port. [Exact nearest-candidate counts](/home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/middle_card_failure_audit.json).

## Next targeted decision

Do not repeat another generic heatmap fit on these same ten development resets. The next test should explicitly establish card index from **board/rail slot geometry** or obtain a safe additional view when the requested slot is offscreen/ambiguous. A coarse board pose can identify the requested 40 mm rail band, while a port-face detector finds the card's shifted position within that band. Slot indices must remain explicit when intermediate cards are absent; sorting only the visible cards is unsafe. Train geometry may supervise board/slot landmarks, but runtime crop and candidate selection must use RGB, task code, measured TCP and fixed calibration only. On fresh non-final complete resets, require the same zero-wrong rule, ≥9/10 early acquisitions (with an explicit separate acquisition clock after a safe view-seeking action if the port starts offscreen), accepted 3D p95 ≤20 mm, and complete live p95 <300 ms before any actor continuation. If the target cannot be made observable safely or the same wrong-card swap persists, stop this coarse-anchor branch and revise transport route representation instead of using a privileged shortcut.

Rootless Docker remained at **200/200 session keys** with only the old active Isaac container. No new container was launched or active process interrupted. Offline PyTorch/Pixi used GPU 2 and finished; GPU 0's old process remains untouched.

## Commands and preserved artifacts

```bash
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
.pixi/envs/default/bin/python scripts/render_sfp_middle_card_visibility_review.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_middle_card_visibility_v2

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
.pixi/envs/default/bin/python scripts/train_sfp_structured_slot_locator.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --init outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1 \
  --updates 2000 --minutes 120
```

[Training summary](/home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/summary.json), [best checkpoint](/home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt), `history.json`, and `predictions_step*.jsonl` in that same directory preserve every checkpoint comparison. [Tracked machine decision](2026-10-04-sfp-middle-card-identity-repair.summary.json).
