# SFP board-slot reference audit

**Date:** October 4, 2026. **Scope:** saved nonfinal development RGB only; no new rollout or training.

The wide-view audit showed the card row but could not certify its numbered slot origin. This bounded follow-up asks whether the fixed board edges, mounting grooves, and four-centimeter slot pitch let an observer identify **all five physical rail slots and which are occupied**, including empty slots, from original center/right wrist RGB. Card numbering increases along the board's rail direction from slot 0 to slot 4 in the official task-board model. This CAD convention is available at deployment; individual card poses from a reset are not.

## Frozen audit before opening labels

Use the ten already saved nonfinal fresh SFP teacher scenes. At simulated time nearest 11 s, select the center and right RGB frames. Save anonymous image aliases; do not inspect the source-scene mapping or simulator port projections until recording predictions. For each image pair, mark the observed card centers in their spatial order and assign physical slot IDs 0–4 only when the board edge/mounting-slot reference can be seen. Mark **uncertain** if the absolute slot origin, empty slot, or selected port's side cannot be inferred. Do not fill missing slots from the requested task code or from the known number of cards. Compare labels afterward by complete scene group.

This is a feasibility audit, not a model accuracy claim. Evidence for a supervised slot-first crop would be at least **8/10 exact five-bit occupancy maps**, no confident assignment of an occupied slot to a physically empty one, and at least two correctly mapped sparse-card scenes. Otherwise, first improve the visual reference or view selection. Do not enable autonomous handoff, actor or RL from this diagnostic. Final configurations stay sealed.

## Result

I saved anonymous center/right RGB aliases at 11 s and wrote `blind_annotations.json` **before** opening `sealed_mapping.json` or scene configs. The audit was conservative: a card face that was cut by the image edge was recorded as partial, and an apparent wider gap did not by itself establish which absolute physical slot was missing. Posthoc `eval_config.yaml` occupancy is only the scoring key.

| Scene group class | Exact maps | Abstentions | Confidently wrong |
|---|---:|---:|---:|
| Five installed cards (5 scenes) | 4 | 1 | 0 |
| Sparse cards (5 scenes) | 0 | 5 | 0 |
| **Total** | **4/10** | **6/10** | **0** |

The extra complete-card abstention had the row clipped at the selected instant. Sparse layouts included slots `[0,1,2,4]`, `[1,2,4]`, `[0,2,3,4]`, `[0,1,3,4]`, and `[1,2,3,4]`. Existing 2 s and 6 s right-camera frames were inspected **after** opening the mapping for five sparse/partial cases; they sometimes show different card faces but still cut off a board end or fail to show a reliable slot-zero reference. This second temporal review is illustrative and is **not** another blind accuracy score.

The CAD shows fixed rails at 40 mm nominal spacing, but individual NIC card positions and yaw vary. The camera images contain mounting grooves, but their visible segments were insufficient to anchor five numbered slots in the sparse cases. This is a limitation of this viewpoint and review, **not proof that a learned visual model cannot do it**. Simulator projection put each requested opening inside all three camera image bounds near 11 s on the saved teacher poses; projection does not certify visibility, and CheatCode itself positioned the camera with privileged target geometry. These teacher-centered images cannot establish what an autonomous transport actor will see.

The predeclared **8/10 with two sparse scenes** gate fails. Do not train a new slot-first model from this result or promote the wider observation. The next distinct, bounded diagnostic should inspect **actual autonomous transport pauses**: does observation-only camera/robot history reliably place the *requested* card and paired ports near the tool while avoiding neighboring-card swaps? If so, a local requested-port estimator and verifier may be simpler than reconstructing all five absolute slot identities. Score the local-port choice posthoc on complete nonfinal episodes, including wrong-card/empty-slot negatives, and keep an explicit invalid outcome. If autonomous pauses do not reliably enter that neighborhood, repair transport or test a view that exposes both board endpoints before fitting perception again.

## Commands and artifacts

```bash
python scripts/prepare_sfp_blind_slot_audit.py \
  --source /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_board_slot_reference_v1 \
  --sim-seconds 11
```

The archive contains `blind_manifest.json`, symlinked unmodified original RGB, pre-score `blind_annotations.json`, `sealed_mapping.json`, `scored_annotations.json`, and `causal_history_review/` symlinks. The [machine summary](2026-10-04-sfp-board-slot-reference.summary.json) records the gate and denominators. No model checkpoint, GPU training, or new Gazebo container was produced.
