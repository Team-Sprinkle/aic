# SC shadow handoff: one-port and hidden-target stress test

**Date:** October 3, 2026. **Decision:** keep the automatic SC handoff and insertion actor disabled. The four earlier two-port development starts still demonstrate accurate transport-only handoffs, but this continuation found a missed valid one-port arrival. The reserved final scenes stayed sealed. No insertion, actor RL, or SFP stop training ran.

## Frozen-model checks on saved frames

I replayed the unchanged SC verifier on the four accepted development pauses using their recorded camera JPEGs, measured TCP pose, and measured wrench. The JPEGs are a proxy for the native RGB that the current live verifier consumes. All **4/4** controls were accepted. The same four image triplets with the *other* SC port requested were rejected **4/4**, and four early transport triplets were rejected **4/4**. The wrong-request cases had high identity probabilities for both ports and were rejected by the measured TCP-to-requested-standoff check (69–85 mm); therefore this does **not** establish that the identity head independently recognizes a wrong port.

The previously reviewed 20 provisionally hidden requested-port triplets were also replayed. The unchanged full verifier rejected **20/20**: 16 at the locator gate, three at three-view opening agreement, one at TCP posture. The last case was especially instructive: the identity head scored all three views near 1 and the geometry appeared self-consistent, but its TCP was about 78.5 mm from the predicted standoff and 29° from the predicted posture. The review labels are single-reviewer and provisional; no wrench was retained in their derived rows, so a benign 20 N was supplied and any hypothetical acceptance would have been treated as suspicious. All real rejections occurred before the force gate. The camera frames were saved as JPEGs, not as the live native RGB buffers. This is a bounded screen, not a certified occlusion test.

## Native RGB one-port tests

I added a diagnostic-only counterfactual request at the same fresh native RGB inspection. It never affects the real requested-port decision or robot action. The frozen original model ran on the prior four one-port **development** configurations, with two selected for bounded live checks: `development_p0_a` and `development_p1_a`. These complete reset groups are held out from identity-head fitting. The results were:

| Scene | Selected real port | Real live handoff | Absent-port counterfactual | Closest measured TCP to true transport standoff, post-run |
| --- | ---: | --- | --- | ---: |
| `development_p0_a` | 0 | rejected one proposal, then no accepted handoff | rejected | 0.94 mm |
| `development_p1_a` | 1 | accepted at 42.7 s | rejected | 0.42 mm |
| `development_p0_a` extended matched repeat | 0 | no pause proposal | none was inspected | 0.85 mm |

The port-0 proposal at 60.444 s was only about **1.27 mm** from the true selected transport standoff. Its opening was visible in the saved center image, but the frozen spatial identity scores were **0.247 center / 0.676 left / 0.919 right**, failing the fixed all-three ≥0.5 gate. In the extended repeat, transport again physically reached the standoff but the late visual proposal lost recent three-view consistency (about 3.56 px residual) and made no new inspection. The two failures are distinct: **verifier false rejection** and **proposal miss**. They do not indicate failed transport. All three runs had no scored contact and no insertion, as intended. The port-1 accepted native verifier took 146 ms; the rejected port-0 check took 53 ms. These timings still exclude an attached insertion actor and its first controller command.

## Two bounded repairs, both rejected

1. **Near-standoff sampling for the identity head.** I retrained only the spatial head on the existing one-port fitting groups and two-port training groups. The frozen RGB locator/backbone and 0.5 per-view threshold were unchanged. Positive one-port training images with TCP 150–220 mm from the opening received a fixed 4× sampling weight, and all available one-port frames were used. On the four held-out one-port development groups, the all-three rule retained **100/100** projected-in-frame real triplets versus **96/100** before and still accepted **0/109** absent-port triplets. But its per-view absent false positives rose from 9/304 in the original presample to **52/428** in this larger presample; its two-of-three absent acceptance rose from 2/109 to **19/109**. Most decisively, replaying the exact missed port-0 image made the center score *worse*: **0.247 → 0.078**. It retained the earlier four accepted controls and the port-1 live control, and rejected the 20 provisional hidden triplets, but did not repair the actual miss. **Do not deploy this checkpoint.** The selected model was not tuned on the held-out groups.
2. **Larger full-image feature map without new training.** The original head was fed a separate 576×512 RGB backbone pass, while the coarse locator remained 288×256. This rejected the port-0 miss but also rejected **two** of five previously accepted live controls; it retained only **3/5**. It rejected all saved wrong requests and provisional hidden cases, but fails usable-arrival recall and adds a full backbone pass. **Do not deploy this path.** The runtime default remains the original input size and checkpoint.

The SC gate requires *both* no accepted wrong/hidden target and useful recall on correct arrivals. Zero false accepts in these small screens cannot compensate for the port-0 missed handoff. Full pause-to-first-insertion-command p50/p95/p99 was not measured because no insertion actor was attached. Earlier local actor runs already exceeded 300 ms complete inference p95, so sequentially composing the current verifier with that actor is a concrete latency risk. No actor training or full insertion rollout was started from this failed gate.

## Targeted temporal retry and proposal diagnosis

The first port-0 rollout retained camera frames after its rejected inspection. Replaying the unchanged verifier every 200 ms from 59.85 to 61.85 s showed the center identity probability rising from roughly 0.23–0.48 to **0.531** at 61.85 s. At that last frame, the opening geometry and 1.35 mm predicted standoff passed, but the current TCP posture was wrong. This suggested a held-state retry might give the existing board-angle correction enough evidence. Separately, the extended port-0 trace contained three consecutive measured near-standoff decisions by 47.1 s (predicted distance 1.6 mm, orientation remainder 0.16°), but the stop logic withheld inspection because its last good three-view target was 1.5 s old rather than ≤1 s. A fresh verifier can safely reject a stale target, so I tested an **opt-in shadow-only** pause at ≤3 mm and ≤1°, with the original strict verifier unchanged.

Two additional rootless runs on the *same* port-0 development reset tested these hypotheses; they are diagnostics, not independent generalization evidence. The first enabled held-state identity retries alone. It had **zero pause proposals** despite reaching **0.615 mm** from the true standoff. The second enabled the stale-target pause as well. It made **three** inspections and **24** additional fresh-frame identity checks, but all three inspections were rejected for incomplete all-view identity; its TCP reached **0.542 mm** from the true standoff. Each of its three absent-port counterfactual checks was rejected. The later image scores did not reliably repeat the one favorable saved frame. The retry and stale-pause switches remain **off by default** and are not a repair. This outcome narrows the failure to unstable selected-port evidence at a genuinely reached transport pose, with a secondary proposal-staleness failure. It does not justify lowering the all-camera identity gate.

The two retry runs and their scored streams are indexed by [`native_oneport_temporal_retry_recall_v1.json`](/home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v2/native_oneport_temporal_retry_recall_v1.json); the saved 200 ms replay is [`p0_identity_temporal_window_v1.json`](/home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v2/p0_identity_temporal_window_v1.json). The opt-in live command added `AIC_SC_SHADOW_IDENTITY_RETRY=1 AIC_SC_SHADOW_ALLOW_STALE_PAUSE=1` to the same `run_sc_shadow_handoff_pilot.sh` invocation and retained `AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST=1`. Neither changed model weights, selected port, or scene.

## SFP arrival label audit

The frozen temporal transport cache has **33,164 SFP training rows and 4,520 SFP validation rows**, grouped into **236/32 episodes**. `scripts/cache_temporal_transport_features.py` assigns every SFP row `stage=6` and `handoff=-1`; only SC rows receive a local-start handoff label. `scripts/train_frozen_transport_handoff.py` samples positives and negatives from `family=='sc'` only. This exactly explains why the earlier opt-in SFP head stopped 162 mm early: it was asked to transfer an SC arrival classifier to SFP without SFP arrival supervision. The audited SFP data can support a *new* label builder, but a near-port positive should be defined from the selected SFP opening and a safe standoff in the scored expert stream, with full-episode split and explicit early-route/wrong-port negatives. No SFP head was trained in this continuation.

## Reproduction and preserved artifacts

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts .pixi/envs/default/bin/python \
  scripts/replay_sc_shadow_handoff_negatives.py \
  /home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v1/direct_rgb_replay_llb_03{00,03,04,09} \
  --hidden-review /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_hidden20_consistency_v1.json \
  --hidden-projections /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/hidden_state_replay/*_projection_all \
  --output /var/tmp/chmin_aic_20261003_shadow_handoff/saved_jpeg_hard_negative_hidden_replay_v2.json --device cuda:0

AIC_SC_SHADOW_DIRECT_NATIVE_RGB=1 AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST=1 \
  AIC_SC_SHADOW_SCENE_ROOT=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1 \
  bash scripts/run_sc_shadow_handoff_pilot.sh \
  /var/tmp/chmin_aic_20261003_shadow_handoff/native_absent_development_p0_a development_p0_a
# Replace p0_a with p1_a for the matched second group.
# The port-0 repeat additionally set MAX_SIMULATION_SEC=100 and MAX_RUNTIME_SEC=260.

PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/score_sc_shadow_transport_recall.py \
  /var/tmp/chmin_aic_20261003_shadow_handoff/native_absent_development_p0_a \
  /var/tmp/chmin_aic_20261003_shadow_handoff/native_absent_development_p1_a \
  /var/tmp/chmin_aic_20261003_shadow_handoff/native_absent_development_p0_a_extended \
  --output /var/tmp/chmin_aic_20261003_shadow_handoff/native_oneport_transport_recall_v1.json
```

The exact near-view fitting command and group paths are recoverable from [`sc_identity_near_boost4_v1/summary.json`](/home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v2/sc_identity_near_boost4_v1/summary.json). Each live audit contains `launcher_snapshot.sh`, `eval_manifest.json`, `inputs_sha256.txt`, native camera capture JPEGs, runtime trace, scored bag, and scored stream. The [NAS archive](/home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v2/) and [compact machine decision](2026-10-03-sc-shadow-oneport-stress.summary.json) preserve the original, boosted, and larger-image comparisons. Rootless Gazebo used GPU 1 and policy GPU 2; the offline probe used GPU 1. At most two project GPUs ran concurrently. Other active containers were left untouched.

## Next bounded experiment

Repair **port-0 late visual evidence and true visible-target recall as a pair**, using only existing training groups first. A useful next representation is a request-conditioned, observation-selected original-RGB port-face crop combined with the full image and measured TCP; train on labeled clear and hard-hidden views from complete fitting reset groups. The other port and physically absent requests must remain explicit negatives. Freeze a new independent one-port development set *before* fitting; then rerun the known port-0 miss, port-1 control, four two-port controls, provisional hidden set, and a small new native-RGB shadow screen. Require no known false accepted handoffs and at least 95% reviewed clear-arrival recall. Once that passes, attach the frozen actor for a nonmoving latency benchmark, measure complete ROS-to-first-insertion-command p50/p95/p99 against 300 ms inference time, and only then allow a moving insertion rollout. If the representation fails the grouped gate, keep the supervised insertion branch paused and treat SFP arrival labeling as a separate bounded task. Do not use final scenes.
