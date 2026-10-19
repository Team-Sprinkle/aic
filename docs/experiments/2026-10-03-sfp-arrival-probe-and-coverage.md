# SFP arrival probe and task-coverage audit

**Date:** October 3, 2026. **Status:** diagnostic failed promotion; automatic SFP stop remains disabled. Reserved final scenes sealed. No insertion actor or RL was run.

## Question and label

The existing temporal transport cache assigned `handoff=-1` to every SFP row, while the SC-only arrival head had once caused an SFP false pause 162 mm from the requested opening. I tested whether the existing frozen three-camera ImageNet features can learn a separate SFP stop **proposal**. A teacher row was labeled positive when its *measured TCP pose in the selected port frame* had axial coordinate between -100 and -70 mm and lateral distance at most 10 mm. This is a buffered local-actor region, not a certified safe insertion pose. Oracle geometry produced labels and posthoc error only; inference used cached RGB features, measured 32D robot state, and public task code. The probe did not select a port or verify its identity.

The source has 236 training and 32 validation complete episodes with disjoint scene hashes. The cached features are one row every 200 ms, with 3 separate 512D camera vectors. The 10D task code explicitly identifies family, target port, and target card. **Coverage is severely skewed:** 227/236 training and 31/32 validation episodes request card 0, port 1. The remaining nine training episodes cover only five other card/port combinations, and validation contains only one different combination (card 4, port 0). These 268 episodes cannot establish multi-card/multi-port SFP stopping performance even if the aggregate metric is good.

**Correction during the October 3 continuation:** I initially looked for a per-attempt `engine_config.yaml` and wrongly concluded the older 180 selected agent/VLM nominal SFP episodes lacked target identity. Their numbered `trials/trial_XXXXXX.yaml` files in fact match every selected replay attempt; all 180 requested card/port pairs are recoverable, with 8–30 scored successes per pair and scores ≥90. The LeRobot task text is generic, but the trial YAML is authoritative. A separate frozen-image arrival diagnostic is now being built from these successes. Earlier audits found invalid zero Cartesian action arrays during motion, so those actions remain excluded from BC. [Corrected provenance audit](../../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/older_agent_provenance_audit_v2.json).

## Bounded frozen-feature diagnostic

I trained three 128-unit heads for 400 balanced minibatch updates on the same frozen cache: measured-state-plus-task only, RGB features only, and RGB plus state/task. A positive proposal required probability ≥0.95 at three consecutive 200 ms rows. Checkpoints were chosen by validation episode event counts, so this validation is **exploratory and optimistic**, not a new sealed test.

| Head | Correct / 32 validation episodes | Early | Missed |
| --- | ---: | ---: | ---: |
| State and task only | 10 | 0 | 22 |
| Frozen RGB only | 28 | 2 | 2 |
| RGB, state and task | 23 | 2 | 7 |

The two early RGB-only proposals were 0.6 and 0.4 s before the labeled region. One had 17.8 mm lateral offset (outside the 10 mm training region); the other was 3.4 mm farther back axially. Two missed episodes never sustained the required threshold. Relaxing to two of three high-probability frames gave 26 correct, 4 early, 2 missed. Three-frame probability averaging gave 28 correct, 2 early, 2 missed. Thus a simple temporal rule did not remove the false proposals or the misses. No identity verifier, contact gate, or live Gazebo stop was attached. The current data also do not show that this buffered TCP region lies within the safe correction envelope of a retrained PoseInsert actor.

## Root cause and next bounded repair

The current SFP stop failure has at least two independently measured causes: the cache had **no SFP stop labels**, and the verified set used here is overwhelmingly one requested card/port combination. The RGB-only probe also makes early proposals on the observed TCP trace, so it cannot be used as a safety check. The state-only failure and RGB+state regression show that adding more inputs alone does not fix it. The existing SFP selected-port perception has known wrong-port locks, so even a perfect arrival proposal would not make an automatic handoff safe.

The next experiment should **use the recovered 180 successful agent/VLM SFP approach recordings** across target cards 0–4 and ports 0–1, with independent complete scene/reset groups, then train a task-conditioned arrival proposal and a selected-port identity verifier as separate components. Keep the invalid action arrays out of BC. Stop the proposal branch if any new-development wrong-port/too-early proposal is accepted by the full visual verifier, or if clear requested-port arrival recall stays below 95%. Do not attach PoseInsert until the SC and SFP handoff gates both pass, full attached-actor latency is below 300 ms p95, and its correction envelope is shown on actor-generated starts.

## Artifacts

- [Training script](../../scripts/train_sfp_arrival_probe.py).
- [Checkpoint and episode-level metrics](../../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/summary.json).
- [Source hashes and exact task distribution](../../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/audit.json).
- [Corrected older agent/VLM selected-replay provenance audit](../../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/older_agent_provenance_audit_v2.json). The earlier `older_agent_provenance_audit.json` used the wrong config lookup and is superseded.

```bash
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/train_sfp_arrival_probe.py \
  --cache outputs/experiments/2026-10-01_transport_five_way/imagenet_features \
  --source /var/tmp/chmin_aic_20260918_act/sfp_verified_268_port_tcp_waypoints_20260924.jsonl \
  --output outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1 \
  --updates 400 --device cuda:1
```
