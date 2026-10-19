# Transport feedback and inference-cadence diagnosis, October 12

## Why this audit ran

The [teacher-generator comparison](2026-10-12-teacher-distribution-transport-diagnosis.md) found 6/6 successful teachers but 0/6 autonomous insertions. Offline first-action imitation was good, yet live actions weakened on the second decision. This bounded follow-up used saved, nonfinal SFP and SC observations, one GPU, and the frozen route-diffusion checkpoint. It then tested two short training ablations and one faster inference setting. No final scene, insertion handoff, or RL was used.

## Frozen input replay: model execution is consistent

[`audit_route_live_input_parity.py`](../../scripts/audit_route_live_input_parity.py) rebuilt the three-camera feature, normalized measured state, task code, causal history, previous action, motion, camera clock, and deterministic diffusion sample from saved actor observations. Its replay of the recorded policy command differed by at most **0.30 mm** on the first 12 SFP val04 proposals, **0.18 mm** on SFP val12, **0.12 mm** on five-card SC LLB0074, and **0.00 mm** on SC LLB0005 (first-target translation). Saved camera frames are JPEG-quality-86 captures of the same stamped observations, so this is close replay, not an exact raw-byte equality test. The small residual makes a frame conversion or sampler implementation error unlikely to explain the 6–19 mm teacher/live action gaps.

At the second decision, the teacher and live visual features still had cosine similarity 0.994/0.982 for the two SFP scenes and 0.977/0.980 for the two SC scenes. The first teacher/live TCP position differences were 1.5, 4.1, 15.6, and 11.5 mm respectively. SC's teacher cache starts after its route planner has begun, so its elapsed-time alignment is approximate; use the swap as a diagnostic of model dependence, not an exact counterfactual episode.

| Scene; second decision | Live first-target gap from time-matched teacher prediction | Swap only previous action from teacher | Swap only visual feature from teacher |
| --- | ---: | ---: | ---: |
| SFP val04 | 6.1 mm | **2.5 mm** | 5.8 mm |
| SFP val12 | 10.0 mm | **1.3 mm** | 10.0 mm |
| SC LLB0074 | 14.0 mm | **2.3 mm** | 14.4 mm |
| SC LLB0005 | 19.0 mm | **3.0 mm** | 19.2 mm |

The same pattern persisted for the next two decisions. Swapping measured state/task also helped, but less than swapping the prior command; motion and camera-clock swaps barely changed predictions. The actor was trained with the previous **teacher** target in 90% of training contexts (`prior_dropout=.1`) and the previous **actor** target at deployment. When its first proposal is small or a control tick is missed, this creates a feedback shortcut: it uses its own small command as evidence that the next command should stay small. This is an inference from the controlled swaps and timing, not a proof that every later route error has this cause.

## Failed zero-prior repair

I fine-tuned the **same architecture** for 2,000 updates on the same 50,129 training rows, initialized from the selected checkpoint, with `prior_dropout=1.0`. This forced the model to use visual/state/history context instead of the previous teacher chunk. Its [checkpoint](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/zero_prior_ft_v1/best.pt) was selected by held-out no-prior action error at update 2,000; this was an update cap, not a plateau.

On identical held-out rows evaluated with previous actions hidden, early SFP first-action p95 improved **12.34→4.38 mm** and SC p95 **24.12→4.48 mm** compared with the original checkpoint with its previous input zeroed. Yet on the four frozen actor-visited traces, its first few proposals were generally **no closer** to the teacher than the original actor's. For example, SFP val12's second-decision gap grew 10.0→11.3 mm; SC LLB0074 improved only 14.0→12.6 mm, and later gaps grew. It failed the predeclared frozen-frame gate, so I did **not** run it autonomously or promote it. Hiding the prior is insufficient: the model needs a reliable indication of route progress when the robot has barely moved or an inference tick is delayed.

## Startup cadence and bounded four-step sampler

The original first inference cost **772 ms** on SFP val04 and **922 ms** on SC LLB0074, with first-to-second proposal gaps of 0.45 and 0.75 simulation seconds. Subsequent proposals usually came every 0.2–0.25 seconds. The teacher cache has roughly 0.2-second spacing. The slow first call skipped early route decisions and left the prior-action feedback behind the teacher sequence.

An opt-in model warmup before the first observation reduced the five-card SC first call to **303 ms** and its next-decision gap to **0.2 s**, but SFP startup varied and 12-step complete-inference p95 on shared GPU 0 was **486 ms** SFP / **393 ms** SC. Warmup therefore did not pass the latency or transport gate. Its first observation also sometimes arrived later, so do not treat the route-offset differences as a clean causal estimate of warmup's benefit.

The original checkpoint's held-out first-action p95 with **four** diffusion steps was 1.94 mm early SFP and 2.45 mm SC, matching or slightly improving the twelve-step offline result. A bounded live rerun used the *same checkpoint*, true SFP JPEG-86 input forwarding, warmup, and four sampling steps on the same nonfinal SFP val04 and five-card SC LLB0074 configs:

| Run | Closest lateral offset in axial band | Safe buffer | Scored contact | Complete inference p95 | Full insertion |
| --- | ---: | --- | --- | ---: | --- |
| SFP, 12 steps, true codec + warmup | 32.65 mm | No | No | 486 ms | No |
| SFP, **4 steps**, same settings | **5.61 mm** | **Yes** | **Tool/card 2** | **241 ms** | No |
| SC five cards, 12 steps + warmup | 263.53 mm | No | No | 393 ms | No |
| SC five cards, **4 steps**, same settings | 246.64 mm | No | No | **222 ms** | No |

The four-step sampler improves cadence and lets one SFP run reach the geometric buffer, but that run hit the card and SC still followed a wrong route. It does **not** pass safe transport across task families. All latency numbers above colocate Gazebo and policy on GPU 0 because other users occupied GPUs 3/4; they are useful for the matched diagnostic but are not a clean deployment latency certificate. The first SFP four-step call itself took 303 ms.

## Critical launcher provenance correction

The original `AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86` ablations were invalid: `run_clean_transport_eval.sh` wrote the variable into `eval_manifest.json` but omitted it from the Docker `-e` arguments. Those old `*codec86*` artifacts are **unchanged-policy repeats**, not a preprocessing comparison. The launcher now forwards SFP codec, zero-prior, prewarm, and sample-step settings. I checked the live container environment with `docker inspect`. The true SFP codec-only val04 rerun missed at **77.65 mm**; codec plus warmup missed at **32.65 mm**. These variations do not establish a reliable codec benefit. The original misleading summary is preserved in `teacher_distribution_decision_pre_env_audit.json`; the current machine summary marks it invalid.

## Failed delayed-prior repair

I also fine-tuned the original checkpoint for a separate 2,000-update bounded pilot using the same 50,129 training rows, but randomly replacing 70% of history previous-action inputs with teacher actions one or two decisions older. This directly simulates a missed decision while preserving the prior command signal. The [checkpoint](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/lagged_prior_ft_v1/best.pt) was selected at update 1,000 by held-out one-decision-late validation error; training finished at its 2,000-update cap. The selected checkpoint improved lag-one first-action p95 from **3.12→2.27 mm** early SFP and **4.38→3.19 mm** SC on identical held-out teacher rows with four denoising steps. On ordinary teacher histories, error worsened **1.94→2.34 mm** early SFP and **2.45→3.48 mm** SC.

The decisive gate was the first five diverging decisions on each of four frozen actor traces, using identical saved frames and four-step inference. Median first-target gaps to the **recorded teacher target** changed: SFP val04 **8.8→7.8 mm**, SFP val12 **12.2→11.1 mm**, SC LLB0074 **16.4→13.8 mm**, and SC LLB0005 **16.2→15.0 mm**. All four improved, but none met the predefined ≥50% gap-reduction gate. The model therefore was **not** run autonomously or promoted. Delaying prior targets alone is too weak a repair.

## Decision and next targeted repair

No transport checkpoint or sampler setting is promoted; no automatic handoff, insertion actor, or RL run follows. Four-step sampling is promising for latency and one SFP route, but must pass **contact-free** buffer arrival on multiple new development starts in both families. The stronger remaining model problem is dependence on teacher previous commands under lag or stalled motion. Zeroing or delaying those inputs helped offline metrics but did not resolve the frozen actor-state gap. A next model experiment needs **explicit planned route progress or causal teacher continuations from actor-visited states** rather than another routine prior-action augmentation. Audit the existing four frozen actor traces and takeover records first; previous takeovers cannot simply be used as labels because some motion begins after a substantial state change and one officially successful continuation exceeded the cable/card point-force gate. Fit at most one matched actor if the causal, safe label gate passes. Its stop rule is a ≥50% reduction of the early frozen actor-frame action gap in **all four** checked scenes, while ordinary grouped teacher-row p95 rises by no more than 1 mm in either family; only then run new autonomous development scenes. SC rotated-board LLB coverage remains a separate data gap. Keep the final split sealed. [Machine-readable decision](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/feedback_cadence_decision.json).

## Commands and artifacts

```bash
# Frozen input replays and one-field swaps (repeat for the other three scenes).
CUDA_VISIBLE_DEVICES=0 .pixi/envs/default/bin/python scripts/audit_route_live_input_parity.py \
  --audit /home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_sfp_codec86_v1/rollouts/sfp_trial1_validation_04 \
  --episode 4300004 --family sfp --max-proposals 8 --ablation-decisions 5 \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/input_ablation_sfp_val04.json
# Zero-prior fit and matched held-out comparison are recorded in the output manifest.
AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86 AIC_ROUTE_DIFFUSION_PREWARM=1 \
AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=4 \
  .pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_teacher_route_sample4_v1 --gpu 0 \
  --only sfp_trial1_validation_04 --only llb_0074 --max-new 2
```

Local outputs under `outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/`: `input_ablation_{sfp_val04,sfp_val12,sc_llb0074,sc_llb0005}.json`, `zero_prior_ft_v1/`, `zero_prior_matched_offline.json`, `zero_prior_frozen_*.json`, `lagged_prior_ft_v1/`, `lagged_prior_matched_{lag1,clean}.json`, `lagged_prior_frozen_*.json`, `base_sample4_frozen_*.json`, `sample{4,6}_offline.json`, `true_codec_no_prewarm_sfp_pilot.json`, `true_codec_prewarm_pilot_analysis.json`, `sample4_teacher_pilot_analysis.json`. NAS roots preserve complete scene configs, Docker launch logs, scoring bags, camera frames, and runtime traces for the true-codec and four-step runs.
