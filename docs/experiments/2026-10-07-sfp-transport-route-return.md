# SFP transport return: requested route choice

## Frozen diagnosis and scope

The matched October 5 development comparison remains frozen. Three candidate
actors reached the requested 10 mm buffered transport region **0/3**; all
matched CheatCode controls inserted. The candidate uses only public images,
ordinary state and request code. The reserved final scenes remain sealed.

`scripts/audit_sfp_paired_first_divergence.py` compares all three candidate
traces with their same-reset scored teachers. Raw equal-clock TCP error passes
10 mm at 3.8, 4.4 and 4.05 simulation seconds, but the teacher starts moving
earlier. The actor starts moving >1 mm at 4.25, 3.75 and 3.6 seconds while the
teachers do so at 2.98, 3.098 and 3.038 seconds. Thus equal-clock error alone
overstates early route error. A progress-independent nearest-teacher-route
check first exceeds 10 mm at 4.9, 6.55 and 10.35 seconds. The actor's images
are timestamped at the proposal time in all three reviewed traces. Measured
motion generally follows commanded direction (when both commanded and measured
motion are material); the remaining error is primarily route/progress selection,
with a startup delay as a separate contributor. The original trace and
commands are preserved at
`outputs/experiments/2026-10-07_sfp_transport_return/first_divergence.json`.

The prior actor's task-code action diameter is 0.554 mm median, compared with
12.44 mm median paired teacher branching. This is the targeted hypothesis.
The previous actor **already appended the task code before its temporal
transformer**, so merely moving the code earlier would not test a new idea.
This experiment instead modulates every camera feature token by the requested
card/port code and adds a discrete route-choice prediction. The supervised
choice is the largest component/sign of measured TCP displacement over the
following second, plus a still class below 4 mm. It is an offline label only;
the deployable actor sees the learned class probabilities and still outputs
the complete TCP-body action chunk. It receives no true port geometry.

## Declared comparison and stopping rule

- Use the existing `temporal_cache_v4` and episode/scene split, balanced
  request/paired-scene sampler, 3,000-update cap and the corrected October 5
  action-only checkpoint as initialization. Train on one GPU. This is one
  bounded architecture test, not a sweep.
- Offline: compare held-out SFP first-action translation p95 against **2.153
  mm**, held-out route-choice accuracy against the majority-class baseline,
  and same-observation early paired-request action diameter against **0.554
  mm** (teacher reference **12.44 mm**). Require no >10% p95 regression and
  a meaningful increase in request sensitivity toward the teacher branch,
  with route-choice accuracy above the majority baseline. These are necessary
  diagnostic gates, not claims of autonomous performance.
- Only after passing offline checks, use three **new nonfinal** SFP starts and
  require at least **2/3** safe requested buffered visits within 10 mm, no
  extra scored contacts and complete inference p95 below 300 ms. If passed,
  expand to ten grouped development starts with target 9/10.
- Stop this route-representation branch if the offline gate fails or if live
  safe arrival remains absent. Diagnose the first live divergence before any
  further fit. Keep precise handoff, insertion, RL and final-scene evaluation
  gated.

## Commands and results

Diagnosis:

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_paired_first_divergence.py \
  --summary /home/nas/chmin/aic_archives/2026-10-05_sfp_paired_development_actor_v1/summary.json \
  --output outputs/experiments/2026-10-07_sfp_transport_return/first_divergence.json
```

Training and results follow below as they finish.

## Completed bounded fit and decision

The one-GPU fit used:

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_v4 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --output outputs/experiments/2026-10-07_sfp_transport_return/request_route_film_v1 \
  --full-state-skip --long-history --no-route-heads \
  --balance-sfp-targets --paired-sfp-mass .5 \
  --init-temporal outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v2/best.pt \
  --task-route-film --sfp-route-stage-loss 0.2 --updates 3000 --budget-minutes 60
```

The fit stopped at update 2,500 because validation plateaued; its selected
checkpoint was update 1,250. On the fixed grouped validation sample, the
discrete future-direction accuracy was **92.5%**, above its **50%** majority
baseline. Same-image ten-code action diameter rose from **0.554 to 3.394 mm**
median at 0.4/0.6/1.0 s, still well below the **12.44 mm** median matched
teacher branch difference. Held-out first-action translation p95 worsened from
**2.153 to 2.811 mm** (+30.6%), exceeding the predeclared 10% regression
limit. The model learned a label and became more code-sensitive, but the full
action got worse. These offline results do not establish a correct live route.
The live three-start gate was therefore **not opened**. No new Gazebo scene,
perception handoff, insertion, or RL test used this checkpoint.

Machine results and immutable checkpoint:

- `outputs/experiments/2026-10-07_sfp_transport_return/request_route_film_v1/{manifest.json,best.pt,history.json,summary.json}`
- `outputs/experiments/2026-10-07_sfp_transport_return/request_route_film_v1_task_sensitivity.json`
- `outputs/experiments/2026-10-07_sfp_transport_return/first_divergence.json`

The frozen trace comparison also shows a substantial delay before actor
motion: >1 mm measured displacement begins roughly 0.56–1.27 s after the
matched teacher. When aligned by geometric route instead of simulation time,
the three candidate trajectories first pass 10 mm from the teacher's route at
4.9, 6.55 and 10.35 s. Commanded versus measured motion direction has median
cosine **0.920, 0.929 and 0.907** when both motions are material; image
timestamps match each proposal. These checks argue against a simple camera
timestamp or wrong-coordinate execution bug. They do not rule out controller
speed/latency contributing to the closed-loop drift.

**Decision:** stop this FiLM plus discrete-route branch under its declared
gate. The remaining targeted transport hypothesis is that frozen globally
pooled camera vectors do not retain enough requested-card geometry for a
strong code-specific route. A future single bounded test should compare
spatial camera tokens or an observation-selected original-image card-row crop
on the *same paired scene split and full-action target*, first requiring a
clearer early requested-route difference without action-error regression.
Any live test remains conditional on that offline gate, and selected-port
identity must remain a separate prerequisite for handoff. Do not infer that
the 92.5% auxiliary direction classification is usable by itself: most
directions are shared among requests and it does not locate the requested port.
