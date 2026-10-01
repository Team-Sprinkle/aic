# What the latest transport BC experiment actually tested

## Did the actor get near the port?

**No, the newest actor did not reach a usable insertion position in either tested SC-to-SC scene.** Its closest measured TCP distance to the *planned transport handoff* was 91 mm on the two-card, port-0 start and 348 mm on the three-card, port-1 start. At the end, those handoff distances were 103 and 603 mm. The official scorer separately reported the plug about 150 and 560 mm from its selected port, with no full insertion. The handoff and plug-to-port distances use different reference points; they should not be equated. The earlier masked actor came closer, especially on the port-1 start (minimum handoff error 23 mm), but still failed the handoff envelope and did not insert.

## How many episodes were used?

The newest BC run used **236 SFP-to-NIC training episodes** (132,303 recorded training rows) and **two SC-to-SC training episodes** (1,751 transport rows). The two SC episodes were full scored successes from the *same five-card physical world*: one selected SC port 0, the other port 1. They are only two independent route examples despite the many video/action rows. A separate 32 SFP episodes and two SC route-only episodes supplied grouped validation; the three-card SC validation teacher itself did not pass its endpoint gate and was retained for diagnosis only. The run sampled SFP and SC with equal total probability, then sampled the two SC target ports equally.

The 268 verified SFP source episodes are a mixture by provenance. At least 140 are explicitly from original accepted CheatCode datasets (130 mixed-card and ten one-card). The other 128 came from later nominal, corrective, student or multicard collections. These were verified sources, but we should audit provenance and scored success again when making the next clean dataset. The latest run did **not** use only original CheatCode SFP demonstrations.

## Was it initialized from the old SC policy?

**Yes. This is an important flaw in interpreting the latest comparison.** The launch script passed `shared_body_target_bc_batch3_20260925/best.pt` as a warm start. The training code loaded all compatible weights, **including the action head**, and then fine-tuned for 1,500 updates. It removed only the old elapsed-time module and froze the old auxiliary pose head. The warm-start checkpoint lists SFP plus many older SC canonical, supplemental, diversity, five-card-lift and teacher-correction sources. They predate the new route admission and form a mixed route set. We have not counted which of those old SC episodes went over cards, so I cannot say every old trajectory did; the checkpoint certainly was **not trained only on the desired new route**.

Consequently, the failed 1,500-update run is **not a fair test of whether BC can learn the new SC route from clean initialization**. With only two new SC route episodes, it may have retained behavior learned from the older SC mixture. That is a plausible explanation, not a proven single cause of failure.

## Name and use of the desired route

Let us call it the **Low Lateral Bypass (LLB)** route:

```text
back → left while descending below card height → forward beside the row
     → right → tilt → back → descend and approach the opening
```

The left/right labels describe our current scene orientation; the defining properties are an outside-row lateral corridor, a sufficiently low crossing, and tilting **before** the final return. Fully inserted, scored LLB episodes are the preferred SC BC targets. A valid transport prefix can be kept separately when its insertion attempt fails, with failed insertion actions excluded.

Your proposed family-specific expert data policy makes sense: use verified successful SFP-to-NIC CheatCode trajectories and verified successful SC-to-SC LLB trajectories. The actor can still be **one shared transport model** with the task code telling it which family and target to serve. It should learn both route styles from the appropriate data. We should cover both SC ports and vary card counts, card/port positions and reset starts; the current two SC episodes in one world are too narrow to establish this.

## What was tested, and what should be tested next?

The newest model received offline validation rows from **both** families, but its **autonomous Gazebo rollouts were only two SC-to-SC starts**. We therefore do not know whether the latest model preserved or regressed autonomous SFP-to-NIC performance.

Before any new training, I would prepare an exact provenance/admission manifest for both families, separate original CheatCode SFP from later correction data, and inventory how many successful LLB SC episodes cover each card count, selected port and reset layout. Then initialize the policy/action modules without the old mixed-SC checkpoint. An ImageNet-initialized visual backbone with a newly initialized policy and action head is a clean reference; a documented SFP-only checkpoint is another possible warm start. Train the same shared transport actor on the admitted SFP and LLB SC data, hold out entire reset/layout groups, and run matched full-start **SFP and SC** development evaluations. Compare any old-checkpoint warm start only as an explicit control. Keep the local port-frame actor and perception/handoff evaluation separate, and retain the 300 ms latency gate. If clean-route BC still fails, diagnose route progress or temporal subgoals next. **No training was started for this clarification.**

Exact evidence: [v9 training manifest](../../outputs/experiments/2026-09-30_transport_continuation/transport_bc_paired_ports/manifest.json), [SC episode manifest](../../outputs/trajectory_datasets/transport_local_verified/2026-09-30_v9_paired_ports/manifest.json), [trainer launcher](../../scripts/run_clean_transport_bc.sh), [weight-loading code](../../scripts/train_clean_transport_bc.py), and [experiment record](../experiments/2026-09-30-transport-to-two-actor-continuation.md#checkpoint-provenance-correction-and-data-decision).
