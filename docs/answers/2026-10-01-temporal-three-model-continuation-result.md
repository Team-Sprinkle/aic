# Temporal transport and three-model continuation result

I applied the requested supervised sequence: **transport actor → task-conditioned perception → port-relative insertion actor**, followed by a stable handoff. The transport actor is shared across SFP-to-NIC and SC-to-SC. Perception and insertion remain separate models. Hierarchical backtracking and RL are later, matched experiments.

The existing data audit passed for **104 SC** full-success episodes split **83 train / 21 validation** by scene/reset and **268 SFP** episodes split **236 / 32**. All six accepted SC Low Lateral Bypass stages and the two scripted holds (8 s and 5 s) are present. The retained action labels reconstruct the TCP-body targets exactly; SC image timestamps are no later than their decision. SFP coverage is skewed: **227/236 training episodes** target card 0, port 1. [Machine audit](../../outputs/experiments/2026-10-01_temporal_transport/label_audit.json).

I cached frozen three-camera visual features from the previous BC checkpoint for **55,142** causal 200 ms decisions and trained a small transformer head with full TCP-body actions, SC route stage, complete TCP-frame subgoal, and handoff supervision. The matched history versus no-history comparison used the same episode split, sampling, architecture, and update ceiling. One separate targeted repair exposed measured TCP position and task code directly to the head.

| Model | Full held-out SC stage accuracy | Full held-out SC subgoal error p95 | SC first-action error p95 |
| --- | ---: | ---: | ---: |
| Temporal history | 91.22% | 41.51 mm | 2.80 mm |
| No history | 90.62% | 46.38 mm | 2.85 mm |
| History + measured-state skip | 87.13% | 39.50 mm | 2.53 mm |

The history helped, but **none passed the 20 mm subgoal gate**. These are predictions on held-out teacher observations, not autonomous route results. A full-validation plot and checkpoint hashes are in the [comparison artifacts](../../outputs/experiments/2026-10-01_temporal_transport/full_heldout_comparison.json) and [machine decision](../../outputs/experiments/2026-10-01_temporal_transport/decision.json).

The auxiliary handoff head also benefited from history: at a fixed 0.5 threshold, it made **7 false handoff predictions** on held-out SC teacher frames versus **43** without history. The positive class had only 37 decisions, and this head is not a safe autonomous switch yet. [Handoff check](../../outputs/experiments/2026-10-01_temporal_transport/full_heldout_handoff_check.json).

The largest failure is at the tilt-hold → return transition. In 21 held-out episodes, adjacent frames around that boundary had only **0.026 mm median measured TCP movement** and **0.061/255 mean center-image pixel difference**. I inspected the [three-view before/after example](../../outputs/experiments/2026-10-01_temporal_transport/tilt_return_boundary_llb_0074.jpg); the scene is almost unchanged. A deployable controller needs causal stage memory, measured waypoint/tilt progress, and a dwell/contact-stability check to end that hold. A clock alone is insufficient. [Boundary measurements](../../outputs/experiments/2026-10-01_temporal_transport/tilt_return_boundary.json).

**Gate decision:** I did not launch a new Gazebo rollout, train perception/insertion, enable backtracking or RL, or open the reserved final scenes. The earlier single-frame BC result remains **0/16 stable 10 mm handoffs**; no live inference latency is claimed for the new temporal candidate. The next falsifiable repair is an explicit observation-only route-progress state and stage-conditioned full subgoal head, on the same split, with a one-GPU limit of 3,000 updates or 60 minutes. Promote it to at most two opened development rollouts only if full held-out SC stage accuracy is at least 90%, subgoal translation p95 is at most 20 mm, and there are no stage skips. It must then stably hold within 10 mm, follow the around-card route with acceptable force, and meet the 300 ms inference p95 limit.

The [experiment record](../experiments/2026-10-01-temporal-three-model-continuation.md) contains the design, commands, scores, plots, and limitations. The [command manifest](../../outputs/experiments/2026-10-01_temporal_transport/commands.json) lists rootless image/GPU/mounts and Python arguments; all checkpoints, source snapshots, logs, and failed results remain under `outputs/experiments/2026-10-01_temporal_transport/`.
