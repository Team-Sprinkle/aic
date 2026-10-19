# Is board-anchored transport still worth pursuing?

**Yes, but not by extending the latest slot-feature heads.** Their failure is strong evidence against that *implementation*, not against the whole direction. The first head predicted independent camera points that disagreed in 3D. The second predicted one consistent 3D point but confused neighboring cards and was overconfident. Neither maintained a verified board origin and slot order as the robot moved. The frozen candidate set also lacked the requested physical point early in 6/10 hard scenes, so merely improving candidate matching is insufficient.

## The next version I would actually test

Build a small **board landmark tracker**, rather than another direct “image → requested port” regressor:

1. From the original-resolution three-camera images, detect **fixed visual references** such as the board/rail end, slot-zero groove, and visible slot centers. Predict *which references are visible* and abstain when their identities are ambiguous. Simulator geometry can supervise training, but cannot choose landmarks during deployment.
2. Use calibrated cameras and **measured TCP motion** to combine observations from different positions into one persistent board-coordinate estimate. A landmark seen earlier remains useful after it leaves the current frame, with growing uncertainty. The 40 mm CAD slot spacing helps constrain the map; it cannot replace a missing slot-zero anchor. Reset this map on every episode reset.
3. Apply the public requested card/port code to that map. Output one selected target cue, its uncertainty, and “unknown” when the board origin or slot order is unresolved. The near-port high-resolution refiner still estimates the final opening; the map only needs transport-level accuracy first.

This uses the feature that the board is **static during an episode**, even though the wrist cameras move. A lightweight landmark detector plus a small geometric filter or windowed optimizer may also fit the 300 ms budget better than repeatedly running large full-image networks. I would compare it with one spatial-token/history network only if the explicit tracker cannot use the visible cues reliably; doing both at once would obscure the failure cause.

## The first question is observability

The corrected 30-panel review packet includes images, task request, history and measured TCP. It still needs an **independent review**. The reviewer should mark when slot zero or another unambiguous board origin is visible, when the target opening is identifiable, and when both are unknown. The earlier 11 RGB-only abstentions omitted TCP state and cannot answer this question.

If the origin is never identifiable along the ordinary route, test **one safe earlier inspection posture** designed to expose it. The previous sideways and tilt views did not solve this; another generic angle sweep is not justified. Check original unmarked images and measured clearance before treating the new posture as usable. If neither ordinary nor one safe inspection view reveals an anchor, then board-anchored numbering is probably **not viable with the present observation/action setup**. At that point we should revise the sensing or route strategy instead of fitting more identity heads to ambiguous images.

## How transport should use the cue

If the map passes its identity gate, feed its selected target cue **into the temporal transport representation before route decisions**, and supervise target-specific waypoints or route stage as well as full actions. Attaching a 3D vector only to a frozen final action head is too weak a test. The privileged true-cue head improved teacher-state one-step p95 by only 2.9%, but those teacher states already had about 2 mm error and do not represent the autonomous routes that missed the approach region. A decisive transport test needs same-board alternate requests, distinct early-route actions, autonomous off-route states, and safe-arrival/contact outcomes.

## Gates and decision

- **Identity first:** zero accepted wrong requested ports and at least 9/10 early usable cues on new complete-reset development scenes. Report abstentions, board-origin availability, per-card results and full inference latency. The 9/10 bar is a development gate, not a reliability claim.
- **Then transport:** a matched target-aware actor must reach the requested 10 mm buffered approach region on at least 2/3 new development starts without more contact, then face a larger grouped check. The cue must be predicted from observations, never true simulator geometry.
- **Then handoff:** fresh correct-target identity, refined opening pose, measured plug/TCP position and orientation, acceptable force, and complete p95 inference below 300 ms. Only then test autonomous insertion and recovery.

**Recommendation:** pursue one explicit landmark-and-memory prototype after the observation review. Do not spend another cycle on crop-width tuning, static world-position priors, or confidence calibration of the failed 3D head. The largest remaining uncertainty is whether the cameras ever provide an unambiguous board origin early enough. Resolve that before another training run.

Evidence: [architecture review](2026-10-07-handoff-architecture-review.md), [bounded redesign result](../experiments/2026-10-07-sfp-slot-feature-port-cue.md), and [observation-complete review packet](../experiments/2026-10-07-sfp-observation-complete-review-and-tight-crop.md).
