# Using approach history for port identity; leaving insertion correction to PoseInsert

Yes, this is a sensible direction, with two distinctions.

## What the current camera check is

The **selected-port identity classifier** is a small head on spatial features from the frozen image locator backbone. It also receives the requested port code and measured TCP state. The **complete camera check** does more than run this head: it refines an opening in the original-resolution images, combines the three calibrated cameras to estimate 3D position, estimates orientation from visible board lines, and checks force, standoff, and view agreement. So calling the identity part “a head on the perception backbone” is accurate; calling the entire check *just* that head is too narrow.

The camera check's main job should be **correct target identity and a trustworthy pose handoff**. It need not independently finish fine insertion alignment. A newly trained PoseInsert-style actor can take the estimated port frame and command complete port-relative tool targets, including rotation, while watching images and force. The verifier still needs to reject absent or wrong ports, implausible pose jumps, and unsafe contact. We should keep its posture requirement conservative until the insertion actor demonstrates a wider safe starting range.

## Why sparse approach history helps

Your close-up thought experiment is right: when only a small piece of the requested port remains visible, a single image may not distinguish port 0 from port 1. Earlier approach images may have shown both ports and their board context clearly. The robot can use those earlier views to establish **which physical opening matches the task code**, then carry that identity forward as its cameras move. The port is stationary within an episode, and measured TCP motion plus fixed camera calibration can predict where that same opening should appear later. Fresh images must confirm the tracked opening when possible; history is not permission to invent a port that was never identified.

I would start with an **event/distance-based keyframe buffer**, not dense video: retain a clear overview at the first reliable identification, then a frame after roughly every **10 mm of measured TCP translation in the final 50–100 mm of approach** or a meaningful view/orientation change. Earlier in transport, 30–50 mm spacing is a reasonable initial budget. Always capture fresh three-camera images at the proposed handoff. These are *starting parameters to measure*, not values demonstrated by current experiments. A strict fixed interval can miss a fast occlusion transition, so also save a frame just before visibility drops. Cap the buffer at about 6–10 synchronized triplets, record measured TCP poses and task code with each, and reset it on episode reset or a target change. Compare, on held-out complete resets, the current single-view head with a history-aware identity/pose check using exactly the same current frame. Measure wrong-port and absent-port accepts, present-port recall, pose error, and full p95 latency below 300 ms. If the history lacks an unambiguous anchor or motion tracking disagrees with a new clear view, return **invalid** and seek a better view.

## Can PoseInsert correct 6.7°?

**Possibly, but this has not been shown.** A 6.7° rotation is modest for free-space alignment, but potentially large at the lip of a tight port. As a scale example, rotating a point 20 mm from the rotation center by 6.7° moves it sideways about $20\sin(6.7^\circ)\approx2.3$ mm, larger than the 0.5 mm lateral insertion corridor. The actual plug-tip motion depends on the TCP-to-tip offset and rotation axis. Therefore a PoseInsert actor should start at a deliberately safe axial standoff, rotate and translate into alignment there, and only then advance into contact. It should be trained and tested on *measured* initial orientation offsets including 0°, about 3°, 6–7°, and perhaps 10°, with varied lateral offsets and both task families. Observe plug-tip position, force, and success, and define the largest safe accepted handoff range from those results. Until then, the current 3° verifier limit is a conservative experiment gate, not proof that 6.7° is uncorrectable.

## Practical next sequence

1. Improve **identity across the approach** with sparse, physically consistent RGB/TCP histories. Keep simulator geometry out of runtime inputs and preserve the existing absent/wrong-port rejection tests.
2. Hand a verified port pose and a safe standoff to a **retrained PoseInsert-style insertion actor**. Let that actor perform final rotation, lateral alignment, advance, and local recovery. The check should not rotate repeatedly just to meet an artificially narrow actor starting distribution.
3. Run matched held-out start-offset tests to establish the actor's real correction envelope, then set the handoff orientation/standoff limits from measured safety and insertion results. Keep the four reserved final IK scenes sealed until development gates pass.

The earlier local port-frame actor did **not** establish this capability: SC near-port runs still scored zero insertions and sometimes developed much larger orientation error during approach. Conversely, an earlier PoseInsert-inspired model succeeded on local Isaac SFP-to-NIC development starts; that does not establish SC or normal-start Gazebo capability. A retrained, attached actor needs its own evaluation.

See the [handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md), [current SC orientation audit](../experiments/2026-10-03-sc-orientation-identity-repair.md), and [local actor diagnosis](../experiments/2026-10-02-two-family-perception-local-bc.md).
