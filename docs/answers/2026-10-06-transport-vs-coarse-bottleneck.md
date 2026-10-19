# Transport versus coarse perception: current bottlenecks

**Short answer:** The latest wrist-view experiment is **not good enough to deploy**. It is good enough to stop *that view idea*: a true +10° view did not reveal the requested numbered SFP card in two cameras and exceeded its 5 mm held-position limit. The recent numbers below concern **SFP-to-NIC**, not a new SC-to-SC evaluation.

## What fails first in an autonomous rollout?

**Transport.** On three new one/two-card development starts, the latest actor reached the requested safe buffered region in **0/3** episodes, while matched CheatCode teachers fully inserted. Its closest lateral distances in the axial approach band were **23.3, 31.6 and 14.4 mm**, above the 10 mm arrival gate. On those *same sparse routes*, a frozen visual locator could identify the requested port early in **3/3** episodes when run separately on saved images. Thus, for those scenes, an available coarse visual target did not translate into a correct transport route. The current transport actor does not consume that locator output; its own pooled features and task code had to learn the route. A same-image request-code swap changed its first target by only **0.554 mm median**, whereas verified same-board teachers branched much more. This is evidence of weak target-specific routing, although the three-start sample is small.

## What blocks a safe transport-to-insertion handoff or a visual transport repair?

**Coarse requested-card identity.** On a harder ten-request replay, the selected-port model acquired an early anchor in only **3/10** episodes. The newer precise single-point crop head was zero-wrong on one ten-scene set but early on only **6/10** in another. A three-camera candidate matcher reached 10/10 early on fresh scenes but accepted **eight wrong neighboring-card points**. Its good reprojection and persistence did not prove it had selected the requested numbered card. The +80 mm, −80 mm and +10° camera-view diagnostics did not establish a reliable physical slot reference. The precise refiner's low error on *accepted* frames does not make a wrong-card or missing proposal safe. The coarse system therefore cannot yet authorize automatic handoff, and it does not provide a dependable target token for a new transport actor across all task settings.

These are **two separate bottlenecks**:

| Question | Current answer | Practical consequence |
|---|---|---|
| Can the current actor approach the requested port? | No: 0/3 safe arrivals on sparse new starts. | Even a perfect refiner cannot insert from the wrong approach corridor. |
| Can the current coarse system reliably identify the requested numbered port? | No across the harder balanced cases: 3/10 early, or wrong-neighbor acceptances in a more permissive method. | It cannot safely trigger handoff or serve as a trusted target cue everywhere. |
| Is the +10° active view a fix? | No on its one tested scene; visual and measured-position gates failed. | Stop this view branch. |

**Priority:** For *current autonomous performance*, transport is the first visible failure. For *unlocking a target-conditioned transport repair*, coarse card identity is the prerequisite. We should solve the identity input on representative hard scenes, then train/test a transport actor that explicitly uses it. More generic BC updates or RL on the current actor would not address the demonstrated target-code insensitivity.

## Next experiments

1. **Complete the 40-panel independent image audit.** An independent reviewer uses the already prepared [unmarked packet](</home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/independent_review_packet.md>) and records selected card, both openings, actual visibility/occlusion, and uncertainty before seeing truth. Only 11 panels currently have formal prereveal annotations. This determines whether original images/history contain enough evidence and in which phases. Do not treat projected-in-frame ports as visually visible.
2. **Use a distinct board-reference hypothesis, not another generic wrist sweep.** If the reviewer can resolve hard scenes, train one full-row plus candidate-crop association head with same-board neighbor negatives and abstention. If not, design an *earlier, safe observation posture* that shows a fixed board end and the complete row, verified first with unmarked original RGB and measured collision/return traces. Stop if two cameras still cannot establish the numbered slot. The later +10° tilt failed its own 5 mm position gate, so it is not a deployment action.
3. **Prove identity on new grouped development starts.** Require zero accepted wrong cards, an early usable requested-port proposal in at least 9/10 reset groups, accepted 3D p95 ≤20 mm, and full inference p95 <300 ms. Keep final scenes sealed. The current spatial-actor experiment explicitly requires this identity gate before fitting.
4. **Run one matched spatial transport comparison.** Feed the observation-derived requested-card/port cue and public task code to a transport actor; compare with the current actor on identical replay, split and update budget. Check same-board alternate requests and early route actions, then at most three new nonfinal live starts. Require at least 2/3 safe buffered arrivals without more contact and p95 inference <300 ms before connecting the refiner/insertion actor. Never pass true simulator port coordinates to the deployed actor.
5. **Only after both gates pass, test insertion and recovery.** The refined pose and insertion policy can then be evaluated autonomously; RL should wait until local states and rewards are trustworthy. Preserve the final split for a later frozen evaluation.

Sources in this repository: [sparse transport/vision comparison](../experiments/2026-10-05-spatial-transport-prerequisite.md), [matched actor continuation](../experiments/2026-10-05-transport-route-handoff-continuation.md), [wrong-card camera candidates](../experiments/2026-10-06-sfp-joint-camera-candidates.md), [failed board-view pilot](../experiments/2026-10-06-sfp-board-tilt-view.md), and [current status](../STATUS.md).
