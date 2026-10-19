# What perception does after transport stops (October 2, 2026)

## Short answer

The current perception pipeline **first predicts image locations for every possible port identity**, then uses the public task code to choose the requested one. It refines **that selected port**, combines camera views, and uses measured robot pose plus camera calibration to estimate a 3D port pose. It does **not** currently produce a trustworthy true-visibility score or a calibrated probability that its 3D pose is accurate.

Also, the two stages are not completely sequential: the experimental SC visual transport controller already estimates the selected SC port **before** transport stops, to approach its waiting pose. The separate insertion perception pipeline runs on fresh camera observations when invoked near the port. The code does not yet implement an approved automatic transport-to-insertion switch.

## Example: SC port 1 is requested

1. The task request is encoded as `[is_SFP=0, is_SC=1, port_0=0, port_1=1, ...]`. Transport uses that code while routing around the cards.
2. Near the port, the **transport visual controller** predicts both SC opening locations in each of the three camera images. It selects the port-1 predictions, triangulates a 3D opening from the cameras, and steers to a pre-insertion waiting pose. Its stop decision is based on this **estimated** pose and measured TCP, not the true simulator geometry. This is the development-only SC stop described in [the stop explanation](2026-10-02-how-transport-stops-during-deployment.md).
3. If a future validated handoff accepts that stop, the robot should hold its measured pose while the already-loaded **local insertion pipeline** starts. It reads fresh RGB images, measured robot state, and the same task code. Its shared coarse locator predicts **12 candidate identities per camera**: two SC ports and two ports on each of five possible NIC cards. Absent candidates can still have a raw coordinate; they should be ignored when their projected-in-frame flag is low. The task code selects `sc_port_1` from the candidates.
4. The selected SC opening is refined using a 160×160 crop cut from each original-resolution image. The crop center comes from the RGB locator's own prediction. At least two agreeing camera views are needed for 3D triangulation. A separate orientation head uses visual features, candidate geometry, selected-target identity, and the measured TCP pose to estimate the port orientation.
5. The resulting port pose and measured TCP pose define the current TCP pose **relative to that port**:

   $$T_{\text{port}\leftarrow\text{TCP}} = T_{\text{base}\leftarrow\text{port}}^{-1}T_{\text{base}\leftarrow\text{TCP}}.$$

   The first transform is estimated from RGB and calibration. The second comes from ordinary robot state. This relative pose, the three images, measured state, and the task code go into the insertion actor. Its four predicted target TCP poses are expressed in the **estimated port frame**; runtime turns them into Cartesian robot commands.

For SFP, the selection is more specific: the code might request `NIC card 3, port 0`. The coarse locator still predicts all candidates, then picks exactly `sfp_card_3_port_0`. The SFP cascade refines that candidate using a 480-pixel region and then a 160-pixel crop from the original image. It currently requires all three selected views to pass its projected-in-frame check. The different port on card 3, or port 0 on another card, must not be substituted.

## Exactly where proprioception and task code enter

| Input | Where it is used |
| --- | --- |
| Three RGB camera images | The coarse locator predicts candidate opening pixels; selected-port crop models refine pixels; visual features help the orientation estimate; the insertion actor also sees RGB. |
| Task code | Selects SC versus SFP and the requested port/card identity **after the coarse all-candidate prediction**. It also conditions the insertion actor. The coarse locator receives family and camera identity metadata, rather than the full selected-target code; the later orientation calculation explicitly includes the selected identity. |
| Measured robot/TCP pose | Places the calibrated cameras in 3D, allowing predicted image points to be triangulated; converts the estimated port into the current TCP-in-port pose. It also enters the orientation head and insertion actor. It does not replace visual detection of the requested opening. |
| Fixed camera calibration | Gives each camera's pose relative to the TCP. It is a calibration constant, not a simulator readout of the target. |

This is why the perception model can predict **image locations** from vision but cannot derive millimeter-scale **3D coordinates** from the pixels alone. The measured robot pose tells it where those moving wrist cameras were when the images were taken.

## Visibility and confidence: what exists today

The coarse locator gives each candidate, in each camera, a probability that its **projected location lies inside the image**. This is useful for rejecting absent or offscreen candidates, but it is **not true visibility**. A port can project inside the image while a cable, card, or robot part blocks it. The returned diagnostics explicitly say `visibility_certified: False`.

The pipeline also checks whether two or three camera rays agree, whether the finer crop remains near the coarse guess, and whether several estimates over time are consistent. For SFP, five sufficiently consistent estimates can establish a static-port lock; SC has a similar five-estimate lock with its own tolerances. These are **engineering consistency checks**, not a calibrated confidence such as “95% probability that the port is within 0.5 mm.” A consistently mistaken identification can pass a consistency check. We have observed accepted wrong-port estimates in development data.

If the selected port is unavailable, the current local runner can briefly reuse a recent estimate, and it can retain a stable locked pose through occlusion. That is useful because the port is fixed during an episode, but the carried or locked pose is **not proof** that the selected port was identified correctly in the first place. A deployable handoff still needs a validated selected-target identity and visibility/accuracy gate. Until that exists, the system should not switch to insertion merely because another port is visible or because repeated estimates agree.

For current evidence, see [perception and local BC experiments](../experiments/2026-10-02-two-family-perception-local-bc.md), [handoff design](../POSE_VISIBILITY_HANDOFF.md), and [local runtime](../../aic_example_policies/aic_example_policies/ros/RunPredictedPortLocal.py).
