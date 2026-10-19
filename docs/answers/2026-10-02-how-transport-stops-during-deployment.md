# How transport would stop during deployment (October 2, 2026)

## The crucial distinction

Transport should stop at a **safe waiting pose in front of the selected port**, not after the plug has entered it. The insertion actor would take over from there. The robot cannot compare itself with the simulator's true port pose at deployment. It must decide from its cameras, measured robot pose, task request, and recent observations.

The current code has several **experimental stopping rules**, but none has been approved as a general switch from transport to insertion. In particular, the transport runner currently **returns when its stop rule fires; it does not automatically start the insertion actor**. The switch and measured-pose hold described below are the deployment design still to be completed.

## One concrete SC-to-SC example

Suppose the task requests `sc_port_1` and there are five cards between the plug and that port.

1. At episode start, the controller encodes the request as SC task, port 1. It reads three camera images and the measured robot state. The temporal transport actor uses those inputs and its motion/command history to route the plug around the card row. It issues short TCP-frame command chunks, then observes again. It does **not** need to see the destination port at the start.
2. Near the destination, the experimental SC visual locator runs. It predicts both possible SC openings in the three images; the task code selects **port 1**, not merely the nearest visible port. Fixed camera calibration and measured TCP pose turn those image predictions into an estimated 3D opening. The controller constructs a **pre-insertion standoff** from that opening. This is a waiting pose, not an instruction to push into the socket.
3. The current SC visual controller considers a candidate usable only after at least 30 seconds of simulated transport, when the estimated standoff is within 80 mm of the measured TCP, the three-view reprojection error is at most 2 pixels, and the new estimate is within 5 mm of the preceding target. It activates its short visual approach after **three consecutive** usable estimates within 60 mm. It then steers toward a filtered standoff estimate, taking at most 8 mm translation per visual correction by default.
4. Its current **stop condition** is three consecutive decisions with the measured TCP within **8 mm of its *predicted* standoff**, predicted orientation correction at most **1°** when the orientation servo is enabled, and a usable visual estimate no more than one second old. The check is made before issuing the next command chunk. If it fires, the runner logs a stop and returns. At a 20 Hz command rate, three consecutive decisions nominally span about 0.15 seconds, although actual observation timing can vary.

Those 8 mm and 1° values are **internal prediction-based stop settings**. The **10 mm position / 3° orientation** numbers reported in our experiments are a *separate post-run test* against the true intended waiting pose. The robot cannot read that true pose while acting. Also, this stop does not certify that the cable took a safe route or that the plug tip is aligned with the opening.

There is another tested rule: a learned transport head estimates a probability that the robot is ready to hand off. In the frozen SC comparison, it required probability at least **0.95 on two consecutive decisions after 15 seconds**. This was a comparison rule, not an active insertion switch. The runner has other optional diagnostic goal-probe rules. Its default learned-head stop is disabled unless a threshold is configured.

## What happened in tests

On seven saved SC development trajectories, the visual stop selected **five valid waiting poses, one false stop, and missed no valid arrival**. The learned transport-only head selected **two valid stops, two false stops, and missed three arrivals**. The visual rule's false stop is the key problem: the measured TCP was only **1.02 mm** from the intended waiting *position*, but its orientation was **6.71°** off and its route was poor. The current rule could therefore hand control to insertion in an unsafe or unrecoverable posture. This is why the visual rule remains a development verifier.

SFP-to-NIC is less ready. The current SFP transport runner has **no validated observation-only stop**. The local SFP perception pipeline can wait for five agreeing port estimates before moving, but that is a lock **after it has been invoked**, not a proven decision for *when to end transport*. In a wrong-target development check, individual accepted estimates selected another physical port. If the requested opening is offscreen or obscured, transport must continue or safely wait; another visible port must not trigger the switch.

## The intended deployment switch

The final controller should run as a small state machine:

```text
TRANSPORT
  ├─ no reliable selected-port pose yet → continue transport
  ├─ unclear target / force or route concern → hold or use a tested recovery rule
  └─ stable selected-port pose + safe waiting position/orientation + safe route
                                      ↓
                                  HOLD TCP
                                      ↓
                     switch to preloaded INSERTION actor
                                      ↓
              align → approach → insert, while checking force/contact
```

The safe-switch check needs **all** of the following, based on observations only: evidence that the *requested* port was selected; a stable pose estimate across a short window; measured TCP close enough to a safe **pre-insertion** standoff with acceptable orientation; no disqualifying route/contact/force signal; and complete inference below 300 ms p95. The exact confidence thresholds still need validation. A repeated estimate alone is insufficient because a model can repeatedly pick the wrong port.

Once the switch is accepted, the controller should keep commanding the **current measured TCP pose** while the already-loaded insertion actor takes over. That avoids a jump or drift during model loading. The insertion actor then gets a fresh camera observation, measured state, the same task code, and the **vision-estimated** TCP-in-port pose. It predicts complete target TCP poses in that estimated port frame; runtime converts them into base-frame Cartesian commands. If the estimate becomes unusable or force rises, a tested hold/recovery rule should intervene. This integrated switch has **not yet passed** autonomous full-episode evaluation.

See [the stop comparison](../experiments/2026-10-02-transport-stop-comparison.md), [handoff design](../../obsolete/docs/2026-10-19-retired-plans/POSE_VISIBILITY_HANDOFF.md), and [the transport runtime](../../aic_example_policies/aic_example_policies/ros/RunTemporalTransport.py) for the current evidence and code.
