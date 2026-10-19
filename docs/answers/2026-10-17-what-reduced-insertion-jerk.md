# What reduced the jerk?

**We reduced the SFP oscillation; we have not fully fixed the jerk.** We did not add an action smoother or backtracking to achieve this result.

The main change was the full-connector pose treatment:

- Estimate the connector's actual orientation relative to the gripper from images.
- Retrain the actor with the connector's position and orientation relative to the port.
- Convert its predicted connector targets into gripper commands using that full estimated grasp relationship.
- Keep the estimated port and grasp transforms fixed during each insertion attempt, avoiding frame changes between predictions. The port lock already existed in the previous comparison; locking alone is not a newly isolated explanation.

The likely benefit is that successive plans agree better about which way the actual plug should point. Previously, connector-tip position was combined with gripper orientation, leaving the actor to infer their relationship. This is an explanation consistent with the results, not a separately proven causal mechanism: estimator, representation, conversion and retraining changed together.

In the first 15 seconds of SFP insertion, before simulator completion locking, the p95 jump between successive plans' commanded positions decreased from **15.4 mm** with the previous H4 actor to **4.4 mm and 3.2 mm** in the two new repeats. These are command-target jumps, not direct physical jerk measurements. History length and the four-step prediction/execution schedule stayed unchanged.

Two problems remain:

1. **The transition into the insertion actor still jumps:** about 4–8 mm in the SFP trials.
2. **Forward demand can build while contact blocks motion:** after alignment improves, the plug can advance rapidly.

SC did not show a consistent smoothness improvement. Also, the long still portions after completion are partly simulator-enforced locking; they do not count as a jerk fix.

The next targeted fixes would be a continuous command transition between controllers/plans, followed by a progress-aware limit on forward demand during blocked contact. Neither has been tested yet.
