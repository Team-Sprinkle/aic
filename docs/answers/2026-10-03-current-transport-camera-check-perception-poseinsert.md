# Which models are in the current SC test?

Yes, for the **latest SC shadow-handoff runs**, the working pipeline is essentially:

```text
three camera images + robot state + requested task/port
                 |
          transport actor
                 |
       near-port visual stop/servo
                 |
     selected-port camera verifier
                 |
          accept / reject only

    [insertion actor / PoseInsert: not connected]
```

The **transport actor** chooses body-frame movement chunks. Its near-port visual stop/servo uses a learned image locator and orientation probe to propose and reach a stopping pose. The **camera check** then holds the robot, obtains fresh images, and verifies that the *requested* opening is visible and geometrically consistent across cameras, that the tool is properly oriented, and that force is acceptable. This check **uses the perception models** (locator, selected-port identity head, and image-derived geometry). Thus “camera-check model” and “perception model” are not two wholly independent boxes; the camera check is the decision logic built around perception outputs.

The perception system currently produces information about the requested opening for **verification and possible alignment correction**. The recent orientation experiment exercised this check and correction only. It did **not** pass the estimated port pose to an insertion actor.

An earlier **PoseInsert-inspired port-frame insertion actor was trained and tested separately**, but it is **not attached to these current transport/shadow-verifier runs**. The runner explicitly stops or resumes after its inspection rather than starting insertion. We have therefore not demonstrated that this current three-part stack can insert autonomously end to end. Attaching the insertion actor is gated on a reliable selected-port handoff, and that gate currently fails for some valid SC arrivals.

See the [selected-port handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md) and [orientation audit](../experiments/2026-10-03-sc-orientation-identity-repair.md) for the detailed intended interface and latest result.
