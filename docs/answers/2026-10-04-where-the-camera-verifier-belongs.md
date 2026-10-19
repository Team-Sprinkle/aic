# Where should the camera check happen?

**At both levels, with different jobs.** Coarse perception should make an **early reject / pause decision**: does the requested destination seem to be in a safe, reachable neighborhood, or is this plainly the wrong card/port or a bad view? This check can run during transport or when the transport actor proposes a stop. It should not authorize insertion by itself: our coarse SFP predictions have made coherent neighboring-port errors.

After the robot holds still, the system uses the original-resolution images to inspect the **requested card and port** more precisely. A final verifier combines the refined multi-camera 3D pose with identity, visibility, tool standoff/orientation, and force/contact checks. It returns a usable estimated pose or **invalid**. Only a valid final result may hand control to the insertion actor.

```text
transport + task code
       |
       v
coarse requested-target check: reject / continue / pause to inspect
       |
       v
hold → original-resolution requested-port refinement
       |
       v
final verifier: identity + 3D pose + view + posture + force
       |
       +--- invalid → transport / seek view / hold
       +--- valid   → insertion actor
```

So the coarse check **is part of the camera-check pipeline**, but its “yes” means *inspect further*, not *insert now*. The paired-port crop experiment targets the final precise identity check; it does not repair an early wrong-route stop by itself. The full handoff still needs both gates tested together on new autonomous development starts, with final scenes sealed.
