# Can visibility replace confidence at deployment?

**Partly, but not completely.** Visibility answers **“Can I identify the requested opening in these images?”** It can reject an offscreen or occluded target. It does **not** answer **“Is my estimated 3D pose accurate enough for insertion?”** A clearly visible opening can still be localized incorrectly because the model picks the neighboring port, places a keypoint a few pixels off, or infers the wrong orientation.

So I recommend **no learned confidence output for the next model**, but also **do not treat visibility = yes as permission to insert**. Use visibility as the first gate, followed by explicit checks on the pose estimate:

```text
Transport proposes a near-target stop
        |
        v
Is the REQUESTED port identifiable in enough camera views?
        | no → hold / improve view / continue transport
        v yes
Do the views agree on one plausible 3D opening and target identity?
        | no → reject the handoff
        v yes
Is that estimate stable across several fresh observations?
        | no → reject the handoff
        v yes
Is the measured tool at a safe pre-insertion pose, with acceptable
orientation, route, and force/contact?
        | no → remain in transport or hold
        v yes
Start insertion
```

The **pose model** gives position and orientation; **visibility** says which camera observations are usable; the other checks are deterministic decisions computed from those outputs and measured robot signals. This keeps the design simpler than a learned confidence head.

However, these checks are only a **proposed safety gate**. In our development data, a wrong physical port once gave a geometrically consistent multi-camera fit. We must measure the gate's false acceptances on wrong-port, occluded-target, and wrong-route scenes before trusting it. If the simple gate cannot separate those failures from good handoffs, then consider a calibrated pose-quality model later. Visibility alone is not a substitute for that evidence.
