# Why the force reading appeared, and what was fixed

The reproduced failure was **the SC plug catching against the selected port**, rather than a visible cable snag. The older 3.5 N number was a wrist-force change from a baseline, not a force assigned to one collision. An instrumented repeat recorded **4.22 N wrist change** and a simultaneous **8.83 N peak normal contact-point force** at the plug/port interface. The exact historical collision pair cannot be proven because that original recording lacked the complete contact stream.

I found a real Gazebo **contact-reporting** bug: its frozen Bullet version reported impulse as identical X/Y/Z force components and reversed the penetration-depth sign. I implemented the [upstream reporting correction](https://github.com/gazebosim/gz-physics/commit/f34d6e388a841aa5eddc276acd01413940841ed5), preserving raw records and collision physics. This bug is separate from the wrist sensor; it does not establish that the wrist reading was false. Telemetry now also uses the correct sensor frame and distinguishes commanded retreat from measured retreat.

The newer **tilt-before-return route inserted fully in 2/2 oracle repeats**, with sampled peak plug/port point forces of **0.79 and 0.75 N**. All collisions and force limits stayed enabled. These are privileged teacher tests on one development scene, not evidence of general autonomous-policy reliability. The older route's recovery remains unsuccessful: 24 mm commanded withdrawal produced only about 0.006 mm actual movement.

I fetched the latest official toolkit and imported compatible documentation and CAD assets. Its SC port-numbering change is staged separately because applying it silently would change the frozen scene and task labels. The customized dependency lock was preserved. No training job was interrupted and no training was started; only GPU 7 was used sequentially for these diagnostic rollouts.

## Videos and evidence

- [Reproduced failure, two close views with contact markers and forces](../../outputs/experiments/2026-09-30_force_oracle/old_oracle_r2/oracle_video/port_contacts_20fps.mp4)
- [Successful new route, same views](../../outputs/experiments/2026-09-30_force_oracle/new_oracle_r2/oracle_video/port_contacts_20fps.mp4)
- [Force versus commanded/measured motion](../../outputs/experiments/2026-09-30_force_oracle/old_oracle_r2/force_and_motion.png)
- [Detailed diagnosis, commands, limitations, and artifact map](../experiments/2026-09-30-force-oracle.md)
- [Machine-readable results and hashes](../experiments/2026-09-30-force-oracle-results.json)

The videos are 20 fps and exclude post-reset frames. I inspected the marked failure images and the seated terminal plug from both angles. Contact data is sampled at 100 Hz; the physics runs at 500 Hz, so very brief impulses may be missed. Corrected forces are normal contact forces, not full friction-inclusive forces.

Main documentation is updated. The next gate remains new-start route generalization and reliable teacher admission; Step 5 and final scenes stay closed to promotion. A compact evidence copy is retained on NAS under `/home/nas/chmin/aic_archive_20260930/force_oracle/`.
