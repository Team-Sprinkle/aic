# Simulation/Data Pipeline TODO ( in order of priority )

> Archived on 2026-09-17 from `plan.md`.
> See the [maintained documentation](../docs/README.md) for current status and workflows.
> Status, recommendations, and commands below describe the original work.

> Historical data-pipeline checklist, retained for context. The maintained
> next-step list is in [docs/STATUS.md](../docs/STATUS.md); record new experiments
> through [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

# P0

- [x] Have a script that runs all commands at once: simulation, policy, and recorder.
- [x] Ability to configure the number of episodes to collect per board setup
- [ ] Verify correctness of trajectory collection (success criteria)
- [x] Randomization: How to make it more realistic
- [ ] More metadata in collected trjectory: score of the run
- [ ] Collect teleoperation on the randomized board: spacemouse and keyboard
- [ ] Improve data collection speed.

# P1
- [ ] Remote rendering
- [ ] Reward model to filter dataset quality
