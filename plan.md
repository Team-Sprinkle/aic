# Simulation/Data Pipeline TODO ( in order of priority )

# P0

- [x] Have a script that runs all commands at once: simulation, policy, and recorder.
- [ ] Ability to configure the number of episodes to collect per board setup
- [ ] Ability to configure the end-effector start position (above the board)
- [ ] Add more metadata to the trajectory info: success boolean, random seed (for replication)
- [ ] Try teleoperation on the randomized board: spacemouse and keyboard
- [ ] Improve data collection speed. Can collection without rendering help, if so by how much?
- [ ] Randomization: How to make it more realistic

# P1
- [ ] Remote rendering
- [ ] Reward model to filter dataset quality