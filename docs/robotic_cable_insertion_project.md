# Robotic cable insertion: planning, perception, and contact recovery

Written by Yoon Jung (yoonjung0705@gmail.com)

This project tackles simulated insertion of a flexible cable connector into a small port while the robot works around nearby circuit cards. It combines high-level visual planning, collision-checked motion, camera-based pose estimation, imitation learning, and vision and contact-aware reinforcement learning. The experiments below use two different evaluation scopes: **full insertion from a transport start in Gazebo** and **local insertion from near the port in Isaac Lab** (Isaac for RL training near port).

## 1. Generate demonstrations around obstacles

I first used teleoperation, but started thinking if there's a systematic way to get expert trajectories. I used a vision-language model to choose an approach strategy (gpt-5-mini), MoveIt 2 for arm motion, trajectory smoothing for replay, and a privileged geometric controller for precise final alignment. Gazebo replay exposed a limitation of rigid-arm planning: the trailing flexible cable can interact with cards even when the arm has a valid path. I then hard-coded a low lateral bypass that descends behind the card row, travels beside it, tilts the plug in a clear area, and returns aligned to the selected port, producing **104 full insertions in 200 varied scenes (52%)**, spanning one to five intervening cards and two destination ports. Note this is a **privileged teacher** result (used information of ground truth target pose, obstacle positions, etc). Prior to insertion, the wrist was tilted according to the target pose, while ensuring the arm would not collide with the nearby cards.

![The robot tilts beyond the card row before returning to the port](images/cable_insertion/tilt_before_return.gif)

*Successful privileged teacher route. [Older-route port blockage](images/cable_insertion/older_route_port_blockage.gif) provides a visual comparison*

## 2. Predict the plug and port from wrist cameras

A three-camera model predicts the plug point and four port-opening corners, then triangulates a relative 3D pose and correction direction. Training uses simulator geometry as labels; the review image below draws **predictions only**. Cyan marks the opening polygon, magenta the plug, yellow the opening center, and green the correction arrow. On 173 near-port decisions from six held-out episodes, lateral pose error was **0.278 mm median / 0.670 mm p95**, with **94.1%** correction-direction accuracy. The remaining tail motivated later work for fusing different channels. 

![Predicted plug and port features in three camera views](images/cable_insertion/predicted_port_pose.png)

## 3. Compare visual policies, then predict connector trajectories

I first trained ACT and a Dreamer v4-style policy on Gazebo demonstrations. ACT learned arm commands from camera images, robot state, and the task description. It did not use the plug-to-port pose estimate or learn to predict that pose. The Dreamer-style model learned to predict future visual and robot-state features as an auxiliary training task, then learned commands from demonstrations. It did not explicitly predict the plug-to-port pose, and its deployed policy did not use imagined rollouts. These were full-start Gazebo experiments, separate from the near-port Isaac Lab experiment below.

For the near-port task, I used the camera-based pose estimator from section 2 as a **frozen input** to a PoseInsert-inspired policy, alongside camera images, force, and robot state. Both the diffusion model and a simpler direct-prediction model learned the connector's desired **position and orientation** after each of the next four 50 ms movements, relative to the port. A fixed measurement of how the robot grips the connector lets the controller turn those predictions into gripper movements.

The direct-prediction pipeline seated **6/6** connectors on six development starts. Each rollout first used a simulator-based correction to restore the plug tip near its requested start pose; this temporarily overrode policy commands. After that startup step, the frozen BC policy controlled the connector without the force-triggered backtracking controller from section 4. These are **8 mm local seating** results from near-port starts, not policy-only results from the first frame.

![Near-port local seating after the startup tip-restoration step](images/cable_insertion/local_policy_seating.gif)

*The left wrist-camera GIF shows one successful local seat. Its visible early backoff occurs during a rollout with the startup tip-restoration override enabled; the recorded policy takes over after the override releases. This is not the collision-recovery backtracking experiment in section 4.*

The world-model branch supplied a useful diagnostic: its supervised visual policy achieved **one full insertion in four development scenes** and reached a 25 mm opening envelope in **17/20** frozen scenes, versus **12/20** for ACT. Neither model fully inserted in the frozen 20-scene test. The preview is a **10× time-lapse of a one-frame-per-second recording** from the successful world model development scene, not a high-rate control video.

![World-model policy development insertion time-lapse](images/cable_insertion/world_model_development_timelapse.gif)

## 4. Detect blockage and retry with measured motion

I also tested a way to recover when the robot pushes but the connector stops moving. The controller compares commanded motion with **measured** robot motion and wrist force. If it detects a forceful stall, it backs along positions the robot actually reached, then lets the policy try again while limiting motion toward the blocked direction. Recovery commands are labeled separately in the replay data, so the learned policy is not trained to copy the retreat.

For this experiment, I started a probabilistic four-waypoint policy from the supervised controller and trained it with SAC-style reinforcement learning. The final RL policy matched, but did not improve on, its starting policy in a metrics-only eight-start test (**3/8** local seats each). In a separate recording-enabled run on eight near-port starts, the same RL policy seated **3/8 with recovery enabled** versus **1/8 without it**.

![Recovery-enabled RL rollout that reaches a local seat](images/cable_insertion/recovery_enabled_local_seating.gif)

*The cable movement in the scene is due to scene startup*
