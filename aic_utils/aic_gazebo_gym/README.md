# AIC Gazebo Gym

This package is the minimal skeleton for a ROS-free, training-only Gazebo
integration under `aic_utils/`.

The intent is to provide a small Python surface that can later launch and drive
Gazebo through pluggable backends and runtimes without modifying the official
evaluation flow.

Initial module boundaries:

- `aic_gazebo_gym.env`: public gym-like environment facade
- `aic_gazebo_gym.backend`: backend protocol for simulator-specific control
- `aic_gazebo_gym.runtime`: runtime protocol for process lifecycle management
- `aic_gazebo_gym.types`: shared dataclasses and type aliases

This milestone only establishes the package structure and import surface.
