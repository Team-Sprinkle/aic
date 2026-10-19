# Teacher versus runtime orientation information

Yes—that is the right understanding of how alignment can be learned.

- During data collection, the teacher knows the actual plug and port poses from simulator geometry. When the plug is tilted, it generates corrective commands. The recorded images and those commands become training examples.
- During deployment, the actor does not receive that privileged geometry. It uses images, robot state, and image-estimated plug/port poses to predict commands resembling the teacher's corrections.

The image-based orientation estimator itself is also trained with simulator-derived labels, but predicts from images during deployment. Ground truth supplies training supervision, not runtime policy inputs.

One qualification: this explains how the actor can learn alignment. We have not identified a specific teacher example that proves why it made a particular correction in the successful rollout, or established that it learned recovery after becoming jammed.
