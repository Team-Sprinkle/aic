# Why the SC port-orientation error dropped so much

The robot needs to know **where the requested port opening is and which way it faces** before it inserts the plug. Its three wrist cameras see the port, but a tiny change in orientation is hard to read from a small image crop. The previous neural network looked at the images and directly estimated the port's full 3D orientation. On the newer SC scenes, its near-port error was **4.434° p95**. That can be enough to put the plug against the rim instead of into the opening.

The trick was to use a larger, clearer visual clue: **the long straight edges of the board that holds the SC ports**. The board and port have a fixed mounting relationship in these simulator settings. When the board turns, its long edges turn with it. Those edges span hundreds of pixels, so their direction is easier to measure than the angle of the small port face.

The calculation works like this:

1. The image model first proposes the requested port's approximate image location and a rough orientation. This narrows the search to the relevant board edges.
2. In the **original 1152×1024 camera images**, an edge detector finds long straight lines near that proposal. The method checks that at least two cameras give compatible answers.
3. Using the measured robot pose and a fixed camera calibration, it works out the board's **yaw** (its turn around the vertical axis) from those lines. In these scenes, the board's other tilt angles and the board-to-port mounting rotation are fixed. Together, those give the port's full orientation. It can average accepted estimates over earlier observations of the same stationary port.

In short: **use the learned model to find the neighborhood, then use large visible board edges and camera geometry to measure the angle.** The learned orientation remains a rough search hint; it is no longer trusted as the final fine angle.

On **87 near-port image triplets from four development reset configurations**, the old orientation estimate had **4.434° p95 error** and the board-edge estimate had **0.056° p95 error**. “p95” means 95% of these sampled errors were at or below that value. The method also kept five previously saved valid transport stops and rejected one known stop where the robot's posture was wrong. Those six stops are a small, previously inspected diagnostic, so they are not a new independent success rate.

The estimator does **not** receive the true port pose during use. Ground-truth simulator poses were used to train/calibrate earlier components and to **measure the error afterward**. The offline test did use ground-truth in-frame and distance labels to choose which saved images to score; deployment therefore still needs an observation-only way to decide whether the board and requested port are actually visible.

This result is specific to the SC board layout with its fixed port mounting and visible straight edges. It does not prove the method will work when the rails are covered, the board geometry changes, or the robot starts insertion. The **100.27 ms p95** timing covered the offline image-processing path, not the complete live ROS-to-action loop. The automatic handoff and insertion actor remain gated until a live visibility, safety, and latency test passes.

The SFP improvement used a different trick: candidate port locations proposed in the three cameras must describe **one consistent 3D point**, and at least two cameras must recognize the requested port. That removed the two wrong accepted candidate triplets in the offline comparison; it does not establish full insertion performance.

[Full experiment, commands, and artifacts](../experiments/2026-10-03-geometric-handoff-repair.md).
