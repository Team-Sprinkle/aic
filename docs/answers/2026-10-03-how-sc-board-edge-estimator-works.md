# How the SC board-edge estimator finds its lines

It sounds more complicated than the code is. **There is no neural network trained to label individual board edges.** The pipeline combines a rough learned port estimate with ordinary image edge detection and checks whether the resulting lines fit the board geometry.

Imagine looking at a long straight rail beside the SC ports. Measuring the angle of that rail across hundreds of pixels is easier than measuring the angle of a tiny port face. The pipeline does the following for each of the three wrist cameras:

1. **Find the neighborhood.** A trained RGB model takes a smaller full-view image and the requested SC port code. It predicts roughly where that port is in the original image. A separate learned head supplies a rough orientation. These are search hints, not the final fine angle.
2. **Find straight segments in the original image.** The code reads the 1152×1024 image. OpenCV's Canny detector marks strong brightness boundaries; a Hough-line detector groups aligned boundary pixels into straight segments. This produces many candidates, including edges of cards, the gripper, and shadows.
3. **Discard unlikely segments.** A segment must be long (at least **240 pixels**) and its midpoint must be within **500 pixels** of the proposed port location. It must also point within **5°** of the direction expected for a board rail given the rough orientation and the camera view. At least **three** surviving segments are needed for one camera to return an angle. These rules do **not** prove that every surviving line is physically part of the board; they select lines that behave like its rails.
4. **Turn line slopes into board yaw.** Parallel 3D rails appear to converge in a camera image, like train tracks in a photograph. The expected image slope therefore depends on where a segment lies in the image. The fixed camera calibration and measured robot pose let the code predict that slope for different board yaw angles. It searches about **±4°** around the rough learned yaw and chooses the yaw that best matches each surviving segment. Longer segments get more weight; a median reduces the effect of outliers.
5. **Require agreement.** The line estimates inside each camera must be tightly grouped, and at least **two cameras** must agree within **1°**. The pipeline can then take the median of accepted estimates over time for the same stationary port. Finally, the board-to-port mounting angle, measured from training geometry, converts board yaw into port orientation.

```text
RGB + requested port code
         |
         v
rough port pixel and yaw from learned models
         |
         v
long straight image segments near that pixel
         |
         v
keep segments matching the calibrated board direction
         |
         v
agree across cameras and time -> board yaw -> port orientation
```

The key simplification is that **these SC scenes keep the board's other tilt angles and the port's mounting rotation fixed**. The image processing needs to measure mainly one variable: board yaw. If the board can tilt freely, its rails are hidden, or another object supplies convincing parallel lines, this method may fail. The current offline evaluator can fall back to its rough learned orientation when it lacks good rail evidence; a live handoff should instead treat weak evidence as a reason to **abstain**. We have not validated that live rejection behavior yet.

The reported **0.056° p95** was measured on selected, saved near-port development images. Simulator pose selected those offline test frames and scored the error afterward; it did **not** choose the lines or supply the orientation estimate. The method is therefore promising, but the result does not certify visibility in arbitrary live views or prove insertion success.

Code: [board-line detector and geometry](../../scripts/evaluate_sc_board_lines_yaw.py), [newer one-port replay](../../scripts/evaluate_sc_board_lines_oneport.py), [experiment results](../experiments/2026-10-03-geometric-handoff-repair.md).
