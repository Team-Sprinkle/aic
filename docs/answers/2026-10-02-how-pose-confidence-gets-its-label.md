# How pose confidence gets a training label

**Yes: for a simple confidence model, each training example can have a binary 0/1 label.** The label is calculated **after the pose model makes its prediction**. It answers whether *that particular prediction* is correct enough for insertion. I was imprecise when I called a visibly confusing scene “low confidence” without specifying the model's prediction.

Suppose the task requests SC port 1. Both ports are clearly visible, so the per-camera **visibility labels for port 1 can all be 1**. Now compare two predictions on that same observation:

| Pose model output | Confidence training label |
| --- | ---: |
| Predicts **port 1** with lateral error 0.2 mm and acceptable axial/orientation error | **1** |
| Predicts the visually similar **port 0**, 23 mm away | **0** |
| Predicts port 1 but its lateral error is 1.4 mm, beyond a preset 0.5 mm acceptance limit | **0** |

So **the confidence target is conditioned on the model's actual output**. A clear image is not automatically label 1, and a confusing-looking image is not automatically label 0. If the model handles a difficult image correctly, its example gets label 1. Across many similar ambiguous scenes, it may make errors more often; a calibrated confidence predictor should then assign lower probabilities to cases whose observable cues resemble its failures.

One possible fixed definition is:

$$y_{\mathrm{safe}} = \mathbf{1}[\text{correct requested-port identity} \land e_{\mathrm{lateral}}\leq 0.5\text{ mm} \land e_{\mathrm{axial}}\leq a \land e_{\mathrm{rotation}}\leq r],$$

where $a$ and $r$ are chosen **before** calibration to match the handoff requirement. The ground-truth port pose is used offline to calculate the errors and this 0/1 label; it is never an input to the deployed model.

The confidence predictor then takes information available at deployment: the **predicted pose**, visual features, per-view visibility predictions, view agreement, selected task identity, and perhaps a short stability history. It outputs a probability such as 0.8. That means “among comparable predictions, about 80% should satisfy the fixed acceptance rule,” **not** “the simulator gave this frame an 0.8 confidence label.” We test that interpretation on held-out complete scenes.

I would train the pose estimator first, freeze it, and generate its predictions on scene groups **not used to fit that pose checkpoint** (or use grouped cross-validation). Then I would train/calibrate the confidence head against those prediction-specific 0/1 labels. This matters because a pose model can appear almost perfect on its own training images, giving the confidence head misleadingly easy examples. If the selected port cannot be identified in the current images and there is no previously verified tracked pose, the runtime result should be **invalid**, regardless of the raw quality score.

Visibility labels are different: they are properties of the **requested opening in each image**, assessed from occlusion/identifiability, independent of whether this particular pose prediction happened to be right. Thus the same row can have visibility `[1,1,1]` and confidence label 0 after a wrong-port prediction.
