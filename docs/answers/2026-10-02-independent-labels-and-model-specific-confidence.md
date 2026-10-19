# Independent perception labels and model-specific confidence

**You are right.** The primary training labels should be properties of the scene, not of whichever checkpoint happens to make a prediction. My previous explanation made a model-specific **calibration target** sound like a permanent dataset label. That distinction matters.

## Keep the main labels independent

For each observation and requested target, store:

- The requested port's **true pose relative to the measured TCP**.
- Its projected image keypoints, where appropriate.
- Whether the **requested opening is actually identifiable in each camera**, including occlusion. This needs validated visibility labels; the current projected-in-frame flag is insufficient.
- A mask indicating whether a visual pose should be supervised for that observation. A completely hidden target should teach rejection, not force a precise pose guess from an image that contains no evidence.

These labels stay the same if we retrain or improve the estimator. Train the task-conditioned, selected-port pose and visibility outputs against them. We can also train a pose uncertainty distribution using a proper loss against the same true pose; that does not require adding a checkpoint-specific 0/1 column to the dataset.

## Confidence has a different meaning

If “confidence” means **the probability that this model's predicted pose is within the insertion tolerance**, that probability is inherently **model-specific**. A poor model and an improved model can see the same image and make different predictions. The correct/incorrect result of *their predictions* can differ. There is no fixed, model-independent 0/1 label for that question.

Therefore, the clean sequence is:

1. Train and choose a pose/visibility checkpoint using the independent ground-truth labels.
2. **Freeze that checkpoint.** On separate, episode-grouped development scenes, compare its predictions with ground truth. Fit or calibrate a quality score to estimate how often predictions with certain observable characteristics meet the fixed error limits.
3. Test the calibrated score on another untouched group of development scenes, including hidden targets and wrong-port cases. Choose the handoff threshold based on false acceptance and missed valid opportunities, then keep the final scenes sealed.
4. If we change the pose model, **recheck and usually recalibrate** confidence. The old calibration does not automatically apply to the improved model.

For example, the permanent dataset label for an image is “SC port 1 is here and visible in the left and center cameras.” Checkpoint A might predict the wrong SC port; checkpoint B might predict port 1 to 0.2 mm. The scene label is unchanged. Their **prediction errors** differ, and so does their model-specific reliability.

## Practical recommendation

I would not make a separate confidence classifier a prerequisite for the first selected-port experiment. Start with **independent pose and true-visibility labels**, train the selected-port estimator, and measure its held-out pose errors and false acceptance. Once the checkpoint is stable, calibrate a **model-specific handoff quality score** or use a conservative threshold based on held-out evidence. This avoids pretending that confidence is an intrinsic property of the image or a fixed label we can collect once and reuse forever.
