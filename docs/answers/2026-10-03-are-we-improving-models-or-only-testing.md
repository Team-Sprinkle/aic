# Are we improving the models, or only testing them?

**We are doing both.** The tests identify a specific failure, and the next training run is meant to repair that failure. In the latest round we actually trained two new perception models:

- **SC orientation:** A model reading original-resolution port crops improved held-out near-port orientation error from **4.62° to 3.70° p95**. We then trained a spatial-feature variant to address the remaining error; it scored **3.95°**, so that attempted repair did not help.
- **SFP port identity:** A model reading an original-resolution crop containing the two neighboring ports was trained on **19 native-image episodes**. On two five-episode validation halves, it accepted **66/87** and **147/159** correct constructed three-camera candidates, compared with **41/87** and **71/159** for the prior low-resolution verifier. It still accepted one wrong candidate triplet in each half.

The oracle-centering experiment was **diagnostic only**: it moved the crop using scored coordinates after the fact to test whether a bad crop location explained SC errors. It did not produce a deployable model or provide ground truth to one.

What we **did not** improve in this round was the transport actor, insertion actor, or RL policy. The agreed decision tree required SC orientation p95 at most **3°** and zero known wrong accepted candidates before attaching the actor. The first crop model and the one bounded repair both missed that gate, so I followed its stop rule. That means the recent work produced real perception gains, but **not yet a safer complete robot policy or more autonomous insertions**.

The next model change should follow the observed failure: train a head to locate several corners or edges of the **requested port face** in the original camera views, then fit its full 3D orientation using the calibrated cameras. This retains spatial geometry that the pooled crop features may discard. For SFP, add a cross-camera candidate-consistency and **invalid/abstain** decision so one confidently wrong view cannot approve a port match. Train these on episode-grouped data, inspect the labels, and compare them with the frozen models on new grouped development scenes. If they pass the same safety and latency gate, continue to the integrated handoff and insertion actor. More generic crop training or more rollouts without a repair hypothesis would be testing activity, not a useful improvement loop.

See the [full experiment record](../experiments/2026-10-03-native-crop-handoff-gate.md) and [tracked machine summary](../experiments/2026-10-03-native-crop-handoff-gate.summary.json).
