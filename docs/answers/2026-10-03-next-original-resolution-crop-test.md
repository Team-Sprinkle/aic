# The next test: can original image detail make the handoff reliable?

The immediate test is a **small perception experiment**, not another transport or insertion training run. We want to learn whether looking closely at the requested port in the original camera image fixes two remaining errors:

- **SC-to-SC:** the current model sometimes recognizes the right port but misjudges its angle. On new development scenes its near-port orientation error is **4.65° at the 95th percentile**, above our **3°** gate. It also accepted one saved stop where the tool was pointing about **6.7°** away from the required attitude.
- **SFP-to-NIC:** in one held-out configuration, the verifier sometimes confused the two neighboring ports on the same card. Their openings are only **22–24 pixels apart** after resizing the full image to 288×256.

## What will actually run

```text
Three full camera views + requested task/card/port code
    |
    v
Existing image-based locator proposes the requested port area
    |
    v
Cut a small patch from the ORIGINAL 1152×1024 camera image
    |
    v
Small crop model examines the opening, port edges, and nearby context
    |
    v
One requested-port pose/angle, plus visible-or-invalid decision
    |
    v
Compare its pose with the measured tool pose before allowing handoff
```

For SC, the first crop candidate is **160×160 original pixels** around the visually proposed port. For SFP, the crop must include **both neighboring openings and enough card context** to tell them apart; an initial **320×320 original-pixel** candidate will be checked on training images before its size is fixed. We will retain full-view features to know which card and task we are looking at. The model will predict **only the requested port**, or say that it cannot see it well enough. It will not predict every port in the scene.

The port's simulator pose can supply **training and scoring labels**. It must never choose the crop during evaluation or be passed to the actor. The requested card/port code will be an input, and training examples will include the same image with a **wrong neighboring-port request** and examples where the requested port is hidden or off-screen. Visibility is a separate output; we are **not adding a learned confidence label**. If the image is ambiguous, the handoff must wait.

## How we will make the comparison fair

1. Audit the existing original images and seven usable SC one-port training resets for board-angle coverage. Only if that audit finds a real coverage gap, collect up to **12 targeted training resets and two new development resets**. Do not collect routine episodes.
2. Split by complete reset configuration. Fit the crop head on training resets only; compare it with the current frozen orientation head on the **same held-out development resets and saved stops**. Keep all reserved final scenes sealed.
3. Measure SC orientation error, including its **95th percentile** and results for each reset. Check whether the known bad SC stop is rejected while clear valid stops are retained. For SFP, count mistakes between the two ports of the same card, correct-port retention, and errors when the requested port is absent. Review marked original crops for the worst cases.
4. Run the full live chain, including locating, cropping, crop inference, and the handoff decision, and record **p50/p95/p99 latency**. The p95 must stay below **300 ms**.

The fixed stopping rule is: independent near-port SC orientation p95 **≤3°**, **no known unsafe stop accepted**, **at least 95%** of clear valid stops retained, adjacent-SFP-port confusion resolved on held-out groups, and live p95 **<300 ms**. The saved SC sample is only five good stops and one bad stop, so passing that replay alone will not be treated as proof. A successful crop test still needs the complete handoff gate before the insertion actor can run autonomously.

This is capped at **four GPU-hours for fitting and replay** and **one day of image review**. If it fails, the failure images and per-scene errors will tell us whether the problem is choosing the wrong crop, missing visual detail, occlusion, or an inaccurate angle label. We will choose one specific repair from that evidence, rather than start RL or broadly retrain the actor.
