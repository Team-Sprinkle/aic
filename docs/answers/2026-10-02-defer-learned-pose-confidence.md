# Defer learned pose confidence

**Recommendation: do not train a separate confidence head in the next perception experiment.** It is not necessary to establish whether a selected-port pose estimator works, and a poorly calibrated confidence number could make the handoff appear safer than it is.

Train the model to answer two direct questions instead:

1. **Can the requested opening be identified in each camera?** This uses validated visibility/occlusion labels, not merely projected-in-frame labels.
2. **Where is that requested opening relative to the measured robot tool?** Supervise pose only when visual evidence supports it. If the requested opening is unavailable, return **no valid pose**, even if another port is visible.

At deployment, transport proposes a near-target pause. The local verifier checks selected-port identity and visibility, estimates pose from enough usable views, and tests camera agreement, short-window stability, measured waiting-pose distance/orientation, and force/contact. Only then may insertion begin. These checks are **not themselves proof of accuracy**—a wrong port once passed multi-view agreement—so the complete accept/reject rule must be measured on held-out wrong-port, occluded, and wrong-route scenes before promotion.

A learned, calibrated confidence score may help **later** if the selected-port model is usually accurate but some visually ambiguous cases still slip through a simple gate. Then we can freeze that model, measure its errors on separate scenes, and ask whether a quality score reduces dangerous false acceptances at an acceptable cost in missed valid handoffs. If it does not improve that tradeoff, omit it. Any new pose checkpoint requires a fresh calibration check.

This keeps the immediate experiment focused on the real open questions: **Does it identify the requested nearby opening, estimate its pose accurately, and know when it cannot see it?**
