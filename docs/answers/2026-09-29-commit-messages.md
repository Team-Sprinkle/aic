I renamed the six commits on `feat/hybrid-train` to say what each contains:

```text
e639cd7 Archive the September 29 commit summary
d97bdb9 Update project status, recovery limits, and output storage map
7bb5d4a Record five-card cable-clearance tests and propose visibility-gated two-actor handoff
7d8ffcb Record SC BC comparisons, teacher routes, and recovery failures on September 25
b165318 Record SC pose-label audits and shared-policy tests on September 24
197bae1 Add SC target-keypoint locator, shared BC tools, and hierarchical recovery prototype
```

**Visibility and confidence serve different purposes.** For each of the three cameras, visibility should say whether the *selected* port can be identified from that view; an optional visible-fraction value can describe how much of the opening is exposed. Confidence should estimate the chance that the final fused pose is accurate enough for local control. A port can be half covered but still yield a high-confidence pose. If only one camera sees it, the any-camera-visible flag is true, while the support count is one out of three; that is not 100% visibility across cameras. The handoff should require a stricter confidence threshold for single-view estimates.

This visibility/confidence handoff is **proposed**, not already trained or validated. The current SC model predicts task-conditioned keypoints and uses projected in-frame masks; those masks do not test occlusion or provide calibrated pose confidence. I clarified the proposed definitions in [POSE_VISIBILITY_HANDOFF.md](/data1/chmin/yj/ws_aic/src/aic/docs/POSE_VISIBILITY_HANDOFF.md).

Because these commits had already been pushed, changing their messages rewrote their hashes. The push used `--force-with-lease`, and a local backup of the old tip was kept.

[Saved response](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-09-29-commit-messages.md)
