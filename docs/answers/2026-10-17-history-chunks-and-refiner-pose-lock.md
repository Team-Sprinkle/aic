## Implemented and running

**History now survives transport → approach → insertion.** The runtime retains the recent measured robot states and re-expresses them in the accepted port frame. It still cancels pending actions at handoff. Reset and cross-episode checks pass.

The insertion model currently uses **1.2 seconds of robot/relative-pose history and current RGB images**. Preserving historical camera images as model inputs would require a separate architecture change.

### Chunk experiments

All steps remain50 ms:

| Model | Predict | Execution amounts being tested |
| --- | --- | --- |
|4-step baseline|200 ms|4 steps|
|8-step|400 ms|2or4 steps|
|12-step|600 ms|4 or 6 steps|
|8-step + continuity loss|400 ms|4 steps|

The four training variants use **60,671common samples** and identical episode splits. Longer labels come from recorded command targets in the same episode. The first two fits are running; the others and live comparisons follow automatically. Training uses validation-plateau checks, and caps will be reported explicitly.

The continuity loss compares overlapping plans at **the same future timestamps**, after aligning their coordinate frames. Full insertion and force behavior remain the success criteria—a smooth stationary policy would still fail.

## How the refiner's pose is used

**The port reference is fixed during an insertion attempt.**

1. During transport, coarse/refiner estimates update approximately every0.3 simulation seconds.
2. During acquisition, repeated estimates must agree.
3. The accepted port pose is locked for the approach and insertion.
4. Current TCP/plug position relative to that port updates from measured robot motion.
5. Large-retreat reacquisition can deliberately establish a new reference, after pending actions are cancelled.

So frequent refiner updates were **not causing the earlier insertion jerks**. The new acquisition branch additionally checks a 3–5 estimate window, rejecting position drift beyond 1 mm or orientation drift beyond 1 degree before locking.

## PoseInsert review

The checked port-frame → TCP target →body-delta transform chain is consistent. I also found and fixed a double-JPEG preprocessing discrepancy in grasp acquisition; its contribution to performance is not established.

However, this remains a **PoseInsert-inspired transformer**, rather than the paper's complete method. The [paper](https://arxiv.org/html/2505.09424v2) uses full connector-relative pose. Our actor receives connector-tip translation and TCP orientation, while the connector's rotation within the gripper is not an approved input. That is a real fine-alignment limitation to investigate separately; it does not explain all the large observed jerks.

No new-model improvement is claimed yet. The older25mm test achieved 4/4 near handoffs on two configurations, but 0/4 full insertions. New history/horizon results are still pending.

[Detailed plan, equations and artifacts](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-17-history-horizon-poseinsert-continuation.md) · [Saved answer](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-17-history-chunks-and-refiner-pose-lock.md).
