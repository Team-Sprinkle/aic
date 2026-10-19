# Backoff follow-up

I’ve queued a matched comparison after the current fixes and evaluations finish:

1. Backoff disabled.
2. Local backoff.
3. Hierarchical backoff with earlier checkpoints and fresh visual reacquisition.

The same frozen BC model and four development starts will be used in all three modes (12rollouts). We’ll check full insertion, forces, repeated retreats and motion smoothness, and record videos. Active jobs remain untouched.

Backoff results are pending. It may help contact-induced misalignment, but it cannot by itself fix a consistently wrong pose estimate.

[Detailed protocol](../experiments/2026-10-17-insertion-control-and-grasp-repair.md#queued-executable-backoff-screen).
