Training and the follow-on evaluation chain are running autonomously.

- **150 mm actor:** matched training uses 67,039 examples from 500 episodes / 377 reset configurations. Convergence checks can extend training when validation is still improving.
- **Perception:** SC/SFP refiner retraining is complete. The additional SC repair passed its first 10 live acquisitions; the 20-case comparison continues.
- **Recovery:** hierarchical backtracking and fresh reacquisition are implemented. Controlled comparisons follow the BC gate.
- **Checks:** fixed Docker setting forwarding and a handoff bug. State/frame audits passed, alongside 28 focused tests.

The intermediate actor pilots did **not** fully insert. Reliable insertion remains unproven; completed-model evaluations are queued. The limit of four GPUs is respected, and final scenes remain sealed.

[Live status](/data1/chmin/yj/ws_aic/src/aic/docs/STATUS.md) · [Saved response](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-16-full-approach-execution-progress.md)
