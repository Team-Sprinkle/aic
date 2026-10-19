# Evaluation scope reduced

Changed the full scenario sweep to **one episode per scenario: 62 episodes total**. The current integration rollout finishes before the updated scheduler resumes. No models, rules or scene settings change.

The separate complete `sample_config.yaml` and released `eval_config.yaml` evaluations remain **three runs each**.

The original plan and manifest are preserved. Results will report the smaller sample size explicitly; one episode per scenario is a coverage check, not evidence of high reliability.
