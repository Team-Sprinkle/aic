# Selected Policy Summary

Generated: 2026-05-14 UTC

## Candidate Table

| Artifact | Step | Type | Runtime entrypoint | Normalizer/preprocessor | Device | Evidence |
|---|---:|---|---|---|---|---|
| `submission_handoff/artifacts/submission_candidate/online_serl/checkpoint_000300.pt` | 300 | online SERL direct TCP delta | `aic_example_policies.ros.RunACTAdapterSERL` | ACT 175k TorchScript sidecar + `act_pretrained_model` normalizer | CUDA preferred | Host CUDA smoke passed. Official Gazebo trial 1 completed, score `1`, no insertion, final distance `0.13m`. |
| `submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.pt` | 175000 | ACT TorchScript | `aic_example_policies.ros.RunACTTorchScript` | `act_pretrained_model` normalizer | CUDA preferred | Known baseline artifact. Not re-scored in this handoff pass. |
| `submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cpu.pt` | 175000 | ACT TorchScript CPU fallback | `aic_example_policies.ros.RunACTTorchScript` | `act_pretrained_model` normalizer | CPU | Fallback artifact only. |

## Selection

The packaged Dockerfile currently selects the online SERL direct checkpoint:

`submission_handoff/artifacts/submission_candidate/online_serl/checkpoint_000300.pt`

This is useful as a runtime-compatibility handoff because it verifies the new direct SERL runtime path, including `n_action_steps=4`, chunk size 8, 20 Hz command cadence, CUDA TorchScript ACT base loading, and official Gazebo lifecycle execution.

It is not a strong performance submission candidate. Official Gazebo trial 1 completed with only Tier 1 model validation score:

- total: `1`
- Tier 2: `0`
- Tier 3: `0`
- message: `No insertion detected. Final plug port distance: 0.13m.`

If a submission must be made immediately for reliability rather than SERL validation, use the ACT 175k TorchScript baseline instead and change the Docker CMD/entrypoint settings to `aic_example_policies.ros.RunACTTorchScript`.

## Official Eval Evidence

Completed one-trial official-style eval:

`submission_handoff/gazebo_eval_v44_official/runtime_eval_serl_direct_v44_trial1_completed/checkpoint_000300/`

Important files:

- `scoring.yaml`
- `eval_summary.json`
- `logs/engine.log`
- `logs/policy.log`
- `bag_trial_1_20260514_191811_949/`

Partial three-trial official eval:

`submission_handoff/gazebo_eval_v44_official/runtime_eval_serl_direct_v44_retry2/checkpoint_000300/`

It completed trial 1, recorded trial 2, and then hit the wrapper timeout before final `scoring.yaml` for all three trials.
