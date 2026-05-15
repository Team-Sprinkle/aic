# Submission Handoff

Generated: 2026-05-14 UTC

This bundle packages the current online SERL direct TCP-delta candidate and the ACT 175k TorchScript artifacts needed by its runtime.

## Current Recommendation

Do not submit the online SERL v44 checkpoint as the final competitive policy yet. It is runtime-compatible but scored only Tier 1 in the completed local official Gazebo trial:

`total: 1`, no insertion, final plug-port distance `0.13m`.

The bundle is still useful for handoff because it contains the exact runtime implementation, checkpoint, Dockerfile, compose override, logs, and verification commands. If a submission must be made immediately, the safer policy candidate is the ACT 175k TorchScript baseline, not this online SERL checkpoint.

## Selected SERL Candidate

- Policy checkpoint: `submission_handoff/artifacts/submission_candidate/online_serl/checkpoint_000300.pt`
- ACT base: `submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.pt`
- ACT normalizer/preprocessor: `submission_handoff/artifacts/submission_candidate/act_pretrained_model/`
- Runtime entrypoint: `aic_example_policies.ros.RunACTAdapterSERL`
- Dockerfile: `docker/aic_submission/Dockerfile`
- Compose override: `docker/docker-compose.submission.yaml`
- Device: CUDA preferred. Challenge cloud docs list one NVIDIA L4 with 24 GiB VRAM.

Runtime settings packaged in the Dockerfile/compose:

- `AIC_SERL_CONTROL_HZ=20`
- `AIC_SERL_N_ACTION_STEPS=4`
- `AIC_SERL_COMMAND_MODE=delta_pose`
- `AIC_SERL_COMMAND_FRAME=gripper/tcp`
- `AIC_SERL_MAX_TRANSLATION_DELTA=0.0007`
- `AIC_SERL_MAX_ROTATION_DELTA=0.0001`

## Verification Results

Host CUDA smoke check passed:

`submission_handoff/logs/host_cuda_smoke_check.log`

Docker-image local compose eval completed all three official sample trials:

`submission_handoff/local_compose_eval_results_20260514_203808/aic_results/scoring.yaml`

Key result:

```yaml
total: 59.560568704817861
trial_1: tier_1=1, tier_2=13.899054545454545, tier_3=16.737781840558927
trial_2: tier_1=1, tier_2=0, tier_3=0
trial_3: tier_1=1, tier_2=18.945419912812273, tier_3=6.9783124059921233
```

All three trials passed model validation and completed without engine failure. No trial detected insertion.

Earlier direct official Gazebo one-trial eval completed:

`submission_handoff/gazebo_eval_v44_official/runtime_eval_serl_direct_v44_trial1_completed/checkpoint_000300/`

Key result:

```yaml
total: 1
tier_1: Model validation succeeded.
tier_2: 0
tier_3: 0
message: "No insertion detected. Final plug port distance: 0.13m."
```

Recorded rollout bag:

`submission_handoff/gazebo_eval_v44_official/runtime_eval_serl_direct_v44_trial1_completed/checkpoint_000300/bag_trial_1_20260514_191811_949/`

Partial three-trial eval:

`submission_handoff/gazebo_eval_v44_official/runtime_eval_serl_direct_v44_retry2/checkpoint_000300/`

It completed trial 1 and recorded trial 2, but the wrapper timeout fired before the full three-trial `scoring.yaml` was written.

## Build Locally

From repo root:

```bash
docker build -f docker/aic_submission/Dockerfile -t aic-submission-candidate:local .
```

Then smoke check inside the image:

```bash
docker run --rm --gpus all --entrypoint /bin/bash aic-submission-candidate:local \
  -lc 'cd /ws_aic/src/aic && pixi run --as-is python scripts/submission_smoke_check.py --checkpoint /opt/aic_policy/submission_candidate/online_serl/checkpoint_000300.pt --act-torchscript /opt/aic_policy/submission_candidate/act_policy_ts_175000_cuda0.pt --device cuda --n-action-steps 4'
```

Or run:

```bash
bash submission_handoff/build_and_verify.sh
```

## Local Compose Eval

After the image builds:

```bash
docker compose -f docker/docker-compose.submission.yaml up --abort-on-container-exit
```

The compose file uses placeholder passwords only. Do not commit or share real credentials in this repo.

## Files To Give The Submission Colleague

- `docker/aic_submission/Dockerfile`
- `docker/docker-compose.submission.yaml`
- `scripts/submission_smoke_check.py`
- `submission_handoff/`
- Runtime source changes currently in the working tree:
  - `aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`
  - `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`

## ECR Push Template

Use credentials provided out-of-band. Do not put secrets in this repo.

```bash
aws configure --profile <team_name>
export AWS_PROFILE=<team_name>
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
docker tag aic-submission-candidate:local 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>
```

Portal image URI:

```text
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>
```

## Known Limitations

- Online SERL v44 is not insertion-capable in the completed Gazebo eval.
- Docker image build and in-image CUDA smoke check pass.
- Docker image local compose eval completes all three official sample trials, but the score is not competitive because there is no insertion.
- A prior direct full three-trial eval did not finish before the wrapper timeout; the later Docker compose eval did finish all three trials.
