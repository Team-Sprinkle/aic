# Rebuild Submission Image Runbook

This document is the repeatable path for preparing a new submission image after
the model checkpoint or runtime policy code changes.

## Current Image Contract

- Local image name: `aic-submission-candidate:local`
- Dockerfile: `docker/aic_submission/Dockerfile`
- Compose file: `docker/docker-compose.submission.yaml`
- Runtime entrypoint: `aic_example_policies.ros.RunACTAdapterSERL`
- Runtime checkpoint path in image:
  `/opt/aic_policy/submission_candidate/online_serl/checkpoint_000300.pt`
- Runtime ACT TorchScript path in image:
  `/opt/aic_policy/submission_candidate/act_policy_ts_175000_cuda0.pt`
- Control settings:
  - `AIC_SERL_CONTROL_HZ=20`
  - `AIC_SERL_N_ACTION_STEPS=4`
  - `AIC_SERL_COMMAND_MODE=delta_pose`
  - `AIC_SERL_COMMAND_FRAME=gripper/tcp`

The Dockerfile intentionally copies only runtime code and
`submission_handoff/artifacts/submission_candidate`. It must not copy `outputs/`,
training datasets, debug logs, wandb runs, or S3-downloaded datasets.

## 1. Select The New Candidate

Choose the checkpoint and runtime artifacts to package. For an online SERL
checkpoint, record:

```bash
NEW_SERL_CKPT=/absolute/or/repo/relative/path/to/checkpoint.pt
NEW_ACT_TS=/absolute/or/repo/relative/path/to/act_policy_ts_175000_cuda0.pt
NEW_ACT_PRETRAINED=/absolute/or/repo/relative/path/to/checkpoints/175000/pretrained_model
```

Keep ACT 175k unless there is a specific reason to change it.

## 2. Update Stable Artifact Directory

From repo root:

```bash
cd /home/ubuntu/ws_aic/src/aic

mkdir -p submission_handoff/artifacts/submission_candidate/online_serl
mkdir -p submission_handoff/artifacts/submission_candidate/act_pretrained_model

cp "$NEW_SERL_CKPT" \
  submission_handoff/artifacts/submission_candidate/online_serl/checkpoint_000300.pt

cp "$NEW_ACT_TS" \
  submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.pt

cp "$NEW_ACT_PRETRAINED"/policy_preprocessor_step_3_normalizer_processor.safetensors \
  submission_handoff/artifacts/submission_candidate/act_pretrained_model/
```

If the ACT TorchScript sidecar JSON changes, update:

```text
submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.json
```

The JSON `checkpoint_dir` must point to the in-container stable path:

```json
"/opt/aic_policy/submission_candidate/act_pretrained_model"
```

The runtime only needs the TorchScript file, sidecar JSON, ACT normalizer
safetensors, and SERL checkpoint. Do not add datasets.

## 3. If Runtime Policy Code Changed

Keep runtime behavior in these packages:

- `aic_example_policies`
- `aic_model`
- `aic_interfaces`
- `aic_utils/gazebo_rl`
- `aic_utils/lerobot_robot_aic`

If a new policy entrypoint is used, update both:

```text
docker/aic_submission/Dockerfile
docker/docker-compose.submission.yaml
```

Specifically update the Docker `CMD` policy parameter and any `AIC_SERL_*` env
vars required by the new policy.

## 4. Build And Smoke Test

Run:

```bash
bash submission_handoff/build_and_verify.sh
```

Expected result:

```text
"status": "ok"
"action_shape": [4, 6]
"n_action_steps": 4
"state_dim": 82
```

The build script writes:

```text
submission_handoff/logs/docker_build.log
submission_handoff/logs/docker_smoke_check.log
submission_handoff/logs/git_status.txt
```

Check image size:

```bash
docker images aic-submission-candidate:local
docker image inspect aic-submission-candidate:local --format 'SizeBytes={{.Size}}'
```

Target displayed size is below about 15 GB. If the image grows unexpectedly:

```bash
docker history aic-submission-candidate:local --no-trunc
docker run --rm --entrypoint /bin/bash aic-submission-candidate:local \
  -lc 'du -sh /ws_aic/src/aic/.pixi /opt/aic_policy/submission_candidate; find /ws_aic/src/aic -maxdepth 4 -type d \( -name outputs -o -name hf_combined -o -name wandb -o -name datasets \) -print'
```

There should be no output for dataset/debug directories.

## 5. Run Local Official-Style Compose Eval

From repo root:

```bash
docker compose -f docker/docker-compose.submission.yaml down --remove-orphans

mkdir -p submission_handoff/logs

docker compose -f docker/docker-compose.submission.yaml up --abort-on-container-exit \
  2>&1 | tee submission_handoff/logs/local_compose_eval_$(date -u +%Y%m%d_%H%M%S).log
```

Before removing containers, copy eval results:

```bash
TS=$(date -u +%Y%m%d_%H%M%S)
OUT=submission_handoff/local_compose_eval_results_$TS
mkdir -p "$OUT"

docker cp aic-submission-local-eval-1:/root/aic_results "$OUT/aic_results"
sed -n '1,220p' "$OUT/aic_results/scoring.yaml"
```

Then clean up:

```bash
docker compose -f docker/docker-compose.submission.yaml down --remove-orphans
```

Success criteria:

- model container loads the intended policy
- model accepts the `InsertCable` goal
- all trials finish without `aic_engine` failure
- `scoring.yaml` is present

## 6. Refresh Handoff Metadata

Update:

```text
submission_handoff/README.md
submission_handoff/selected_policy_summary.md
submission_handoff/MANIFEST.txt
submission_handoff/checksums.txt
```

Regenerate checksums for packaged artifacts:

```bash
find submission_handoff/artifacts/submission_candidate -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > submission_handoff/checksums.txt
```

At minimum, record:

- selected checkpoint path
- ACT TorchScript path
- policy entrypoint
- image size
- smoke result
- local eval score
- whether insertion was detected
- known limitations

## 7. Push Handoff Commands

Do not put credentials in the repo. The colleague who submits should run:

```bash
aws configure --profile <team_name>
export AWS_PROFILE=<team_name>

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

docker tag aic-submission-candidate:local \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>

docker push \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>
```

Portal image URI:

```text
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:<unique_tag>
```

## 8. Current Verified Baseline

As of 2026-05-14, the packaged image:

- builds as `aic-submission-candidate:local`
- displays as `14.7GB`
- passes in-image CUDA smoke
- completes three local compose eval trials
- scores `59.560568704817861`
- does not achieve insertion

Current score file:

```text
submission_handoff/local_compose_eval_results_20260514_203808/aic_results/scoring.yaml
```
