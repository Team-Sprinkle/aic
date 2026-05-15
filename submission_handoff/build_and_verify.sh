#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/submission_handoff/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
{
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git log -1 --oneline)"
  echo "status:"
  git status --short
} | tee "$LOG_DIR/git_status.txt"

test -f submission_handoff/artifacts/submission_candidate/online_serl/checkpoint_000300.pt
test -f submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.pt
test -f submission_handoff/artifacts/submission_candidate/act_policy_ts_175000_cuda0.json
test -f submission_handoff/artifacts/submission_candidate/act_pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors

docker build \
  -f docker/aic_submission/Dockerfile \
  -t aic-submission-candidate:local \
  . 2>&1 | tee "$LOG_DIR/docker_build.log"

docker run --rm \
  --gpus all \
  --entrypoint /bin/bash \
  aic-submission-candidate:local \
  -lc 'cd /ws_aic/src/aic && pixi run --as-is python scripts/submission_smoke_check.py --checkpoint /opt/aic_policy/submission_candidate/online_serl/checkpoint_000300.pt --act-torchscript /opt/aic_policy/submission_candidate/act_policy_ts_175000_cuda0.pt --device cuda --n-action-steps 4' \
  2>&1 | tee "$LOG_DIR/docker_smoke_check.log"

echo "Build and smoke check completed. Local eval, if desired:"
echo "docker compose -f docker/docker-compose.submission.yaml up --abort-on-container-exit"
