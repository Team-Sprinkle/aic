#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
TASK_ID="${TASK_ID:-AIC-Task-v0}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_EPISODES="${NUM_EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-4096}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is required, e.g. CHECKPOINT=/workspace/isaaclab/aic/outputs/.../model_200.pt" >&2
  exit 2
fi

export AIC_ISAAC_DISABLE_CAMERAS="${AIC_ISAAC_DISABLE_CAMERAS:-0}"

cd "${ISAACLAB_ROOT}"

args=(
  -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/eval.py
  --task "${TASK_ID}"
  --headless
  --num_envs "${NUM_ENVS}"
  --num_episodes "${NUM_EPISODES}"
  --max_steps "${MAX_STEPS}"
  --seed "${SEED}"
  --checkpoint "${CHECKPOINT}"
  --enable_cameras
)

if [[ "${VIDEO}" == "1" || "${VIDEO}" == "true" ]]; then
  args+=(--video --video_length "${VIDEO_LENGTH}")
fi

./isaaclab.sh "${args[@]}"
