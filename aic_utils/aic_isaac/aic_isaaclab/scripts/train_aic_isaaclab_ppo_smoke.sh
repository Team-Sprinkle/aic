#!/usr/bin/env bash
set -euo pipefail

cd "${ISAACLAB_ROOT:-/workspace/isaaclab}"

TASK_ID="${TASK_ID:-AIC-Task-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-stage5_ppo_smoke_camera}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/isaaclab/aic/outputs/train/isaac_stage5_smoke}"

export AIC_ISAAC_RANDOMIZATION_PROFILE="${AIC_ISAAC_RANDOMIZATION_PROFILE:-none}"
export AIC_ISAAC_DISABLE_CAMERAS="${AIC_ISAAC_DISABLE_CAMERAS:-0}"
export AIC_ISAAC_OUTPUT_DIR="$OUTPUT_DIR"

./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task "$TASK_ID" \
  --headless \
  --enable_cameras \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --run_name "$RUN_NAME"
