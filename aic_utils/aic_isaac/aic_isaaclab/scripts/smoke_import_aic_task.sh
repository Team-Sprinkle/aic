#!/usr/bin/env bash
set -euo pipefail

cd "${ISAACLAB_ROOT:-/workspace/isaaclab}"
./isaaclab.sh -p -c "import gymnasium as gym; import aic_task; import aic_task.tasks; print('aic_task import OK'); print([spec.id for spec in gym.registry.values() if 'AIC' in spec.id])"
