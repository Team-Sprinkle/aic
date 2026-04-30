#!/usr/bin/env bash
set -euo pipefail

cd "${ISAACLAB_ROOT:-/workspace/isaaclab}"

if ! ./isaaclab.sh -p -c "import isaaclab" >/dev/null 2>&1; then
  ./isaaclab.sh -p -m pip install --no-build-isolation flatdict==4.0.1
  ./isaaclab.sh -p -m pip install -e source/isaaclab
fi

./isaaclab.sh -p -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
./isaaclab.sh -p -c "import aic_task; print('aic_task import OK')"
