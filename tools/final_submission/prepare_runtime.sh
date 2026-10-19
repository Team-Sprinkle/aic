#!/usr/bin/env bash
# Install only inference dependencies; no ROS, training stack or datasets.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 /absolute/runtime-prefix /path/to/micromamba" >&2
  exit 2
fi
repo=$(cd "$(dirname "$0")/../.." && pwd)
prefix=$1
mamba=$2
[[ "$prefix" = /* ]] || { echo 'Runtime prefix must be absolute' >&2; exit 2; }
[[ ! -e "$prefix" ]] || { echo 'Choose a new runtime prefix' >&2; exit 2; }
"$mamba" create --yes --prefix "$prefix" --file "$repo/docker/final_controller/conda-linux-64.lock"
# The bootstrap Python needs pip >=22.3 (supports --python).
python3 -m pip --python "$prefix/bin/python" install --no-cache-dir --no-deps \
  -r "$repo/docker/final_controller/requirements.lock"
"$prefix/bin/python" -m pip check
