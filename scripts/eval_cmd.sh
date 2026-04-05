#!/usr/bin/env bash
set -euo pipefail

export DBX_CONTAINER_MANAGER=docker
CONTAINER_NAME="${AIC_EVAL_CONTAINER_NAME:-aic_eval}"

if [ "$#" -eq 0 ]; then
  echo "error: no command provided"
  exit 1
fi

"$(dirname "$0")/eval_status.sh" >/dev/null
distrobox enter -r "$CONTAINER_NAME" -- "$@"
