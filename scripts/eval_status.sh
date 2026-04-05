#!/usr/bin/env bash
set -euo pipefail

export DBX_CONTAINER_MANAGER=docker
CONTAINER_NAME="${AIC_EVAL_CONTAINER_NAME:-aic_eval}"

if ! command -v distrobox >/dev/null 2>&1; then
  echo "error: distrobox is not installed"
  exit 1
fi

if distrobox list --no-color | awk 'NR>1 {print $3}' | grep -qx "$CONTAINER_NAME"; then
  echo "ok: distrobox container '${CONTAINER_NAME}' is available via ${DBX_CONTAINER_MANAGER}"
  exit 0
else
  echo "error: distrobox container '${CONTAINER_NAME}' was not found via ${DBX_CONTAINER_MANAGER}"
  echo "hint: create it with scripts/eval_up.sh or the documented quickstart commands"
  exit 1
fi
