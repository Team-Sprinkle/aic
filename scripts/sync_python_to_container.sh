#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEFAULT_ISAACLAB_ROOT}}"

CONTAINER_PROFILE="${CONTAINER_PROFILE:-base}"
DOCKER_NAME_SUFFIX="${DOCKER_NAME_SUFFIX:-}"
if [[ -n "${DOCKER_NAME_SUFFIX}" ]]; then
  CONTAINER_NAME_DEFAULT="isaac-lab-${CONTAINER_PROFILE}-${DOCKER_NAME_SUFFIX}"
else
  CONTAINER_NAME_DEFAULT="isaac-lab-${CONTAINER_PROFILE}"
fi
CONTAINER_NAME="${CONTAINER_NAME:-${CONTAINER_NAME_DEFAULT}}"
ISAACLAB_ROOT_IN_CONTAINER="${ISAACLAB_ROOT_IN_CONTAINER:-/workspace/isaaclab}"

HOST_PY_PATH="${HOST_PY_PATH:-${ISAACLAB_ROOT}/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py}"
CONTAINER_PY_PATH="${CONTAINER_PY_PATH:-${ISAACLAB_ROOT_IN_CONTAINER}/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found on host." >&2
  exit 1
fi

if [[ ! -f "${HOST_PY_PATH}" ]]; then
  echo "Error: host python file not found: ${HOST_PY_PATH}" >&2
  exit 1
fi

if [[ "$(docker container inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "running" ]]; then
  echo "Error: container '${CONTAINER_NAME}' is not running." >&2
  echo "Start it with: ./docker/container.py start ${CONTAINER_PROFILE}" >&2
  exit 1
fi

echo "[sync] copying ${HOST_PY_PATH}"
echo "[sync]      into ${CONTAINER_NAME}:${CONTAINER_PY_PATH}"
docker cp "${HOST_PY_PATH}" "${CONTAINER_NAME}:${CONTAINER_PY_PATH}"
echo "[sync] done"
