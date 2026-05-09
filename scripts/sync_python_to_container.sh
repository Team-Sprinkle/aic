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
LOCAL_AIC_SOURCE_ROOT="${LOCAL_AIC_SOURCE_ROOT:-${ISAACLAB_ROOT}/aic}"
LOCAL_AIC_MIRROR_ROOT="${LOCAL_AIC_MIRROR_ROOT:-/home/brucekimrok/projects/ws_aic/src/aic}"

SYNC_LOCAL_AIC_MIRROR="${SYNC_LOCAL_AIC_MIRROR:-1}"

DEFAULT_SYNC_REL_PATHS=(
  "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py"
  "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py"
  "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_eval_helpers.py"
  "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_analysis_from_csv.py"
  "aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py"
  "aic/outputs/configs/fixed_1_trials_sfp2nic.yaml"
)

# Optional override:
#   SYNC_REL_PATHS="path/one.py path/two.py"
if [[ -n "${SYNC_REL_PATHS:-}" ]]; then
  # shellcheck disable=SC2206
  SYNC_REL_PATHS_ARR=(${SYNC_REL_PATHS})
else
  SYNC_REL_PATHS_ARR=("${DEFAULT_SYNC_REL_PATHS[@]}")
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found on host." >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | rg -x "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Error: container not running (or wrong CONTAINER_NAME): ${CONTAINER_NAME}" >&2
  exit 1
fi

for rel_path in "${SYNC_REL_PATHS_ARR[@]}"; do
  host_path="${ISAACLAB_ROOT}/${rel_path}"
  container_path="${ISAACLAB_ROOT_IN_CONTAINER}/${rel_path}"

  if [[ ! -f "${host_path}" ]]; then
    echo "Error: host file not found: ${host_path}" >&2
    exit 1
  fi

  container_dir="$(dirname "${container_path}")"
  docker exec "${CONTAINER_NAME}" mkdir -p "${container_dir}"

  echo "[sync] copying ${host_path}"
  echo "[sync]      into ${CONTAINER_NAME}:${container_path}"
  docker cp "${host_path}" "${CONTAINER_NAME}:${container_path}"

  if [[ "${SYNC_LOCAL_AIC_MIRROR}" == "1" ]]; then
    if [[ "${host_path}" != "${LOCAL_AIC_SOURCE_ROOT}/"* ]]; then
      echo "Error: expected rel path under 'aic/' for local mirror sync: ${rel_path}" >&2
      exit 1
    fi
    mirror_rel_path="${host_path#${LOCAL_AIC_SOURCE_ROOT}/}"
    mirror_path="${LOCAL_AIC_MIRROR_ROOT}/${mirror_rel_path}"
    mkdir -p "$(dirname "${mirror_path}")"
    cp "${host_path}" "${mirror_path}"
    echo "[sync] mirrored ${host_path}"
    echo "[sync]      into ${mirror_path}"
  fi
done

echo "[sync] done"
