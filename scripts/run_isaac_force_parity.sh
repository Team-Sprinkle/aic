#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEFAULT_ISAACLAB_ROOT}}"
AIC_REPO_IN_ISAAC="${AIC_REPO_IN_ISAAC:-${ISAACLAB_ROOT}/aic}"
CONTAINER_PROFILE="${CONTAINER_PROFILE:-base}"
DOCKER_NAME_SUFFIX="${DOCKER_NAME_SUFFIX:-}"
if [[ -n "${DOCKER_NAME_SUFFIX}" ]]; then
  CONTAINER_NAME_DEFAULT="isaac-lab-${CONTAINER_PROFILE}-${DOCKER_NAME_SUFFIX}"
else
  CONTAINER_NAME_DEFAULT="isaac-lab-${CONTAINER_PROFILE}"
fi
CONTAINER_NAME="${CONTAINER_NAME:-${CONTAINER_NAME_DEFAULT}}"
ISAACLAB_ROOT_IN_CONTAINER="${ISAACLAB_ROOT_IN_CONTAINER:-/workspace/isaaclab}"
CONFIG_FILE_DEFAULT="${AIC_REPO_IN_ISAAC}/scripts/force_parity_config.env"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_FILE_DEFAULT}}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

# README requires running Isaac commands from inside the Isaac Lab Docker container.
if [[ ! -f "/.dockerenv" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command not found on host." >&2
    exit 1
  fi
  if [[ "$(docker container inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "running" ]]; then
    echo "Error: container '${CONTAINER_NAME}' is not running." >&2
    echo "Start it with: ./docker/container.py start ${CONTAINER_PROFILE}" >&2
    exit 1
  fi
  CONTAINER_SCRIPT_PATH="${ISAACLAB_ROOT_IN_CONTAINER}/aic/scripts/run_isaac_force_parity.sh"
  CONTAINER_COMPARE_PATH="${ISAACLAB_ROOT_IN_CONTAINER}/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py"
  CONTAINER_CSV_OUT="${CONTAINER_CSV_OUT:-${ISAACLAB_ROOT_IN_CONTAINER}/aic/outputs/force_parity/isaac_force.csv}"
  HOST_CSV_OUT="${HOST_CSV_OUT:-${ISAACLAB_ROOT}/aic/outputs/force_parity/isaac_force.csv}"
  HOST_HEADLESS_DEFAULT="${HOST_HEADLESS_DEFAULT:-true}"
  docker cp "${SCRIPT_DIR}/run_isaac_force_parity.sh" "${CONTAINER_NAME}:${CONTAINER_SCRIPT_PATH}"
  docker cp "${ISAACLAB_ROOT}/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py" "${CONTAINER_NAME}:${CONTAINER_COMPARE_PATH}"
  echo "[host] running parity script inside container '${CONTAINER_NAME}'"
  docker exec -i "${CONTAINER_NAME}" bash -lc "cd '${ISAACLAB_ROOT_IN_CONTAINER}' && chmod +x '${CONTAINER_SCRIPT_PATH}' && CSV_OUT='${CONTAINER_CSV_OUT}' FORCE_PARITY_HEADLESS='${HEADLESS:-${HOST_HEADLESS_DEFAULT}}' bash '${CONTAINER_SCRIPT_PATH}'"
  status=$?
  if [[ ${status} -ne 0 ]]; then
    exit ${status}
  fi
  mkdir -p "$(dirname "${HOST_CSV_OUT}")"
  docker cp "${CONTAINER_NAME}:${CONTAINER_CSV_OUT}" "${HOST_CSV_OUT}"
  echo "[host] copied csv to ${HOST_CSV_OUT}"
  exit 0
fi

# Some container environments export ISAACLAB_ROOT=/root/IsaacLab, but Isaac Lab is mounted at /workspace/isaaclab.
if [[ -f "/.dockerenv" && ! -d "${ISAACLAB_ROOT}" && -d "${ISAACLAB_ROOT_IN_CONTAINER}" ]]; then
  ISAACLAB_ROOT="${ISAACLAB_ROOT_IN_CONTAINER}"
fi

if [[ ! -d "${AIC_REPO_IN_ISAAC}" ]]; then
  AIC_REPO_IN_ISAAC="${ISAACLAB_ROOT}/aic"
fi

if [[ ! -d "${ISAACLAB_ROOT}" ]]; then
  echo "Error: ISAACLAB_ROOT does not exist: ${ISAACLAB_ROOT}" >&2
  exit 1
fi
cd "${ISAACLAB_ROOT}"

# Isaac Sim expects these user directories to exist inside the container.
DOCKER_USER_HOME_EFFECTIVE="${DOCKER_USER_HOME:-${HOME:-/workspace}}"
mkdir -p \
  "${DOCKER_USER_HOME_EFFECTIVE}/Documents" \
  "${DOCKER_USER_HOME_EFFECTIVE}/.cache/ov" \
  "${DOCKER_USER_HOME_EFFECTIVE}/.cache/pip" \
  "${DOCKER_USER_HOME_EFFECTIVE}/.cache/nvidia/GLCache" \
  "${DOCKER_USER_HOME_EFFECTIVE}/.nvidia-omniverse/logs" \
  "/isaac-sim/kit/cache" \
  "/isaac-sim/kit/logs/Kit/Isaac-Sim" || true

OUT_DIR_REL="${OUT_DIR:-outputs/force_parity}"
OUT_DIR="${OUT_DIR_ABS:-${AIC_REPO_IN_ISAAC}/${OUT_DIR_REL}}"
CSV_OUT="${CSV_OUT:-${OUT_DIR}/isaac_force.csv}"

TASK="${TASK:-AIC-Task-v0}"
MISALIGN_Y_M="${MISALIGN_Y_M:-0.000}"
DESCEND_SPEED_MPS="${DESCEND_SPEED_MPS:-0.0009}"
SETTLE_SECONDS="${SETTLE_SECONDS:-2.0}"
ALIGN_SECONDS="${ALIGN_SECONDS:-1.0}"
DESCEND_SECONDS="${DESCEND_SECONDS:-6.0}"
HOLD_SECONDS="${HOLD_SECONDS:-2.0}"
TOTAL_SECONDS="${TOTAL_SECONDS:-12.0}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${FORCE_PARITY_HEADLESS:-${HEADLESS:-false}}"

mkdir -p "${OUT_DIR}"

if [[ -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  ISAACLAB_LAUNCHER="./isaaclab.sh"
elif [[ -x "${ISAACLAB_ROOT}/isaaclab" ]]; then
  ISAACLAB_LAUNCHER="./isaaclab"
elif command -v isaaclab >/dev/null 2>&1; then
  ISAACLAB_LAUNCHER="isaaclab"
else
  echo "Error: could not find Isaac Lab launcher. Expected '${ISAACLAB_ROOT}/isaaclab.sh', '${ISAACLAB_ROOT}/isaaclab', or 'isaaclab' on PATH." >&2
  exit 1
fi

CMD=(
  "${ISAACLAB_LAUNCHER}" -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py
  isaac-log
  --task "${TASK}"
  --out "${CSV_OUT}"
  --seconds "${TOTAL_SECONDS}"
  --settle-seconds "${SETTLE_SECONDS}"
  --align-seconds "${ALIGN_SECONDS}"
  --descend-seconds "${DESCEND_SECONDS}"
  --hold-seconds "${HOLD_SECONDS}"
  --misalign-y-m "${MISALIGN_Y_M}"
  --descend-speed-mps "${DESCEND_SPEED_MPS}"
  --device "${DEVICE}"
)

if [[ "${HEADLESS}" == "true" ]]; then
  CMD+=(--headless)
fi

# AppLauncher reads HEADLESS env var as int, so avoid "true/false" strings.
if [[ "${HEADLESS}" == "true" ]]; then
  APP_LAUNCHER_HEADLESS_ENV=1
else
  APP_LAUNCHER_HEADLESS_ENV=0
fi

echo "[isaac] running parity log, csv=${CSV_OUT}"
cd "${ISAACLAB_ROOT}"
echo "[isaac] root: ${ISAACLAB_ROOT}"
echo "[isaac] command: ${CMD[*]}"
HEADLESS="${APP_LAUNCHER_HEADLESS_ENV}" "${CMD[@]}"
echo "[isaac] done. csv=${CSV_OUT}"
