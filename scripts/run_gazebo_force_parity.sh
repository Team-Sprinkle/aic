#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_FILE_DEFAULT="${ROOT_DIR}/scripts/force_parity_config.env"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_FILE_DEFAULT}}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

OUT_DIR_REL="${OUT_DIR:-outputs/force_parity}"
OUT_DIR="${OUT_DIR_ABS:-${ROOT_DIR}/${OUT_DIR_REL}}"
CSV_OUT="${CSV_OUT:-${OUT_DIR}/gazebo_force.csv}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
ENGINE_CONFIG_FILE_REL="${ENGINE_CONFIG_FILE:-outputs/configs/fixed_1_trials_sfp2nic.yaml}"
ENGINE_CONFIG_FILE="${ENGINE_CONFIG_FILE_ABS:-${ROOT_DIR}/${ENGINE_CONFIG_FILE_REL}}"
SIM_DISTROBOX_NAME="${SIM_DISTROBOX_NAME:-aic_eval}"
SHUTDOWN_ON_ENGINE_EXIT="${SHUTDOWN_ON_ENGINE_EXIT:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"

MISALIGN_X_M="${MISALIGN_X_M:-0.004}"
MISALIGN_Y_M="${MISALIGN_Y_M:-0.000}"
DURATION_S="${DURATION_S:-20.0}"
GROUND_TRUTH="${GROUND_TRUTH:-true}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

if ! command -v distrobox >/dev/null 2>&1; then
  echo "Error: distrobox is required to launch eval-style sim environment." >&2
  exit 1
fi

if [[ ! -f "${ENGINE_CONFIG_FILE}" ]]; then
  echo "Error: engine config file not found: ${ENGINE_CONFIG_FILE}" >&2
  echo "Set ENGINE_CONFIG_FILE to an existing trials config." >&2
  exit 1
fi

SIM_CMD="/entrypoint.sh ground_truth:=${GROUND_TRUTH} start_aic_engine:=true aic_engine_config_file:=${ENGINE_CONFIG_FILE} shutdown_on_aic_engine_exit:=${SHUTDOWN_ON_ENGINE_EXIT} gazebo_gui:=${GAZEBO_GUI} launch_rviz:=${LAUNCH_RVIZ}"

echo "[gazebo] launching eval-style sim via distrobox '${SIM_DISTROBOX_NAME}'"
export DBX_CONTAINER_MANAGER="${DBX_CONTAINER_MANAGER:-docker}"
distrobox enter -r "${SIM_DISTROBOX_NAME}" -- bash -lc "cd \"${ROOT_DIR}\" && ${SIM_CMD}" \
  >"${LOG_DIR}/gazebo_bringup.log" 2>&1 &
PIDS+=("$!")

sleep 8

echo "[gazebo] starting policy CheatCodeModified with misalignment x=${MISALIGN_X_M}, y=${MISALIGN_Y_M}"
AIC_CHEATCODE_MODIFIED_MISALIGN_X_M="${MISALIGN_X_M}" \
AIC_CHEATCODE_MODIFIED_MISALIGN_Y_M="${MISALIGN_Y_M}" \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCodeModified \
  >"${LOG_DIR}/gazebo_policy.log" 2>&1 &
PIDS+=("$!")

sleep 3

echo "[gazebo] logging /observations to ${CSV_OUT} for ${DURATION_S}s"
pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/gazebo_force_logger.py \
  --out "${CSV_OUT}" \
  --duration-s "${DURATION_S}"

echo "[gazebo] done. csv=${CSV_OUT}"
