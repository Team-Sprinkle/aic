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
GAZEBO_CSV="${GAZEBO_CSV:-${OUT_DIR}/gazebo_force.csv}"
ISAAC_CSV="${ISAAC_CSV:-${OUT_DIR}/isaac_force.csv}"
ALIGNED_CSV="${ALIGNED_CSV:-${OUT_DIR}/force_parity_aligned.csv}"
DT="${DT:-0.01}"

python aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py \
  compare \
  --gazebo "${GAZEBO_CSV}" \
  --isaac "${ISAAC_CSV}" \
  --dt "${DT}" \
  --out "${ALIGNED_CSV}"

echo "[compare] aligned csv: ${ALIGNED_CSV}"
