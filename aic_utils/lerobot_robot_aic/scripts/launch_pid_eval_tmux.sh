#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SESSION_NAME="${SESSION_NAME:-aic_pid_eval}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_DIR_DEFAULT}}"
ENGINE_CONFIG_FILE="${ENGINE_CONFIG_FILE:-${WORKSPACE_DIR}/outputs/configs/sfp2nic_only_trials.yaml}"
POLICY_CLASS="${POLICY_CLASS:-aic_example_policies.ros.CheatCodePIDController}"
SIM_DISTROBOX_NAME="${SIM_DISTROBOX_NAME:-aic_eval}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_DIR}/outputs/pid_eval_logs}"
AUTO_ATTACH="${AUTO_ATTACH:-true}"
PLOT_ON_EXIT="${PLOT_ON_EXIT:-true}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${WORKSPACE_DIR}/plot_pid_debug_logs.py}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Launch tmux windows for:
1) eval simulation container
2) CheatCode PID policy with log capture
3) PID plot generation after policy exit

This mirrors launch_policy_recording_tmux.sh more closely, but swaps the
recording window for PID eval logging and plotting to support gain tuning.

Options:
  --session-name NAME       tmux session name (default: ${SESSION_NAME})
  --workspace-dir PATH      workspace root containing pixi.toml (default: ${WORKSPACE_DIR})
  --engine-config PATH      aic_engine config yaml (default: ${ENGINE_CONFIG_FILE})
  --policy-class CLASS      policy class path (default: ${POLICY_CLASS})
  --sim-distrobox NAME      distrobox name for simulation window (default: ${SIM_DISTROBOX_NAME})
  --log-root PATH           root directory for captured logs and plots (default: ${LOG_ROOT})
  --plot-script PATH        PID plot helper script (default: ${PLOT_SCRIPT})
  --no-plot                 skip automatic plot generation when policy exits
  --no-attach               do not auto-attach to tmux session
  -h, --help                show this help text

Environment variable equivalents are also supported:
  SESSION_NAME, WORKSPACE_DIR, ENGINE_CONFIG_FILE, POLICY_CLASS,
  SIM_DISTROBOX_NAME, LOG_ROOT, AUTO_ATTACH, PLOT_ON_EXIT, PLOT_SCRIPT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --workspace-dir)
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    --engine-config)
      ENGINE_CONFIG_FILE="$2"
      shift 2
      ;;
    --policy-class)
      POLICY_CLASS="$2"
      shift 2
      ;;
    --sim-distrobox)
      SIM_DISTROBOX_NAME="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --plot-script)
      PLOT_SCRIPT="$2"
      shift 2
      ;;
    --no-plot)
      PLOT_ON_EXIT="false"
      shift
      ;;
    --no-attach)
      AUTO_ATTACH="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is required but not installed." >&2
  exit 1
fi

if ! command -v distrobox >/dev/null 2>&1; then
  echo "Error: distrobox is required but not installed." >&2
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "Error: pixi is required but not installed." >&2
  exit 1
fi

if [[ ! -d "${WORKSPACE_DIR}" ]]; then
  echo "Error: workspace directory does not exist: ${WORKSPACE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ENGINE_CONFIG_FILE}" ]]; then
  echo "Error: engine config file does not exist: ${ENGINE_CONFIG_FILE}" >&2
  echo "Hint: generate one with:" >&2
  echo "  python generate_random_trials_config.py --output ./outputs/configs/random_trials_10.yaml --num_trials 10 --seed 2026" >&2
  echo "  or use the fixed-pose PID eval config at ./outputs/configs/sfp2nic_only_trials.yaml" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Error: tmux session '${SESSION_NAME}' already exists." >&2
  echo "Either attach to it with: tmux attach -t ${SESSION_NAME}" >&2
  echo "Or use a different name via --session-name." >&2
  exit 1
fi

if [[ "${PLOT_ON_EXIT}" == "true" && ! -f "${PLOT_SCRIPT}" ]]; then
  echo "Error: PID plot helper script does not exist: ${PLOT_SCRIPT}" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_LOG_DIR}"

SIM_LOG="${RUN_LOG_DIR}/simulation.log"
POLICY_LOG="${RUN_LOG_DIR}/policy.log"
PLOT_LOG="${RUN_LOG_DIR}/plot.log"
PLOT_OUTPUT="${RUN_LOG_DIR}/pid_debug_plot.png"
POLICY_STATUS_FILE="${RUN_LOG_DIR}/policy.exit_code"

printf -v WORKSPACE_DIR_Q '%q' "${WORKSPACE_DIR}"
printf -v ENGINE_CONFIG_FILE_Q '%q' "${ENGINE_CONFIG_FILE}"
printf -v SIM_DISTROBOX_NAME_Q '%q' "${SIM_DISTROBOX_NAME}"
printf -v SIM_LOG_Q '%q' "${SIM_LOG}"
printf -v POLICY_LOG_Q '%q' "${POLICY_LOG}"
printf -v PLOT_LOG_Q '%q' "${PLOT_LOG}"
printf -v PLOT_OUTPUT_Q '%q' "${PLOT_OUTPUT}"
printf -v POLICY_STATUS_FILE_Q '%q' "${POLICY_STATUS_FILE}"
printf -v POLICY_SOURCE_ROOT_Q '%q' "${WORKSPACE_DIR}/aic_example_policies"
printf -v POLICY_CLASS_Q '%q' "${POLICY_CLASS}"
printf -v POLICY_RUNNER_Q '%q' "${WORKSPACE_DIR}/aic_utils/lerobot_robot_aic/scripts/run_aic_policy_from_source.py"
printf -v PLOT_SCRIPT_Q '%q' "${PLOT_SCRIPT}"

SIM_CMD="/entrypoint.sh ground_truth:=true start_aic_engine:=true aic_engine_config_file:=${ENGINE_CONFIG_FILE_Q} shutdown_on_aic_engine_exit:=true"
#SIM_CMD="/entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=false launch_rviz:=false aic_engine_config_file:=${ENGINE_CONFIG_FILE_Q} shutdown_on_aic_engine_exit:=true"
SIM_CMD_IN_CONTAINER="export DBX_CONTAINER_MANAGER=docker && distrobox enter -r ${SIM_DISTROBOX_NAME_Q} -- bash -lc \"cd ${WORKSPACE_DIR_Q} && ${SIM_CMD}\""
POLICY_CMD="pixi run --frozen python ${POLICY_RUNNER_Q} --policy-source-root ${POLICY_SOURCE_ROOT_Q} --policy-class ${POLICY_CLASS_Q}"
PLOT_CMD="pixi run --frozen python ${PLOT_SCRIPT_Q} --log-file ${POLICY_LOG_Q} --output ${PLOT_OUTPUT_Q}"

SIM_WINDOW_CMD="cd ${WORKSPACE_DIR_Q} && bash -lc 'set -o pipefail; ${SIM_CMD_IN_CONTAINER} 2>&1 | tee ${SIM_LOG_Q}'"
POLICY_WINDOW_CMD="cd ${WORKSPACE_DIR_Q} && bash -lc 'set -o pipefail; ${POLICY_CMD} 2>&1 | tee ${POLICY_LOG_Q}; policy_status=\${PIPESTATUS[0]}; printf \"%s\n\" \"\${policy_status}\" > ${POLICY_STATUS_FILE_Q}; exit \${policy_status}'"
PLOT_WINDOW_CMD="cd ${WORKSPACE_DIR_Q} && bash -lc 'set -o pipefail; while [[ ! -f ${POLICY_STATUS_FILE_Q} ]]; do sleep 1; done; ${PLOT_CMD} 2>&1 | tee ${PLOT_LOG_Q}; plot_status=\${PIPESTATUS[0]}; policy_status=\$(cat ${POLICY_STATUS_FILE_Q}); if [[ \${plot_status} -ne 0 ]]; then exit \${plot_status}; fi; exit \${policy_status}'"

tmux new-session -d -s "${SESSION_NAME}" -n simulation
tmux send-keys -t "${SESSION_NAME}:simulation" "${SIM_WINDOW_CMD}" C-m

tmux new-window -t "${SESSION_NAME}" -n policy
tmux send-keys -t "${SESSION_NAME}:policy" "${POLICY_WINDOW_CMD}" C-m

if [[ "${PLOT_ON_EXIT}" == "true" ]]; then
  tmux new-window -t "${SESSION_NAME}" -n plot
  tmux send-keys -t "${SESSION_NAME}:plot" "${PLOT_WINDOW_CMD}" C-m
fi

tmux select-window -t "${SESSION_NAME}:simulation"

if [[ "${PLOT_ON_EXIT}" == "true" ]]; then
  echo "Launched tmux session '${SESSION_NAME}' with windows: simulation, policy, plot."
else
  echo "Launched tmux session '${SESSION_NAME}' with windows: simulation, policy."
fi
echo "Run logs:"
echo "  simulation: ${SIM_LOG}"
echo "  policy: ${POLICY_LOG}"
if [[ "${PLOT_ON_EXIT}" == "true" ]]; then
  echo "  plot log: ${PLOT_LOG}"
  echo "  plot: ${PLOT_OUTPUT}"
fi
echo "Attach with: tmux attach -t ${SESSION_NAME}"

if [[ "${AUTO_ATTACH}" == "true" ]]; then
  tmux attach -t "${SESSION_NAME}"
fi
