#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SESSION_NAME="${SESSION_NAME:-aic_lerobot}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_DIR_DEFAULT}}"
ENGINE_CONFIG_FILE="${ENGINE_CONFIG_FILE:-${WORKSPACE_DIR}/outputs/configs/random_trials_10.yaml}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER:-local}/aic_mixed_dataset}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_DIR}/outputs/lerobot_datasets}"
DATASET_SINGLE_TASK="${DATASET_SINGLE_TASK:-Insert cable into target port}"
ACTION_MODE="${ACTION_MODE:-cartesian}"
MAX_EPISODES="${MAX_EPISODES:-}"
POLICY_CLASS="${POLICY_CLASS:-aic_example_policies.ros.CheatCode}"
SIM_DISTROBOX_NAME="${SIM_DISTROBOX_NAME:-aic_eval}"
AUTO_ATTACH="${AUTO_ATTACH:-true}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_DIR}/outputs/scores/$(basename "${DATASET_ROOT}")}"
PID_TUNING_PLOTS_DIR="${PID_TUNING_PLOTS_DIR:-${WORKSPACE_DIR}/outputs/pid_tuning_plots}"
REMOVE_BAG_DATA="${REMOVE_BAG_DATA:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Launch three tmux windows for:
1) simulation
2) policy
3) policy recorder

Options:
  --session-name NAME       tmux session name (default: ${SESSION_NAME})
  --workspace-dir PATH      workspace root containing pixi.toml (default: ${WORKSPACE_DIR})
  --engine-config PATH      aic_engine config yaml (default: ${ENGINE_CONFIG_FILE})
  --policy-class CLASS      policy class path (default: ${POLICY_CLASS})
  --sim-distrobox NAME      distrobox name for simulation window (default: ${SIM_DISTROBOX_NAME})
  --dataset-repo-id ID      LeRobot dataset repo id (default: ${DATASET_REPO_ID})
  --dataset-root PATH       LeRobot dataset root (default: ${DATASET_ROOT})
  --dataset-single-task TXT Dataset task prompt (default: "${DATASET_SINGLE_TASK}")
  --action-mode MODE        recorder action mode (default: ${ACTION_MODE})
  --max-episodes N          recorder max episodes (default: auto from config trials count)
  --results-root PATH       root directory for scoring outputs (default: ${RESULTS_ROOT})
  --pid-plots-dir PATH      directory for PID tuning plots (default: ${PID_TUNING_PLOTS_DIR})
  --remove-bag-data BOOL    remove per-trial bag_* dirs after run (default: ${REMOVE_BAG_DATA})
  --gazebo-gui BOOL         pass gazebo_gui:=true/false to /entrypoint.sh (default: ${GAZEBO_GUI})
  --launch-rviz BOOL        pass launch_rviz:=true/false to /entrypoint.sh (default: ${LAUNCH_RVIZ})
  --no-attach               do not auto-attach to tmux session
  -h, --help                show this help text

Environment variable equivalents are also supported:
  SESSION_NAME, WORKSPACE_DIR, ENGINE_CONFIG_FILE, POLICY_CLASS,
  SIM_DISTROBOX_NAME, DATASET_REPO_ID, DATASET_ROOT, DATASET_SINGLE_TASK,
  ACTION_MODE, MAX_EPISODES, RESULTS_ROOT, REMOVE_BAG_DATA, GAZEBO_GUI,
  LAUNCH_RVIZ, PID_TUNING_PLOTS_DIR
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
    --dataset-repo-id)
      DATASET_REPO_ID="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset-single-task)
      DATASET_SINGLE_TASK="$2"
      shift 2
      ;;
    --action-mode)
      ACTION_MODE="$2"
      shift 2
      ;;
    --max-episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --pid-plots-dir)
      PID_TUNING_PLOTS_DIR="$2"
      shift 2
      ;;
    --remove-bag-data)
      REMOVE_BAG_DATA="$2"
      shift 2
      ;;
    --gazebo-gui)
      GAZEBO_GUI="$2"
      shift 2
      ;;
    --launch-rviz)
      LAUNCH_RVIZ="$2"
      shift 2
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

count_trials_in_engine_config() {
  local config_path="$1"
  cd "${WORKSPACE_DIR}"
  pixi run python - "${config_path}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("Engine config must be a YAML map.")
trials = data.get("trials")
if not isinstance(trials, dict):
    raise SystemExit("Engine config missing YAML map key: 'trials'.")
print(len(trials))
PY
}

bool_or_die() {
  local value="$1"
  local name="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "Error: ${name} must be 'true' or 'false' (got '${value}')." >&2
    exit 1
  fi
}

bool_or_die "${REMOVE_BAG_DATA}" "--remove-bag-data"
bool_or_die "${GAZEBO_GUI}" "--gazebo-gui"
bool_or_die "${LAUNCH_RVIZ}" "--launch-rviz"

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
  exit 1
fi

if [[ -z "${MAX_EPISODES}" ]]; then
  MAX_EPISODES="$(count_trials_in_engine_config "${ENGINE_CONFIG_FILE}")"
  if ! [[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to determine trial count from config: ${ENGINE_CONFIG_FILE}" >&2
    exit 1
  fi
  if [[ "${MAX_EPISODES}" -le 0 ]]; then
    echo "Error: config has no trials. Cannot auto-set max episodes." >&2
    exit 1
  fi
  echo "Auto-set recorder max episodes to ${MAX_EPISODES} from config trials."
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Error: tmux session '${SESSION_NAME}' already exists." >&2
  echo "Either attach to it with: tmux attach -t ${SESSION_NAME}" >&2
  echo "Or use a different name via --session-name." >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}"
mkdir -p "${PID_TUNING_PLOTS_DIR}"

SCORE_SUMMARY_CSV="${RESULTS_ROOT}/score_summary.csv"
SCORING_YAML="${RESULTS_ROOT}/scoring.yaml"

# Write post-simulation scoring script (runs on host after sim exits)
POST_SIM_SCRIPT="${RESULTS_ROOT}/post_sim.sh"
cat > "${POST_SIM_SCRIPT}" <<POSTSIM
#!/usr/bin/env bash
set -uo pipefail
echo
echo "Simulation exited. Generating score summary..."

SCORING_YAML="${SCORING_YAML}"
SCORE_SUMMARY_CSV="${SCORE_SUMMARY_CSV}"
RESULTS_ROOT="${RESULTS_ROOT}"
REMOVE_BAG_DATA="${REMOVE_BAG_DATA}"
WORKSPACE_DIR="${WORKSPACE_DIR}"

if [[ ! -f "\${SCORING_YAML}" ]]; then
  echo "Warning: scoring.yaml not found at \${SCORING_YAML}"
  echo "Engine may not have completed successfully."
  exit 0
fi

cd "\${WORKSPACE_DIR}"
pixi run python - "\${SCORING_YAML}" "\${SCORE_SUMMARY_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

scoring_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

data = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    print("Error: scoring.yaml is not a YAML map.", file=sys.stderr)
    sys.exit(1)

grand_total = data.get("total", 0.0)
trial_keys = [k for k in data if k != "total"]
trial_keys.sort()

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run_index", "trial_id", "status", "total_score", "scoring_yaml"])
    for idx, trial_id in enumerate(trial_keys, start=1):
        trial = data[trial_id]
        t1 = trial.get("tier_1", {}).get("score", 0.0)
        t2 = trial.get("tier_2", {}).get("score", 0.0)
        t3 = trial.get("tier_3", {}).get("score", 0.0)
        total = t1 + t2 + t3
        writer.writerow([idx, trial_id, "OK", total, str(scoring_path.resolve())])

print()
print(f"Score summary ({len(trial_keys)} trials, grand total: {grand_total})")
print(f"{'idx':<5} {'trial_id':<30} {'tier1':>6} {'tier2':>8} {'tier3':>8} {'total':>10}")
print("-" * 70)
for idx, trial_id in enumerate(trial_keys, start=1):
    trial = data[trial_id]
    t1 = trial.get("tier_1", {}).get("score", 0.0)
    t2 = trial.get("tier_2", {}).get("score", 0.0)
    t3 = trial.get("tier_3", {}).get("score", 0.0)
    total = t1 + t2 + t3
    print(f"{idx:<5} {trial_id:<30} {t1:>6.1f} {t2:>8.2f} {t3:>8.1f} {total:>10.2f}")
print("-" * 70)
print(f"{'':>51} grand total: {grand_total:.2f}")
print()
print(f"CSV written to: {csv_path}")
PY

# Clean up bag directories if requested
if [[ "\${REMOVE_BAG_DATA}" == "true" && -d "\${RESULTS_ROOT}" ]]; then
  removed=0
  shopt -s nullglob
  for bag_dir in "\${RESULTS_ROOT}"/bag_*; do
    if [[ -d "\${bag_dir}" ]]; then
      rm -rf "\${bag_dir}"
      removed=\$((removed + 1))
    fi
  done
  shopt -u nullglob
  if [[ "\${removed}" -gt 0 ]]; then
    echo "Removed \${removed} bag dir(s) from \${RESULTS_ROOT}"
  fi
fi
POSTSIM
chmod +x "${POST_SIM_SCRIPT}"

SIM_CMD="/entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=${GAZEBO_GUI} launch_rviz:=${LAUNCH_RVIZ} aic_engine_config_file:=${ENGINE_CONFIG_FILE} shutdown_on_aic_engine_exit:=true"
SIM_CMD_IN_CONTAINER="export DBX_CONTAINER_MANAGER=docker && distrobox enter ${SIM_DISTROBOX_NAME} -- bash -lc 'cd \"${WORKSPACE_DIR}\" && export AIC_RESULTS_DIR=\"${RESULTS_ROOT}\" && ${SIM_CMD}'"
POLICY_CMD="export AIC_PID_TUNING_PLOTS_DIR=\"${PID_TUNING_PLOTS_DIR}\" && pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=${POLICY_CLASS}"
RECORDER_CMD="pixi run aic-policy-recorder --dataset.repo_id=${DATASET_REPO_ID} --dataset.single_task=\"${DATASET_SINGLE_TASK}\" --dataset.root=${DATASET_ROOT} --dataset.fps=30 --action_mode=${ACTION_MODE} --max_episodes=${MAX_EPISODES} --dataset.push_to_hub"

tmux new-session -d -s "${SESSION_NAME}" -n simulation
tmux send-keys -t "${SESSION_NAME}:simulation" "cd \"${WORKSPACE_DIR}\" && ${SIM_CMD_IN_CONTAINER}; bash \"${POST_SIM_SCRIPT}\"" C-m

tmux new-window -t "${SESSION_NAME}" -n policy
tmux send-keys -t "${SESSION_NAME}:policy" "cd \"${WORKSPACE_DIR}\" && ${POLICY_CMD}" C-m

tmux new-window -t "${SESSION_NAME}" -n recorder
tmux send-keys -t "${SESSION_NAME}:recorder" "cd \"${WORKSPACE_DIR}\" && ${RECORDER_CMD}" C-m

tmux select-window -t "${SESSION_NAME}:simulation"

echo "Launched tmux session '${SESSION_NAME}' with windows: simulation, policy, recorder."
echo "  scoring results dir: ${RESULTS_ROOT}"
echo "  PID tuning plots dir: ${PID_TUNING_PLOTS_DIR}"
echo "  scoring yaml (after sim exits): ${SCORING_YAML}"
echo "  score summary csv (after sim exits): ${SCORE_SUMMARY_CSV}"
echo "  remove bag data: ${REMOVE_BAG_DATA}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"

if [[ "${AUTO_ATTACH}" == "true" ]]; then
  tmux attach -t "${SESSION_NAME}"
fi
