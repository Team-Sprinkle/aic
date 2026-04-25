#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_DIR_DEFAULT}}"
ENGINE_CONFIG_FILE="${ENGINE_CONFIG_FILE:-${WORKSPACE_DIR}/outputs/configs/random_trials_10.yaml}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER:-local}/aic_mixed_dataset}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_DIR}/outputs/lerobot_datasets}"
DATASET_SINGLE_TASK="${DATASET_SINGLE_TASK:-Insert cable into target port}"
ACTION_MODE="${ACTION_MODE:-cartesian}"
POLICY_CLASS="${POLICY_CLASS:-aic_example_policies.ros.CheatCode}"
AIC_OFFICIAL_TEACHER_TRAJECTORY="${AIC_OFFICIAL_TEACHER_TRAJECTORY:-}"
AIC_OFFICIAL_TEACHER_ACTION_MODE="${AIC_OFFICIAL_TEACHER_ACTION_MODE:-relative_delta_gripper_tcp}"
SIM_DISTROBOX_NAME="${SIM_DISTROBOX_NAME:-aic_eval_0415}"
SAVE_FAILED_EPISODES="${SAVE_FAILED_EPISODES:-false}"
PER_TRIAL_TIMEOUT_SEC="${PER_TRIAL_TIMEOUT_SEC:-0}"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-8}"
PAUSE_BETWEEN_TRIALS_SEC="${PAUSE_BETWEEN_TRIALS_SEC:-3}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
TMP_DIR="${TMP_DIR:-}"
RECORDER_DRAIN_SEC="${RECORDER_DRAIN_SEC:-120}"
REQUIRE_RECORDER_SAVE_LOG="${REQUIRE_RECORDER_SAVE_LOG:-false}"
SUDO_KEEPALIVE="${SUDO_KEEPALIVE:-false}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_DIR}/outputs/aic_results_per_trial}"
REMOVE_BAG_DATA="${REMOVE_BAG_DATA:-true}"

usage() {
  cat <<EOF_USAGE
Usage: $(basename "$0") [options]

Run autonomous rollout recording one trial at a time, fully restarting:
- simulation (/entrypoint.sh)
- policy node
- aic-policy-recorder

for every trial in --engine-config.

Options:
  --workspace-dir PATH           Workspace root containing pixi.toml (default: ${WORKSPACE_DIR})
  --engine-config PATH           Multi-trial engine config YAML (default: ${ENGINE_CONFIG_FILE})
  --policy-class CLASS           Policy class path (default: ${POLICY_CLASS})
  --teacher-trajectory PATH      Smooth trajectory JSON for
                                 aic_teacher_official.OfficialTeacherReplay
                                 (sets AIC_OFFICIAL_TEACHER_TRAJECTORY)
  --teacher-action-mode MODE     Replay action mode:
                                 relative_delta_gripper_tcp or
                                 absolute_cartesian_pose_base_link
                                 (default: ${AIC_OFFICIAL_TEACHER_ACTION_MODE})
  --sim-distrobox NAME           Distrobox name for simulation (default: ${SIM_DISTROBOX_NAME})
  --dataset-repo-id ID           LeRobot dataset repo id (default: ${DATASET_REPO_ID})
  --dataset-root PATH            LeRobot dataset root (default: ${DATASET_ROOT})
  --dataset-single-task TXT      Dataset task prompt (default: "${DATASET_SINGLE_TASK}")
  --action-mode MODE             Recorder action mode (default: ${ACTION_MODE})
  --save-failed-episodes BOOL    true/false passed to recorder (default: ${SAVE_FAILED_EPISODES})
  --per-trial-timeout-sec N      Kill trial if it exceeds N seconds (0 disables timeout)
                                 (default: ${PER_TRIAL_TIMEOUT_SEC})
  --startup-delay-sec N          Delay after simulation start before policy/recorder (default: ${STARTUP_DELAY_SEC})
  --pause-between-trials-sec N   Sleep between trials (default: ${PAUSE_BETWEEN_TRIALS_SEC})
  --continue-on-failure BOOL     Continue remaining trials after failure (default: ${CONTINUE_ON_FAILURE})
  --push-to-hub BOOL             Pass through to recorder (default: ${PUSH_TO_HUB})
  --tmp-dir PATH                 Directory for generated one-trial YAML files and logs
  --recorder-drain-sec N         Extra wait after sim exits for recorder save/finalize
                                 before forcing teardown (default: ${RECORDER_DRAIN_SEC})
  --require-recorder-save-log BOOL
                                 Strict mode: require recorder log to contain
                                 "Episode saved" before trial is marked OK
                                 (default: ${REQUIRE_RECORDER_SAVE_LOG})
  --sudo-keepalive BOOL          Run one-time sudo auth + keepalive loop
                                 during this script run (default: ${SUDO_KEEPALIVE})
  --gazebo-gui BOOL              Pass gazebo_gui:=true/false to /entrypoint.sh
                                 (default: ${GAZEBO_GUI})
  --launch-rviz BOOL             Pass launch_rviz:=true/false to /entrypoint.sh
                                 (default: ${LAUNCH_RVIZ})
  --results-root PATH            Root directory for per-trial AIC scoring outputs
                                 (default: ${RESULTS_ROOT})
  --remove-bag-data BOOL         Remove per-trial scoring bag_* dirs after each trial
                                 (default: ${REMOVE_BAG_DATA})
  -h, --help                     Show this help text

Environment variable equivalents:
  WORKSPACE_DIR, ENGINE_CONFIG_FILE, POLICY_CLASS,
  AIC_OFFICIAL_TEACHER_TRAJECTORY, AIC_OFFICIAL_TEACHER_ACTION_MODE,
  SIM_DISTROBOX_NAME,
  DATASET_REPO_ID, DATASET_ROOT, DATASET_SINGLE_TASK, ACTION_MODE,
  SAVE_FAILED_EPISODES, PER_TRIAL_TIMEOUT_SEC, STARTUP_DELAY_SEC,
  PAUSE_BETWEEN_TRIALS_SEC, CONTINUE_ON_FAILURE, PUSH_TO_HUB, TMP_DIR,
  RECORDER_DRAIN_SEC, REQUIRE_RECORDER_SAVE_LOG, SUDO_KEEPALIVE,
  GAZEBO_GUI, LAUNCH_RVIZ, RESULTS_ROOT, REMOVE_BAG_DATA
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-dir) WORKSPACE_DIR="$2"; shift 2 ;;
    --engine-config) ENGINE_CONFIG_FILE="$2"; shift 2 ;;
    --policy-class) POLICY_CLASS="$2"; shift 2 ;;
    --teacher-trajectory) AIC_OFFICIAL_TEACHER_TRAJECTORY="$2"; shift 2 ;;
    --teacher-action-mode) AIC_OFFICIAL_TEACHER_ACTION_MODE="$2"; shift 2 ;;
    --sim-distrobox) SIM_DISTROBOX_NAME="$2"; shift 2 ;;
    --dataset-repo-id) DATASET_REPO_ID="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --dataset-single-task) DATASET_SINGLE_TASK="$2"; shift 2 ;;
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --save-failed-episodes) SAVE_FAILED_EPISODES="$2"; shift 2 ;;
    --per-trial-timeout-sec) PER_TRIAL_TIMEOUT_SEC="$2"; shift 2 ;;
    --startup-delay-sec) STARTUP_DELAY_SEC="$2"; shift 2 ;;
    --pause-between-trials-sec) PAUSE_BETWEEN_TRIALS_SEC="$2"; shift 2 ;;
    --continue-on-failure) CONTINUE_ON_FAILURE="$2"; shift 2 ;;
    --push-to-hub) PUSH_TO_HUB="$2"; shift 2 ;;
    --tmp-dir) TMP_DIR="$2"; shift 2 ;;
    --recorder-drain-sec) RECORDER_DRAIN_SEC="$2"; shift 2 ;;
    --require-recorder-save-log) REQUIRE_RECORDER_SAVE_LOG="$2"; shift 2 ;;
    --sudo-keepalive) SUDO_KEEPALIVE="$2"; shift 2 ;;
    --gazebo-gui) GAZEBO_GUI="$2"; shift 2 ;;
    --launch-rviz) LAUNCH_RVIZ="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --remove-bag-data) REMOVE_BAG_DATA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

bool_or_die() {
  local value="$1"
  local name="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "Error: ${name} must be 'true' or 'false' (got '${value}')." >&2
    exit 1
  fi
}

int_or_die() {
  local value="$1"
  local name="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "Error: ${name} must be a non-negative integer (got '${value}')." >&2
    exit 1
  fi
}

for cmd in distrobox pixi; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Error: ${cmd} is required but not installed." >&2
    exit 1
  fi
done

if [[ ! -d "${WORKSPACE_DIR}" ]]; then
  echo "Error: workspace directory does not exist: ${WORKSPACE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ENGINE_CONFIG_FILE}" ]]; then
  echo "Error: engine config file does not exist: ${ENGINE_CONFIG_FILE}" >&2
  exit 1
fi
mkdir -p "${RESULTS_ROOT}"

bool_or_die "${SAVE_FAILED_EPISODES}" "--save-failed-episodes"
bool_or_die "${CONTINUE_ON_FAILURE}" "--continue-on-failure"
bool_or_die "${PUSH_TO_HUB}" "--push-to-hub"
bool_or_die "${REQUIRE_RECORDER_SAVE_LOG}" "--require-recorder-save-log"
bool_or_die "${SUDO_KEEPALIVE}" "--sudo-keepalive"
bool_or_die "${GAZEBO_GUI}" "--gazebo-gui"
bool_or_die "${LAUNCH_RVIZ}" "--launch-rviz"
bool_or_die "${REMOVE_BAG_DATA}" "--remove-bag-data"
int_or_die "${PER_TRIAL_TIMEOUT_SEC}" "--per-trial-timeout-sec"
int_or_die "${STARTUP_DELAY_SEC}" "--startup-delay-sec"
int_or_die "${PAUSE_BETWEEN_TRIALS_SEC}" "--pause-between-trials-sec"
int_or_die "${RECORDER_DRAIN_SEC}" "--recorder-drain-sec"

if [[ -z "${TMP_DIR}" ]]; then
  TMP_DIR="$(mktemp -d -t aic_policy_per_trial.XXXXXX)"
  CLEAN_TMP_DIR="true"
else
  mkdir -p "${TMP_DIR}"
  CLEAN_TMP_DIR="false"
fi

cleanup() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]] && kill -0 "${SUDO_KEEPALIVE_PID}" >/dev/null 2>&1; then
    kill "${SUDO_KEEPALIVE_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${CLEAN_TMP_DIR}" == "true" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

start_sudo_keepalive() {
  if [[ "${SUDO_KEEPALIVE}" != "true" ]]; then
    return
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Error: --sudo-keepalive=true requires 'sudo'." >&2
    exit 1
  fi

  echo "Running one-time sudo authentication for this run..."
  sudo -v

  local parent_pid="$$"
  (
    while true; do
      sudo -n true >/dev/null 2>&1 || exit
      sleep 50
      kill -0 "${parent_pid}" >/dev/null 2>&1 || exit
    done
  ) &
  SUDO_KEEPALIVE_PID=$!
}

list_trial_ids() {
  local config_path="$1"
  cd "${WORKSPACE_DIR}"
  pixi run python - "${config_path}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("Engine config must be a YAML map.")
trials = config.get("trials")
if not isinstance(trials, dict) or not trials:
    raise SystemExit("Engine config must contain a non-empty YAML map key: 'trials'.")
for trial_id in trials.keys():
    print(trial_id)
PY
}

write_single_trial_config() {
  local src_config="$1"
  local trial_id="$2"
  local out_config="$3"
  cd "${WORKSPACE_DIR}"
  pixi run python - "${src_config}" "${trial_id}" "${out_config}" <<'PY'
import sys
from pathlib import Path
import yaml

src_path = Path(sys.argv[1])
trial_id = sys.argv[2]
out_path = Path(sys.argv[3])

config = yaml.safe_load(src_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("Engine config must be a YAML map.")
trials = config.get("trials")
if not isinstance(trials, dict):
    raise SystemExit("Engine config missing key: 'trials'.")
if trial_id not in trials:
    raise SystemExit(f"Trial '{trial_id}' not found in source config.")

single = dict(config)
single["trials"] = {trial_id: trials[trial_id]}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(single, sort_keys=False), encoding="utf-8")
PY
}

terminate_process() {
  local pid="$1"
  local name="$2"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -INT "${pid}" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name} did not exit after SIGINT, sending SIGTERM..."
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}

cleanup_stale_sim_router() {
  # echo "  preflight: cleaning stale rmw_zenohd in distrobox '${SIM_DISTROBOX_NAME}'..."
  # Best-effort host cleanup in case rmw_zenohd is bound in host namespace.
  pkill -f rmw_zenohd >/dev/null 2>&1 || true
  pkill -f "rmw_zenoh_cpp rmw_zenohd" >/dev/null 2>&1 || true

  sleep 10

  # local attempt
  # for attempt in {1..5}; do
  #   if (
  #     export DBX_CONTAINER_MANAGER=docker
  #     distrobox enter -r "${SIM_DISTROBOX_NAME}" -- bash -lc "pkill -f rmw_zenohd >/dev/null 2>&1 || true; pkill -f 'rmw_zenoh_cpp rmw_zenohd' >/dev/null 2>&1 || true"
  #   ); then
  #     return 0
  #   fi
  #   echo "  preflight: distrobox cleanup attempt ${attempt}/5 failed; retrying..."
  #   sleep 2
  # done

  # echo "  preflight: WARNING unable to run distrobox cleanup after retries; stale router may remain."
}

cleanup_trial_bags() {
  local trial_results_dir="$1"
  if [[ "${REMOVE_BAG_DATA}" != "true" ]]; then
    return
  fi
  if [[ ! -d "${trial_results_dir}" ]]; then
    return
  fi

  local removed=0
  local bag_dir
  shopt -s nullglob
  for bag_dir in "${trial_results_dir}"/bag_*; do
    if [[ -d "${bag_dir}" ]]; then
      rm -rf "${bag_dir}"
      removed=$((removed + 1))
    fi
  done
  shopt -u nullglob

  if [[ "${removed}" -gt 0 ]]; then
    echo "  removed ${removed} bag dir(s) from ${trial_results_dir}"
  else
    echo "  no bag dirs to remove in ${trial_results_dir}"
  fi
}

mapfile -t TRIAL_IDS < <(list_trial_ids "${ENGINE_CONFIG_FILE}")
TOTAL_TRIALS="${#TRIAL_IDS[@]}"
if [[ "${TOTAL_TRIALS}" -eq 0 ]]; then
  echo "Error: no trials found in config ${ENGINE_CONFIG_FILE}" >&2
  exit 1
fi

start_sudo_keepalive

echo "Starting per-trial autonomous recording run"
echo "  workspace: ${WORKSPACE_DIR}"
echo "  engine config: ${ENGINE_CONFIG_FILE}"
echo "  total trials: ${TOTAL_TRIALS}"
echo "  output dataset: ${DATASET_ROOT} (${DATASET_REPO_ID})"
echo "  temp dir: ${TMP_DIR}"
echo "  recorder drain after sim exit: ${RECORDER_DRAIN_SEC}s"
echo "  strict save-log check: ${REQUIRE_RECORDER_SAVE_LOG}"
echo "  sudo keepalive: ${SUDO_KEEPALIVE}"
  echo "  gazebo gui: ${GAZEBO_GUI}"
  if [[ -n "${AIC_OFFICIAL_TEACHER_TRAJECTORY}" ]]; then
    echo "  teacher trajectory: ${AIC_OFFICIAL_TEACHER_TRAJECTORY}"
    echo "  teacher action mode: ${AIC_OFFICIAL_TEACHER_ACTION_MODE}"
  fi
echo "  launch rviz: ${LAUNCH_RVIZ}"
echo "  per-trial scoring results root: ${RESULTS_ROOT}"
echo "  remove bag data: ${REMOVE_BAG_DATA}"

DATASET_EXISTS_BEFORE_RUN="false"
if [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
  DATASET_EXISTS_BEFORE_RUN="true"
  echo "  dataset root already exists, first trial will use --dataset.resume"
fi

FAILURES=0
RUN_INDEX=0
SCORE_SUMMARY_CSV="${RESULTS_ROOT}/score_summary.csv"
echo "run_index,trial_id,status,total_score,scoring_yaml" > "${SCORE_SUMMARY_CSV}"

for TRIAL_ID in "${TRIAL_IDS[@]}"; do
  RUN_INDEX=$((RUN_INDEX + 1))

  SINGLE_CONFIG_PATH="${TMP_DIR}/engine_${RUN_INDEX}_${TRIAL_ID}.yaml"
  write_single_trial_config "${ENGINE_CONFIG_FILE}" "${TRIAL_ID}" "${SINGLE_CONFIG_PATH}"

  LOG_PREFIX="${TMP_DIR}/trial_${RUN_INDEX}_${TRIAL_ID}"
  SIM_LOG="${LOG_PREFIX}_simulation.log"
  POLICY_LOG="${LOG_PREFIX}_policy.log"
  RECORDER_LOG="${LOG_PREFIX}_recorder.log"

  echo
  echo "[$(date +'%F %T')] Trial ${RUN_INDEX}/${TOTAL_TRIALS}: ${TRIAL_ID}"
  echo "  single-trial config: ${SINGLE_CONFIG_PATH}"
  echo "  logs: ${LOG_PREFIX}_{simulation,policy,recorder}.log"
  TRIAL_RESULTS_DIR="${RESULTS_ROOT}/trial_${RUN_INDEX}_${TRIAL_ID}"
  SCORING_FILE="${TRIAL_RESULTS_DIR}/scoring.yaml"
  mkdir -p "${TRIAL_RESULTS_DIR}"
  echo "  scoring dir: ${TRIAL_RESULTS_DIR}"

  cleanup_stale_sim_router

  SIM_CMD="/entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=${GAZEBO_GUI} launch_rviz:=${LAUNCH_RVIZ} aic_engine_config_file:=${SINGLE_CONFIG_PATH} shutdown_on_aic_engine_exit:=true"

  (
    export DBX_CONTAINER_MANAGER=docker
    distrobox enter -r "${SIM_DISTROBOX_NAME}" -- bash -lc "cd \"${WORKSPACE_DIR}\" && export AIC_RESULTS_DIR=\"${TRIAL_RESULTS_DIR}\" && ${SIM_CMD}"
  ) >"${SIM_LOG}" 2>&1 &
  SIM_PID=$!

  sleep "${STARTUP_DELAY_SEC}"

  RECORDER_CMD=(
    pixi run aic-policy-recorder
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.single_task=${DATASET_SINGLE_TASK}"
    "--dataset.root=${DATASET_ROOT}"
    --dataset.fps=20
    "--action_mode=${ACTION_MODE}"
    --max_episodes=1
  )
  if [[ "${SAVE_FAILED_EPISODES}" == "true" ]]; then
    RECORDER_CMD+=(--save_failed_episodes)
  fi
  if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    RECORDER_CMD+=(--dataset.push_to_hub)
  fi
  if [[ "${RUN_INDEX}" -gt 1 || "${DATASET_EXISTS_BEFORE_RUN}" == "true" ]]; then
    RECORDER_CMD+=(--dataset.resume)
  fi

  (
    cd "${WORKSPACE_DIR}"
    "${RECORDER_CMD[@]}"
  ) >"${RECORDER_LOG}" 2>&1 &
  RECORDER_PID=$!

  (
    cd "${WORKSPACE_DIR}"
    if [[ -n "${AIC_OFFICIAL_TEACHER_TRAJECTORY}" ]]; then
      export AIC_OFFICIAL_TEACHER_TRAJECTORY
      export AIC_OFFICIAL_TEACHER_ACTION_MODE
    fi
    pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p "policy:=${POLICY_CLASS}"
  ) >"${POLICY_LOG}" 2>&1 &
  POLICY_PID=$!

  START_EPOCH="$(date +%s)"
  SIM_EXIT_EPOCH=0
  RECORDER_DRAIN_TIMEOUT="false"
  TRIAL_TIMED_OUT="false"

  while true; do
    if ! kill -0 "${RECORDER_PID}" >/dev/null 2>&1; then
      break
    fi

    if ! kill -0 "${SIM_PID}" >/dev/null 2>&1; then
      if [[ "${SIM_EXIT_EPOCH}" -eq 0 ]]; then
        SIM_EXIT_EPOCH="$(date +%s)"
      fi
      NOW_EPOCH="$(date +%s)"
      if [[ "$((NOW_EPOCH - SIM_EXIT_EPOCH))" -ge "${RECORDER_DRAIN_SEC}" ]]; then
        echo "  recorder did not finish within ${RECORDER_DRAIN_SEC}s after sim exit."
        RECORDER_DRAIN_TIMEOUT="true"
        break
      fi
    fi

    if [[ "${PER_TRIAL_TIMEOUT_SEC}" -gt 0 ]]; then
      NOW_EPOCH="$(date +%s)"
      ELAPSED="$((NOW_EPOCH - START_EPOCH))"
      if [[ "${ELAPSED}" -ge "${PER_TRIAL_TIMEOUT_SEC}" ]]; then
        echo "  timeout reached (${PER_TRIAL_TIMEOUT_SEC}s), stopping trial ${TRIAL_ID}."
        TRIAL_TIMED_OUT="true"
        break
      fi
    fi

    sleep 1
  done

  if kill -0 "${SIM_PID}" >/dev/null 2>&1; then
    terminate_process "${SIM_PID}" "simulation"
  fi
  if kill -0 "${POLICY_PID}" >/dev/null 2>&1; then
    terminate_process "${POLICY_PID}" "policy"
  fi
  if kill -0 "${RECORDER_PID}" >/dev/null 2>&1; then
    terminate_process "${RECORDER_PID}" "recorder"
  fi

  SIM_EXIT=0
  if ! wait "${SIM_PID}"; then SIM_EXIT=$?; fi
  POLICY_EXIT=0
  if ! wait "${POLICY_PID}"; then POLICY_EXIT=$?; fi
  RECORDER_EXIT=0
  if ! wait "${RECORDER_PID}"; then RECORDER_EXIT=$?; fi

  EPISODE_SAVED_LOG_MATCH="false"
  if [[ -f "${RECORDER_LOG}" ]]; then
    if grep -q "Episode saved" "${RECORDER_LOG}"; then
      EPISODE_SAVED_LOG_MATCH="true"
      EPISODE_SAVED_LOG_LINE="$(grep -n "Episode saved" "${RECORDER_LOG}" | tail -n 1)"
      echo "  Episode saved: ${EPISODE_SAVED_LOG_LINE}"
    else
      echo "  Episode saved: NOT FOUND in recorder log"
    fi
  else
    echo "  Episode saved: recorder log missing (${RECORDER_LOG})"
  fi

  TRIAL_FAILED="false"
  if [[ "${TRIAL_TIMED_OUT}" == "true" ]]; then TRIAL_FAILED="true"; fi
  if [[ "${RECORDER_DRAIN_TIMEOUT}" == "true" ]]; then TRIAL_FAILED="true"; fi
  if [[ "${RECORDER_EXIT}" -ne 0 ]]; then TRIAL_FAILED="true"; fi
  if [[ "${REQUIRE_RECORDER_SAVE_LOG}" == "true" && "${EPISODE_SAVED_LOG_MATCH}" != "true" ]]; then
    TRIAL_FAILED="true"
  fi
  TRIAL_TOTAL_SCORE=""
  if [[ -f "${SCORING_FILE}" ]]; then
    TRIAL_TOTAL_SCORE="$(awk '/^total:/{print $2; exit}' "${SCORING_FILE}")"
    echo "  scoring total: ${TRIAL_TOTAL_SCORE}"
  else
    echo "  scoring file missing: ${SCORING_FILE}"
  fi

  if [[ "${TRIAL_FAILED}" == "true" ]]; then
    FAILURES=$((FAILURES + 1))
    echo "  result: FAILED"
    echo "  exit codes: sim=${SIM_EXIT}, policy=${POLICY_EXIT}, recorder=${RECORDER_EXIT}"
    echo "  inspect logs: ${SIM_LOG}"
    echo "${RUN_INDEX},${TRIAL_ID},FAILED,${TRIAL_TOTAL_SCORE},${SCORING_FILE}" >> "${SCORE_SUMMARY_CSV}"
    if [[ "${CONTINUE_ON_FAILURE}" != "true" ]]; then
      echo "Stopping early due to failure (continue_on_failure=false)."
      exit 1
    fi
  else
    echo "  result: OK"
    echo "  exit codes: sim=${SIM_EXIT}, policy=${POLICY_EXIT}, recorder=${RECORDER_EXIT}"
    echo "${RUN_INDEX},${TRIAL_ID},OK,${TRIAL_TOTAL_SCORE},${SCORING_FILE}" >> "${SCORE_SUMMARY_CSV}"
  fi

  cleanup_trial_bags "${TRIAL_RESULTS_DIR}"

  sleep "${PAUSE_BETWEEN_TRIALS_SEC}"
done

echo
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Completed with failures: ${FAILURES}/${TOTAL_TRIALS} trials failed."
  echo "Per-trial score summary: ${SCORE_SUMMARY_CSV}"
  exit 1
fi

echo "Completed successfully: ${TOTAL_TRIALS}/${TOTAL_TRIALS} trials."
echo "Per-trial score summary: ${SCORE_SUMMARY_CSV}"
