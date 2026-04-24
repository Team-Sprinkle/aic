#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${HF_USER:-}" ]]; then
  echo "Error: HF_USER must be set before running this script." >&2
  exit 1
fi

for number in {2..6}; do
  echo "--------- Starting run ${number}/6... ------------"

  "${SCRIPT_DIR}/launch_policy_recording_per_trial.sh" \
    --engine-config /home/jk/ws_aic/src/aic/outputs/configs/fixed_20_trials_sfp2nic.yaml \
    --policy-class aic_example_policies.ros.CheatCode \
    --dataset-repo-id "${HF_USER}/sfp2nic" \
    --dataset-root "/home/jk/ws_aic/src/aic/outputs/test_${number}_dataset" \
    --results-root "/home/jk/ws_aic/src/aic/outputs/test_${number}_scores" \
    --gazebo-gui false \
    --launch-rviz false \
    --require-recorder-save-log true \
    --sudo-keepalive true

  if [[ "${number}" -lt 6 ]]; then
    echo "Sleeping 10 seconds before the next run..."
    sleep 10
  fi

done
