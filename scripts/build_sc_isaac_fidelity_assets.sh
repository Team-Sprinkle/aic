#!/usr/bin/env bash
# Build the SC grasp/cable and port collision assets used by the September 24
# normal-collision Isaac probes. Run inside the rootless IsaacLab container.
set -euo pipefail

aic_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
aic_isaaclab_root="${AIC_ISAACLAB_ROOT:-/workspace/isaaclab}"
aic_python="${aic_isaaclab_root}/_isaac_sim/python.sh"
aic_assets="${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets"
export PYTHONPATH="${aic_isaaclab_root}/source/isaaclab:${aic_isaaclab_root}/source/isaaclab_assets:${aic_isaaclab_root}/source/isaaclab_tasks:${aic_isaaclab_root}/source/isaaclab_mimic:${aic_isaaclab_root}/source/isaaclab_rl:${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/source/aic_task${PYTHONPATH:+:${PYTHONPATH}}"

"${aic_python}" "${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/scripts/build_sc_reversed_robot_usd.py" \
  --headless \
  --source "${aic_assets}/aic_unified_robot_cable_sdf.usd" \
  --output "${aic_assets}/aic_unified_robot_cable_reversed_gazebo_firstlink_aligned_filtered.generated.usd" \
  --mode reversed_topology \
  --gazebo-tool-sc-calibration "${aic_repo_root}/configs/hierarchical_recovery/sc_tool_plug_gazebo_tf_251.json" \
  --gazebo-cable-end0-calibration "${aic_repo_root}/configs/hierarchical_recovery/sc_cable_first_links_gazebo_tf_251.json" \
  --filter-grasp-internal-collisions \
  --filter-fixed-gripper-ati-base-pair

"${aic_python}" "${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/scripts/build_sc_port_sdf_collisions_usd.py" \
  --headless \
  --source "${aic_assets}/assets/SC Port/sc_port.usd" \
  --sdf "${aic_repo_root}/aic_assets/models/SC Port/model.sdf" \
  --output "${aic_assets}/assets/SC Port/sc_port_sdf_aligned.generated.usd"

sha256sum \
  "${aic_assets}/aic_unified_robot_cable_reversed_gazebo_firstlink_aligned_filtered.generated.usd" \
  "${aic_assets}/assets/SC Port/sc_port_sdf_aligned.generated.usd"
