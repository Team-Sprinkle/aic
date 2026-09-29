#!/usr/bin/env bash
# Replay the corrected SC mechanics scene inside a rootless IsaacLab container.
# Usage: bash scripts/run_sc_isaac_fidelity_probe.sh full_start|near_center|near_plus3|near_minus3
set -euo pipefail

aic_scenario="${1:?choose full_start, near_center, near_plus3, or near_minus3}"
aic_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
aic_isaaclab_root="${AIC_ISAACLAB_ROOT:-/workspace/isaaclab}"
aic_assets="${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets"
aic_run_root="${AIC_SC_FIDELITY_OUTPUT_DIR:-${aic_repo_root}/outputs/experiments/2026-09-24_isaac_sc_fidelity}"
mkdir -p "${aic_run_root}"

export PYTHONPATH="${aic_isaaclab_root}/source/isaaclab:${aic_isaaclab_root}/source/isaaclab_assets:${aic_isaaclab_root}/source/isaaclab_tasks:${aic_isaaclab_root}/source/isaaclab_mimic:${aic_isaaclab_root}/source/isaaclab_rl:${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/source/aic_task${PYTHONPATH:+:${PYTHONPATH}}"
export AIC_ISAAC_EPISODE_CONFIG_DIR="${aic_repo_root}/configs/hierarchical_recovery/isaac_sc_fidelity"
export AIC_ISAAC_ROBOT_USD_PATH="${aic_assets}/aic_unified_robot_cable_reversed_gazebo_firstlink_aligned_filtered.generated.usd"
export AIC_ISAAC_SC_PORT_USD_PATH="${aic_assets}/assets/SC Port/sc_port_sdf_aligned.generated.usd"
# The imported USD's robot self-collision enables spurious fixed-gripper and
# distal-cable contacts (about 20 kN at the fingers in a stationary probe).
# The Gazebo cable SDF does not request self-collision. Scene contacts with
# cards, board and ports remain active and are checked in the fidelity audit.
export AIC_ISAAC_ENABLE_ROBOT_SELF_COLLISIONS=0

aic_lateral_x=0
aic_steps=120
aic_descent_start=20
aic_descent_end=80
aic_near_reset=1
case "${aic_scenario}" in
  full_start)
    aic_steps=160
    aic_descent_start=80
    aic_descent_end=130
    aic_near_reset=0
    ;;
  near_center) ;;
  near_plus3) aic_lateral_x=0.003 ;;
  near_minus3) aic_lateral_x=-0.003 ;;
  *) echo "unknown scenario: ${aic_scenario}" >&2; exit 2 ;;
esac

aic_reset_args=()
if [[ "${aic_near_reset}" == 1 ]]; then
  aic_reset_args=(
    --near_gate_reset_joint_seed -0.6777207255 -2.1153156757 -1.5493192673 -1.3591722250 1.9228538275 0.8320367336
    --near_gate_reset_max_iterations 30
    --near_gate_reset_physical_interpolation
    --near_gate_reset_interpolation_steps 600
    --near_gate_reset_hold_steps 300
    --near_gate_reset_park_scene
  )
fi

"${aic_isaaclab_root}/_isaac_sim/python.sh" \
  "${aic_repo_root}/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/probe_target_reward.py" \
  --headless --task_family sc_to_sc --target_body sc_tip_link \
  --controller target_body_pose_root \
  --no-fix_isaac_ik_xy_sign --no-fix_isaac_ik_z_sign \
  --target_position_offset "${aic_lateral_x}" 0 0.002 \
  --insert_offset "${aic_lateral_x}" 0 0.002 \
  --steps "${aic_steps}" --descent_start_step "${aic_descent_start}" \
  --descent_end_step "${aic_descent_end}" --reaching_threshold 0.0005 \
  "${aic_reset_args[@]}" --debug_named_contacts \
  --output "${aic_run_root}/${aic_scenario}.json" \
  > "${aic_run_root}/${aic_scenario}.log" 2>&1

printf 'trace=%s\nlog=%s\n' "${aic_run_root}/${aic_scenario}.json" "${aic_run_root}/${aic_scenario}.log"
