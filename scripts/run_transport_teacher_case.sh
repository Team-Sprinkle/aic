#!/usr/bin/env bash
set -euo pipefail
case_name=${1:?case name}
insert=${2:-false}
root=$(realpath outputs/experiments/2026-09-30_transport_continuation)
export AIC_MOVEIT_PILOT_GPU=${AIC_MOVEIT_PILOT_GPU:-7}
export AIC_MOVEIT_PILOT_ACTION_MODE=${AIC_MOVEIT_PILOT_ACTION_MODE:-absolute_cartesian_pose_base_link}
export AIC_MOVEIT_PILOT_SCENE_DIR="$root/scenes/$case_name"
export AIC_MOVEIT_PILOT_ONLINE_INSERTION="$insert"
export AIC_MOVEIT_PILOT_TRACKING_GATE_FORCE_FRACTION=6.0
export AIC_MOVEIT_PILOT_FORCE_CONFIRM_THRESHOLD_N=2.0
export AIC_MOVEIT_PILOT_NOMINAL_INITIAL_LATERAL_GATE_M=.012
export AIC_MOVEIT_PILOT_PRECONTACT_PORT_ALIGN_SEC=1.2
export AIC_MOVEIT_PILOT_SC_TRACKING_GATE_SPEED_MPS=.008
export AIC_MOVEIT_PILOT_PRECONTACT_SOFT_FORCE_N=1.5
export AIC_MOVEIT_PILOT_PRECONTACT_FATAL_FORCE_N=2.0
export AIC_MOVEIT_PILOT_SC_Z_GATE_MAX_HOLD_SEC=1.5
export AIC_MOVEIT_PILOT_SC_CONTACT_RECOVERY=true
export AIC_MOVEIT_PILOT_GUARDED_LATERAL_SERVO=true
export AIC_MOVEIT_PILOT_SC_RECOVERY_MAX_BACKOFF_M=.04
export AIC_MOVEIT_PILOT_SC_RECOVERY_BACKOFF_SEC=1.2
export AIC_MOVEIT_PILOT_SC_RECOVERY_REQUIRED_MEASURED_M=.015
suffix=held
if [[ "$insert" == true ]]; then suffix=insert; fi
bash scripts/run_moveit_sc_route_pilot.sh "$root/plans/${case_name}_${suffix}" "${3:-$root/runs/$case_name}"
