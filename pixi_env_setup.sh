#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

_AIC_WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_SIM_RESOURCE_PATH="${_AIC_WORKSPACE_ROOT}/aic_assets/models:${_AIC_WORKSPACE_ROOT}/aic_assets:${_AIC_WORKSPACE_ROOT}/aic_description:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${_AIC_WORKSPACE_ROOT}/install/lib/aic_gazebo:${CONDA_PREFIX:-${_AIC_WORKSPACE_ROOT}/.pixi/envs/default}/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_CONFIG_PATH="${CONDA_PREFIX:-${_AIC_WORKSPACE_ROOT}/.pixi/envs/default}/share/gz:${GZ_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${_AIC_WORKSPACE_ROOT}/install/lib/aic_gazebo:${_AIC_WORKSPACE_ROOT}/install/lib:${LD_LIBRARY_PATH:-}"
