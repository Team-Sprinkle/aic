#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

_AIC_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_AIC_REPO_ROOT}/aic_teacher_official:${_AIC_REPO_ROOT}/aic_example_policies:${_AIC_REPO_ROOT}/aic_model:${PYTHONPATH:-}"
