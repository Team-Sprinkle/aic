#!/usr/bin/env bash
set -euo pipefail
# Colcon's generated setup scripts read optional variables without defaults.
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export LD_LIBRARY_PATH="/opt/aic/runtime/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/opt/aic/runtime/lib/libjpeg.so.8:/opt/aic/runtime/lib/libpng16.so.16"
export PYTHONPATH="/opt/aic/runtime/lib/python3.12/site-packages/cv2/python-3.12:/opt/aic/source:/opt/aic/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
if [[ -n "${AIC_ROUTER_ADDR:-}" ]]; then
  export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/$AIC_ROUTER_ADDR\"];transport/shared_memory/enabled=false"
  if [[ "${AIC_ENABLE_ACL:-false}" == true || "${AIC_ENABLE_ACL:-0}" == 1 ]]; then
    : "${AIC_MODEL_PASSWD:?AIC_MODEL_PASSWD is required for ACL mode}"
    umask 077
    printf 'model:%s\n' "$AIC_MODEL_PASSWD" > /tmp/aic-final-credentials
    export ZENOH_CONFIG_OVERRIDE="$ZENOH_CONFIG_OVERRIDE;transport/auth/usrpwd/user=\"model\";transport/auth/usrpwd/password=\"$AIC_MODEL_PASSWD\";transport/auth/usrpwd/dictionary_file=\"/tmp/aic-final-credentials\""
  fi
fi
exec /usr/bin/python3 -m aic_model.aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_example_policies.final.FinalInsertionPolicy "$@"
