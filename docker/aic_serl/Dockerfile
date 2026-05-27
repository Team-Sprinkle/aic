FROM docker.io/library/ros:kilted-ros-core AS build

RUN apt update && apt install -y curl git && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils/gazebo_rl /ws_aic/src/aic/aic_utils/gazebo_rl
COPY aic_utils/lerobot_robot_aic /ws_aic/src/aic/aic_utils/lerobot_robot_aic
COPY docker/aic_serl/pixi.toml /ws_aic/src/aic/pixi.toml
COPY robostack_mapping.json /ws_aic/src/aic/robostack_mapping.json
COPY pixi_env_setup.sh /ws_aic/src/aic/
RUN mkdir -p /ws_aic/src/aic/scripts
COPY scripts/submission_smoke_check.py /ws_aic/src/aic/scripts/submission_smoke_check.py
COPY submission_handoff/artifacts/submission_candidate /opt/aic_policy/submission_candidate

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic \
    && pixi install \
    && rm -rf /ws_aic/src/aic/.pixi/bld /root/.cache/uv \
    && rm -rf /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/triton* \
    && find /ws_aic/src/aic/.pixi/envs/default -type f -name '*.a' -delete

WORKDIR /ws_aic/src/aic
ENV PYTHONPATH=/ws_aic/src/aic/aic_utils/lerobot_robot_aic
ENV AIC_SERL_CHECKPOINT=/opt/aic_policy/submission_candidate/online_serl/checkpoint_000300.pt
ENV AIC_SERL_ACT_TORCHSCRIPT=/opt/aic_policy/submission_candidate/act_policy_ts_175000_cuda0.pt
ENV AIC_SERL_DEVICE=cuda
ENV AIC_SERL_CONTROL_HZ=20
ENV AIC_SERL_N_ACTION_STEPS=4
ENV AIC_SERL_COMMAND_MODE=delta_pose
ENV AIC_SERL_COMMAND_FRAME=gripper/tcp
ENV AIC_SERL_MAX_TRANSLATION_DELTA=0.0007
ENV AIC_SERL_MAX_ROTATION_DELTA=0.0001
ENV AIC_SERL_TRANSLATION_DEADBAND=0.0
ENV AIC_SERL_ROTATION_DEADBAND=0.0

COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.RunACTAdapterSERL", "-p", "use_sim_time:=true"]
