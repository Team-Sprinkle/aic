FROM docker.io/library/ros:kilted-ros-core AS build

# 1. Install System Dependencies & Pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# 2. Add Source Code Dependencies
RUN mkdir -p /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros \
    /ws_aic/src/aic/aic_example_policies/resource
COPY aic_example_policies/package.xml /ws_aic/src/aic/aic_example_policies/package.xml
COPY aic_example_policies/pixi.toml /ws_aic/src/aic/aic_example_policies/pixi.toml
COPY aic_example_policies/setup.py /ws_aic/src/aic/aic_example_policies/setup.py
COPY aic_example_policies/setup.cfg /ws_aic/src/aic/aic_example_policies/setup.cfg
COPY aic_example_policies/resource/aic_example_policies /ws_aic/src/aic/aic_example_policies/resource/aic_example_policies
COPY aic_example_policies/aic_example_policies/__init__.py /ws_aic/src/aic/aic_example_policies/aic_example_policies/__init__.py
COPY aic_example_policies/aic_example_policies/ros/__init__.py /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/__init__.py
COPY aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# 3. Add Trained Model Artifacts (The "Stable Path" strategy)
RUN mkdir -p /opt/aic_policy/act_080000/checkpoints/080000/pretrained_model
COPY outputs/act_policy_ts_080000_cpu.pt /opt/aic_policy/act_080000/
COPY outputs/act_policy_ts_080000_cpu.json /opt/aic_policy/act_080000/
COPY outputs/checkpoints/080000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors \
     /opt/aic_policy/act_080000/checkpoints/080000/pretrained_model/

# 4. Install Python Dependencies via Pixi
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked -vvv

# 5. Set ACT Runtime Environment Variables
ENV AIC_ACT_TORCHSCRIPT=/opt/aic_policy/act_080000/act_policy_ts_080000_cpu.pt
ENV AIC_ACT_DEVICE=cpu
ENV AIC_ACT_RUNTIME_COMMAND_MODE=delta_pose
ENV AIC_ACT_COMMAND_FRAME=gripper/tcp
ENV AIC_ACT_CONTROL_HZ=20
ENV AIC_ACT_MAX_RUNTIME_SEC=170

WORKDIR /ws_aic/src/aic

# 6. Modified Entrypoint (Preserving Zenoh Logic)
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

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model with Zenoh overrides
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

# Source ROS workspace before execution
source /opt/ros/kilted/setup.bash

exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]

# 7. Default CMD for ACT Policy
CMD ["--ros-args", \
     "-p", "policy:=aic_example_policies.ros.RunACTTorchScript", \
     "-p", "use_sim_time:=true", \
     "-p", "translation_clamp:=0.01", \
     "-p", "rotation_clamp:=0.1"]
