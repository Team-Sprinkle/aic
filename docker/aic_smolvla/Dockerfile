FROM docker.io/library/ros:kilted-ros-core AS build

# 1. Install System Dependencies & Pixi
RUN apt update && apt install -y curl git && curl -fsSL https://pixi.sh/install.sh | bash
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
COPY aic_example_policies/aic_example_policies/ros/RunSmolVLA.py /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunSmolVLA.py
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# 3. Add Trained Model Artifacts (The "Stable Path" strategy)
ENV AIC_SMOLVLA_ARTIFACT_ROOT=/opt/aic_smolvla
ENV AIC_SMOLVLA_POLICY_PATH=${AIC_SMOLVLA_ARTIFACT_ROOT}/checkpoints/050000/pretrained_model
ENV AIC_SMOLVLA_VLM_PATH=${AIC_SMOLVLA_ARTIFACT_ROOT}/models/SmolVLM2-500M-Video-Instruct
RUN mkdir -p ${AIC_SMOLVLA_ARTIFACT_ROOT}/checkpoints ${AIC_SMOLVLA_ARTIFACT_ROOT}/models
COPY outputs/smolvla/smolvla_050626/050000 ${AIC_SMOLVLA_ARTIFACT_ROOT}/checkpoints/050000
COPY outputs/smolvla/smolvla_050626/models ${AIC_SMOLVLA_ARTIFACT_ROOT}/models
RUN sed -i "s#HuggingFaceTB/SmolVLM2-500M-Video-Instruct#${AIC_SMOLVLA_VLM_PATH}#g" \
      ${AIC_SMOLVLA_POLICY_PATH}/config.json \
      ${AIC_SMOLVLA_POLICY_PATH}/policy_preprocessor.json \
      ${AIC_SMOLVLA_POLICY_PATH}/train_config.json

# 4. Install Python Dependencies via Pixi
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked -vvv

# 5. Set SmolVLA Runtime Environment Variables
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV AIC_SMOLVLA_COMMAND_MODE=delta_pose
ENV AIC_SMOLVLA_COMMAND_FRAME=gripper/tcp
ENV AIC_SMOLVLA_CONTROL_HZ=4
ENV AIC_SMOLVLA_MAX_RUNTIME_SEC=170

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

# 7. Default CMD for SmolVLA Policy
CMD ["--ros-args", \
     "-p", "policy:=aic_example_policies.ros.RunSmolVLA", \
     "-p", "use_sim_time:=true", \
     "-p", "translation_clamp:=0.01", \
     "-p", "rotation_clamp:=0.1"]
