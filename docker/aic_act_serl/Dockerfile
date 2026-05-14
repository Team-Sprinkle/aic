FROM docker.io/library/ros:kilted-ros-core AS build

# 1. Install Pixi
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
COPY aic_example_policies/aic_example_policies/ros/RunACTAdapterSERL.py /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACTAdapterSERL.py
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils/gazebo_rl /ws_aic/src/aic/aic_utils/gazebo_rl
COPY aic_utils/lerobot_robot_aic /ws_aic/src/aic/aic_utils/lerobot_robot_aic
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/
COPY robostack_mapping.json /ws_aic/src/aic/robostack_mapping.json

# 3. Add Trained Model Artifacts (The "Stable Path" strategy)
ARG AIC_SERL_CHECKPOINT_SOURCE=outputs/act_adapter_serl/checkpoint_latest.pt
ARG AIC_ACT_TORCHSCRIPT_SOURCE=outputs/act_policy_ts_080000_cpu.pt
ARG AIC_ACT_TORCHSCRIPT_METADATA_SOURCE=outputs/act_policy_ts_080000_cpu.json
ARG AIC_ACT_NORMALIZER_SOURCE=outputs/checkpoints/080000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors

RUN mkdir -p /opt/aic_policy/act_adapter_serl \
    /opt/aic_policy/act_080000/checkpoints/080000/pretrained_model
COPY ${AIC_SERL_CHECKPOINT_SOURCE} /opt/aic_policy/act_adapter_serl/checkpoint_latest.pt
COPY ${AIC_ACT_TORCHSCRIPT_SOURCE} /opt/aic_policy/act_080000/act_policy_ts_080000_cpu.pt
COPY ${AIC_ACT_TORCHSCRIPT_METADATA_SOURCE} /opt/aic_policy/act_080000/act_policy_ts_080000_cpu.json
COPY ${AIC_ACT_NORMALIZER_SOURCE} \
     /opt/aic_policy/act_080000/checkpoints/080000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors

# 4. Install Python Dependencies via Pixi
SHELL ["/bin/bash", "-c"]
RUN rm -rf /root/.cache/rattler/cache /ws_aic/src/aic/.pixi \
 && cd /ws_aic/src/aic \
 && rm -f pixi.lock \
 && pixi install --concurrent-downloads 1 --concurrent-solves 1




# 5. Set ACT-adapter SERL Runtime Environment Variables
ENV AIC_SERL_CHECKPOINT=/opt/aic_policy/act_adapter_serl/checkpoint_latest.pt
ENV AIC_SERL_ACT_TORCHSCRIPT=/opt/aic_policy/act_080000/act_policy_ts_080000_cpu.pt
ENV AIC_SERL_COMMAND_MODE=delta_pose
ENV AIC_SERL_COMMAND_FRAME=gripper/tcp
ENV AIC_SERL_CONTROL_HZ=20
ENV AIC_SERL_MAX_RUNTIME_SEC=170
ENV AIC_SERL_N_ACTION_STEPS=4

WORKDIR /ws_aic/src/aic
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
