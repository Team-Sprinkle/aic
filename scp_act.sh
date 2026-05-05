#!/bin/bash

# Define local workspace for easy reference in this shell session
WORKSPACE="/home/brucekimrok/projects/ws_aic/src/aic"

# 1. Copy the Python Script
scp -P 7910 knuth:/data1/chmin/yj/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py \
${WORKSPACE}/aic_example_policies/aic_example_policies/ros/

# 2. Copy the Policy JSON and PT files
scp -P 7910 knuth:/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/act/bc/20260504_act_identity_taskvec_50k_4gpu/act_policy_ts_080000_cpu.{json,pt} \
${WORKSPACE}/outputs/

# 3. Copy the Normalizer (using the 20260504 version for consistency)
# Create the directory first locally if it doesn't exist
mkdir -p ${WORKSPACE}/outputs/checkpoints/080000/pretrained_model/
scp -P 7910 knuth:/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/act/bc/20260504_act_identity_taskvec_50k_4gpu/checkpoints/080000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors \
${WORKSPACE}/outputs/checkpoints/080000/pretrained_model/