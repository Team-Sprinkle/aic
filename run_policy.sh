#!/usr/bin/env bash
set -e

WORKSPACE_DIR=$(pwd)

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Workspace not found at $WORKSPACE_DIR"
    exit 1
fi

cd "$WORKSPACE_DIR"

# List available policies
echo "Available example policies:"
POLICIES=$(find aic_example_policies/aic_example_policies/ros -name "*.py" \
           ! -name "__init__.py" | sed 's|aic_example_policies/aic_example_policies|aic_example_policies|' | sed 's|/|.|g' | sed 's|\.py$||')
echo "$POLICIES"
echo

# Prompt user to select policy
read -p "Enter the policy to run (default WaveArm): " POLICY_NAME
POLICY_NAME=${POLICY_NAME:-aic_example_policies.ros.WaveArm}

echo "Running policy: $POLICY_NAME"
echo

# Run the policy using Pixi on the host
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:="$POLICY_NAME"