#!/usr/bin/env bash
set -e

# Optional: set container manager (default Docker)
export DBX_CONTAINER_MANAGER=docker

# Ask user whether to pull the latest container
read -p "Pull latest aic_eval container? [y/N]: " PULL_CHOICE
PULL_CHOICE=${PULL_CHOICE:-n}

if [[ "$PULL_CHOICE" =~ ^[Yy]$ ]]; then
    echo "Pulling latest aic_eval container..."
    docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
else
    echo "Skipping docker pull. Using existing image."
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected. Launching container with GPU support..."
    distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval || true
else
    echo "No NVIDIA GPU detected. Launching container without GPU support..."
    distrobox create -r -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval || true
fi

# Enter the container and start the evaluation engine
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true