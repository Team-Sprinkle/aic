#!/usr/bin/env bash
set -euo pipefail
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval || true
echo "Container created. Start interactive eval with:"
echo "  distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true"
