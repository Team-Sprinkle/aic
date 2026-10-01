#!/usr/bin/env bash
# Own one GPU; no active simulation may share this experiment's GPU.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
export DOCKER_HOST=${DOCKER_HOST:-unix:///run/user/1008/docker.sock}
name=${AIC_TRANSPORT_TRAIN_CONTAINER:-aic_clean_transport_bc_gpu7_20260930}
output=${AIC_TRANSPORT_TRAIN_OUTPUT:-"$repo/outputs/experiments/2026-09-30_transport_continuation/transport_bc"}
labels=${AIC_TRANSPORT_SC_LABELS:-"$repo/outputs/trajectory_datasets/transport_local_verified/2026-09-30_v6/transport_chunks.jsonl"}
mkdir -p "$output"
if [[ -f "$output/initial.pt" || -f "$output/summary.json" ]]; then
  echo "Existing training artifacts: preserve this run; choose a new output for a new experiment." >&2
  exit 1
fi
if docker inspect "$name" >/dev/null 2>&1; then echo "Existing job: $name" >&2; exit 1; fi
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
docker run -i --rm --name "$name" --gpus device=7 --shm-size=4g --net=none --entrypoint /bin/bash \
  -e NVIDIA_VISIBLE_DEVICES=7 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -e AIC_TRANSPORT_TRAIN_OUTPUT="$output" -e AIC_TRANSPORT_SC_LABELS="$labels" \
  -e AIC_TRANSPORT_ROTATION_SCALE_DEG="${AIC_TRANSPORT_ROTATION_SCALE_DEG:-}" \
  -e AIC_TRANSPORT_MASK_CONTROLLER_ERROR="${AIC_TRANSPORT_MASK_CONTROLLER_ERROR:-0}" \
  -e AIC_TRANSPORT_BALANCE_SC_PORTS="${AIC_TRANSPORT_BALANCE_SC_PORTS:-0}" \
  -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act:ro \
  -w "$repo" "$image" -s > "$output/train.log" 2>&1 <<'TRAIN_CONTAINER'
  export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
  # Frozen ROS image can load older system codecs before conda OpenCV. Pin only
  # this training process to its environment's ABI-compatible image codecs.
  export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
  loss_args=()
  if [[ -n "$AIC_TRANSPORT_ROTATION_SCALE_DEG" ]]; then
    loss_args+=(--rotation-scale-deg "$AIC_TRANSPORT_ROTATION_SCALE_DEG")
  fi
  if [[ "$AIC_TRANSPORT_MASK_CONTROLLER_ERROR" == 1 ]]; then
    loss_args+=(--mask-controller-error)
  fi
  if [[ "$AIC_TRANSPORT_BALANCE_SC_PORTS" == 1 ]]; then
    loss_args+=(--balance-sc-ports)
  fi
  .pixi/envs/default/bin/python scripts/train_clean_transport_bc.py \
    --sfp-labels /var/tmp/chmin_aic_20260918_act/sfp_verified_268_port_tcp_waypoints_20260924.jsonl \
    --sc-labels "$AIC_TRANSPORT_SC_LABELS" \
    --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
    --warm-start /var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_20260925/best.pt \
    --reset-audit outputs/experiments/2026-09-30_transport_continuation/sfp_physical_reset_audit.json \
    --output "$AIC_TRANSPORT_TRAIN_OUTPUT" \
    --updates 1500 --budget-minutes 90 "${loss_args[@]}"
TRAIN_CONTAINER
