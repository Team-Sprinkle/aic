#!/usr/bin/env bash
# Rootless one-GPU transport BC on the immutable cleaned SC chunks.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
gpu=${1:?physical GPU index}
initialization=${2:?imagenet_clean or mixed_warm_start}
output=$(realpath -m "${3:?new output directory}")
case "$initialization" in imagenet_clean|mixed_warm_start) ;; *) echo 'Invalid initialization' >&2; exit 2 ;; esac
export DOCKER_HOST=${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}
labels=${AIC_LLB_SC_LABELS:-$repo/outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks/transport_chunks.jsonl}
labels=$(realpath "$labels")
manifest="${labels%/*}/manifest.json"
test -s "$labels" && test -s "$manifest"
"$repo/.pixi/envs/default/bin/python" - "$manifest" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['training_ready'] and x['episode_count']==104
assert not x['final_scenes_opened'] and not x['actor_geometry_input']
assert x['counts']['transport']>0
PY
if [[ -e "$output/initial.pt" || -e "$output/summary.json" ]]; then
  echo "Existing training artifacts: $output" >&2; exit 1
fi
mkdir -p "$output"
name="aic_sc_llb_bc_${initialization}_gpu${gpu}_20261001"
if docker inspect "$name" >/dev/null 2>&1; then echo "Container already exists: $name" >&2; exit 1; fi
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
docker run --rm --name "$name" --gpus "device=$gpu" --shm-size=4g --net=none \
  --entrypoint /bin/bash \
  -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -e AIC_LLB_INIT="$initialization" -e AIC_LLB_OUTPUT="$output" -e AIC_LLB_LABELS="$labels" \
  -e AIC_LLB_INCLUDE_ELAPSED="${AIC_LLB_INCLUDE_ELAPSED:-0}" \
  -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act:ro \
  -v /var/tmp/aic_sc_fix_20260924:/var/tmp/aic_sc_fix_20260924:ro \
  -v /home/nas:/home/nas:ro -w "$repo" "$image" -lc '
  export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
  args=()
  if [[ "$AIC_LLB_INIT" == imagenet_clean ]]; then
    args+=(--imagenet-resnet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth)
  else
    args+=(--warm-start /var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_20260925/best.pt)
  fi
  if [[ "$AIC_LLB_INCLUDE_ELAPSED" == 1 ]]; then
    args+=(--include-elapsed)
  fi
  .pixi/envs/default/bin/python scripts/train_clean_transport_bc.py \
    --sfp-labels /var/tmp/chmin_aic_20260918_act/sfp_verified_268_port_tcp_waypoints_20260924.jsonl \
    --sc-labels "$AIC_LLB_LABELS" \
    --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
    --reset-audit outputs/experiments/2026-09-30_transport_continuation/sfp_physical_reset_audit.json \
    --output "$AIC_LLB_OUTPUT" --initialization "$AIC_LLB_INIT" \
    --updates 6000 --min-updates 2000 --budget-minutes 300 \
    --sc-validation-samples-per-episode 64 \
    --rotation-scale-deg 1 --mask-controller-error --balance-sc-ports "${args[@]}"
  ' > "$output/train.log" 2>&1
