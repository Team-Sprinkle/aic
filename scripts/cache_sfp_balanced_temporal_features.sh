#!/usr/bin/env bash
# Rootless frozen three-view cache for the matched balanced-transport BC arms.
set -euo pipefail
merged_sfp=$(realpath "${1:?merged SFP labels JSONL}")
output=$(realpath -m "${2:?new feature cache output}")
gpu=${3:?GPU index}
repo=$(realpath "$(dirname "$0")/..")
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU index must be 0..7' >&2; exit 2; }
[[ -f "$merged_sfp" ]] || { echo "Missing SFP labels: $merged_sfp" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Preserve existing cache: $output" >&2; exit 2; }
mkdir -p "$(dirname "$output")"
sc="$repo/outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/transport_chunks.jsonl"
spec="$repo/outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json"
sfp_cache=/var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
resnet=/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
image='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
printf '%s\n' "merged_sfp=$merged_sfp" "output=$output" "gpu=$gpu" \
  "sc=$sc" "sc_spec=$spec" "sfp_image_cache=$sfp_cache" "checkpoint=$base" \
  "resnet=$resnet" > "$output.command.txt"
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
  --gpus "device=$gpu" --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act \
  -v /var/tmp/aic_sc_fix_20260924:/var/tmp/aic_sc_fix_20260924 \
  -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
  -v /home/nas/chmin/aic_llb_collection_20260930:/home/nas/chmin/aic_llb_collection_20260930:ro \
  -w "$repo" "$image" -lc \
  'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
  _ .pixi/envs/default/bin/python scripts/cache_temporal_transport_features.py \
  --sfp-labels "$merged_sfp" --sc-labels "$sc" --sc-spec "$spec" \
  --sfp-cache "$sfp_cache" --checkpoint "$base" --output "$output" \
  --feature-source imagenet_views --imagenet-resnet "$resnet" \
  > "$output.log" 2>&1
"$repo/.pixi/envs/default/bin/python" "$repo/scripts/make_temporal_camera_offsets.py" \
  --cache "$output" --sc-rows "$sc" >> "$output.log" 2>&1
