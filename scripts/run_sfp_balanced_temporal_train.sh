#!/usr/bin/env bash
# Matched rootless supervised SFP/SC transport heads on a frozen balanced cache.
set -euo pipefail
variant=${1:?ordinary or zero_previous}
gpu=${2:?GPU index}
cache=$(realpath "${3:?feature cache}")
output=$(realpath -m "${4:?new model output}")
case "$variant" in
  ordinary) extra=() ;;
  zero_previous) extra=(--zero-previous-command) ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
repo=$(realpath "$(dirname "$0")/..")
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU index must be 0..7' >&2; exit 2; }
[[ -f "$cache/camera_offsets.npy" ]] || { echo 'Missing complete feature cache' >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Preserve prior model: $output" >&2; exit 2; }
mkdir -p "$(dirname "$output")"
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
updates=${AIC_SFP_BALANCED_UPDATES:-3000}
min_updates=${AIC_SFP_BALANCED_MIN_UPDATES:-1000}
budget_minutes=${AIC_SFP_BALANCED_BUDGET_MINUTES:-60}
image='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
printf '%s\n' "variant=$variant" "gpu=$gpu" "cache=$cache" "checkpoint=$base" \
  "output=$output" "updates=$updates budget_minutes=$budget_minutes" "extra=${extra[*]}" > "$output.command.txt"
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
  --gpus "device=$gpu" --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act \
  -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
  -w "$repo" "$image" -lc \
  'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
  _ .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache "$cache" --checkpoint "$base" --output "$output" \
  --full-state-skip --no-route-heads --long-history --task-family both \
  --balance-sfp-targets --updates "$updates" --min-updates "$min_updates" --budget-minutes "$budget_minutes" \
  "${extra[@]}" > "$output.log" 2>&1
