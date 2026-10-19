#!/usr/bin/env bash
# One bounded full-action continuation from the frozen ordinary temporal actor.
set -euo pipefail
gpu=${1:?GPU index}
cache=$(realpath "${2:?corrective feature cache}")
prior=$(realpath "${3:?frozen ordinary temporal checkpoint}")
output=$(realpath -m "${4:?new model output}")
repo=$(realpath "$(dirname "$0")/..")
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU index must be 0..7' >&2; exit 2; }
[[ -f "$cache/camera_offsets.npy" && -f "$prior" ]] || { echo 'Missing cache or prior' >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Preserve prior model: $output" >&2; exit 2; }
mkdir -p "$(dirname "$output")"
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
image='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
printf '%s\n' "gpu=$gpu" "cache=$cache" "init_temporal=$prior" "output=$output" \
  'updates=5000 min_updates=1000 budget_minutes=60 corrective_sfp_mass=.3 learning_rate=5e-5' \
  > "$output.command.txt"
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
  --gpus "device=$gpu" --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act \
  -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
  -w "$repo" "$image" -lc \
  'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
  _ .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache "$cache" --checkpoint "$base" --output "$output" --init-temporal "$prior" \
  --full-state-skip --no-route-heads --long-history --task-family both \
  --balance-sfp-targets --corrective-sfp-mass .3 --learning-rate 5e-5 \
  --updates 5000 --min-updates 1000 --budget-minutes 60 > "$output.log" 2>&1
