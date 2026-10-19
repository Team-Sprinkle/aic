#!/usr/bin/env bash
# One bounded rootless SFP slot/paired-port fit. See docs/experiments/2026-10-04-sfp-slot-pair-identity.md.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
archive=/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1
output="$archive/fit_v3"
image='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
[[ ! -e "$output" ]] || { echo "Preserve previous fit: $output" >&2; exit 2; }
mkdir -p "$archive"
printf '%s\n' "gpu=1" "updates=2000" "minutes=120" "batch_size=6" "image=$image" \
  "labels=$archive/labels" "init=$archive/../2026-10-04_sfp_structured_slot_v1/best.pt" \
  "output=$output" > "$archive/fit_v3.command.txt"
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
  --gpus device=1 --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
  -w "$repo" "$image" -lc \
  'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; export PYTHONPATH=scripts:aic_utils/lerobot_robot_aic; exec "$@"' \
  _ .pixi/envs/default/bin/python scripts/train_sfp_slot_pair_identity.py \
  --labels "$archive/labels" --init "$archive/../2026-10-04_sfp_structured_slot_v1/best.pt" \
  --output "$output" --updates 2000 --minutes 120 --batch-size 6 \
  > "$archive/fit_v3.log" 2>&1
