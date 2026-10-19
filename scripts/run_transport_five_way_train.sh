#!/usr/bin/env bash
set -euo pipefail

# Rootless, bounded supervised comparison. Run one branch per free GPU.
branch=${1:?branch: no_history|long_history|route|diffusion|sfp_only|sc_only}
gpu=${2:?GPU index}
updates=${3:-3000}
budget_minutes=${4:-60}
repo=$PWD
out=outputs/experiments/2026-10-01_transport_five_way
cache=$out/imagenet_features
base=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
[[ -f "$cache/camera_offsets.npy" ]] || { echo 'Missing completed feature cache' >&2; exit 1; }
[[ ! -e "$out/$branch" ]] || { echo "Output already exists: $out/$branch" >&2; exit 1; }
case "$branch" in
  no_history) script=scripts/train_temporal_transport_probe.py; extra=(--full-state-skip --no-route-heads --no-history) ;;
  long_history) script=scripts/train_temporal_transport_probe.py; extra=(--full-state-skip --no-route-heads --long-history) ;;
  route) script=scripts/train_temporal_transport_probe.py; extra=(--full-state-skip --long-history) ;;
  diffusion) script=scripts/train_diffusion_transport_probe.py; extra=() ;;
  sfp_only|sfp_only_3000) script=scripts/train_temporal_transport_probe.py; extra=(--full-state-skip --no-route-heads --long-history --task-family sfp) ;;
  sc_only|sc_only_3000) script=scripts/train_temporal_transport_probe.py; extra=(--full-state-skip --no-route-heads --long-history --task-family sc) ;;
  *) echo "Unknown branch: $branch" >&2; exit 1 ;;
esac
mkdir -p "$out"
cat > "$out/$branch-command.txt" <<EOF
branch=$branch gpu=$gpu updates=$updates budget_minutes=$budget_minutes
script=$script
extra=${extra[*]}
cache=$cache
base=$base
EOF
DOCKER_HOST=unix:///run/user/1008/docker.sock docker run --rm --gpus "device=$gpu" \
  --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -w "$repo" \
  ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923 \
  -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
  _ .pixi/envs/default/bin/python "$script" --cache "$cache" --checkpoint "$base" \
  --output "$out/$branch" --updates "$updates" --budget-minutes "$budget_minutes" "${extra[@]}" \
  > "$out/$branch.log" 2>&1
