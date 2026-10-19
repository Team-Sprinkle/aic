#!/usr/bin/env bash
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
out=$(realpath -m "${1:?new output archive}")
gpu=${2:?physical GPU index}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
[[ ! -e "$out" ]] || { echo "Preserve prior result: $out" >&2; exit 2; }
mkdir -p "$out"
cache=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_late15_v1
labels=$(realpath "${3:-$repo/outputs/experiments/2026-10-08_sfp_short_waypoint_feasibility_v1/labels.npz}")
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
prior=/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
for arm in teacher_only boundary15; do
  mass=0
  [[ "$arm" == boundary15 ]] && mass=.15
  command=(.pixi/envs/default/bin/python scripts/train_sfp_full_route_short_waypoint.py
    --cache "$cache" --base-checkpoint "$base" --prior "$prior"
    --labels "$labels" --output "$out/$arm" --actor-boundary-mass "$mass"
    --updates 2000 --seed 20261008)
  printf '%q ' "${command[@]}" > "$out/$arm.command.txt"
  printf '\n' >> "$out/$arm.command.txt"
  DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
    --gpus "device=$gpu" --network none --shm-size 8g --entrypoint bash \
    -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
    -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act \
    -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
    -w "$repo" "$image" -lc \
    'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
    _ "${command[@]}" > "$out/$arm.log" 2>&1
  .pixi/envs/default/bin/python - "$out" "$arm" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);arm=sys.argv[2]
s=json.loads((root/arm/'summary.json').read_text())
p=root/'progress.json';d=json.loads(p.read_text()) if p.exists() else {}
d[arm]=dict(best_update=s['best_update'],teacher_moving_p95=s['teacher']['moving_ge10mm']['error_mm_p95'],boundary_p95=s['boundary']['all']['error_mm_p95'],gate=s['offline_gate'])
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({arm:d[arm]}))
PY
done
