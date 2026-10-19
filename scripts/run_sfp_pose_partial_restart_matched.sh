#!/usr/bin/env bash
# Bounded matched pose-conditioned actor test: warm versus reset full-action head.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
out=$(realpath -m "${1:?new output archive}")
gpu=${2:?physical GPU index}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
[[ ! -e "$out" ]] || { echo "Preserve previous result: $out" >&2; exit 2; }
mkdir -p "$out"
cache=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_late15_v1
values=/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/selected_port_route_values_v1/values.npy
prior=/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
filter=/home/nas/chmin/aic_archives/2026-10-07_sfp_port_route_recovery_matched_v1/matched_native_only_with_late15.json
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
for arm in warm reset_head; do
  extra=()
  [[ "$arm" == reset_head ]] && extra+=(--reset-action-head)
  command=(.pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py
    --cache "$cache" --checkpoint "$base" --output "$out/$arm"
    --full-state-skip --long-history --no-route-heads --balance-sfp-targets
    --init-temporal "$prior" --route-goal-values "$values"
    --excluded-train-indices "$filter"
    --corrective-sfp-mass .4 --selection-corrective-weight 1
    --learning-rate 5e-5 --updates 2000 --min-updates 2000
    --budget-minutes 90 --seed 20261008 "${extra[@]}")
  printf '%q ' "${command[@]}" > "$out/$arm.command.txt"
  printf '\n' >> "$out/$arm.command.txt"
  DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
    --gpus "device=$gpu" --network none --shm-size 8g --entrypoint bash \
    -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
    -v "$repo:$repo" -v /var/tmp/chmin_aic_20260918_act:/var/tmp/chmin_aic_20260918_act \
    -v /var/tmp/aic_sc_fix_20260924:/var/tmp/aic_sc_fix_20260924 \
    -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
    -w "$repo" "$image" -lc \
    'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"; exec "$@"' \
    _ "${command[@]}" > "$out/$arm.log" 2>&1
  .pixi/envs/default/bin/python - "$out" "$arm" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);arm=sys.argv[2]
s=json.loads((out/arm/'summary.json').read_text())
p=out/'progress.json';d=json.loads(p.read_text()) if p.exists() else {}
d[arm]=dict(best_update=s['best_update'],updates_completed=s['updates_completed'],
            stop_reason=s['stop_reason'],ordinary=s['best_validation']['overall'])
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({arm:d[arm]}))
PY
done
