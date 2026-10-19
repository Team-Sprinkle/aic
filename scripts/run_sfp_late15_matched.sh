#!/usr/bin/env bash
# Matched continuation: isolate new 15-second teacher labels on an unchanged actor.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
cache=$(realpath "${1:?frozen feature cache}")
out=$(realpath -m "${2:?new comparison output}")
gpu=${3:?physical GPU index}
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU must be 0..7' >&2; exit 2; }
[[ -f "$cache/camera_offsets.npy" ]] || { echo 'Incomplete feature cache' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "Preserve existing comparison: $out" >&2; exit 2; }
mkdir -p "$out"
prior=${4:-/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt}
prior=$(realpath "$prior")
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
image=ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
filter="$out/control_exclude_new_late15.json"
"$repo/.pixi/envs/default/bin/python" - "$cache" "$filter" <<'PY'
import json,sys
from pathlib import Path
cache=Path(sys.argv[1]);out=Path(sys.argv[2])
rows=[json.loads(line) for line in (cache/'rows.jsonl').open()]
indices=[i for i,r in enumerate(rows) if r['split']=='train' and
         r['family']=='sfp' and 8200000<=int(r['episode'])<8400000]
if len(indices)<100:raise SystemExit('Too few admitted corrected training rows')
out.write_text(json.dumps(dict(schema='aic_sfp_stall_loss_filter/v1',
    cache=str(cache),indices=indices,purpose='matched control excludes new 15-second teacher action loss',
    final_scenes_opened=False),indent=2)+'\n')
print(json.dumps(dict(corrective_train_cache_rows=len(indices))))
PY
common=(--cache "$cache" --checkpoint "$base" --full-state-skip
        --long-history --no-route-heads --balance-sfp-targets
        --init-temporal "$prior"
        --frozen-reference "$prior" --reference-old-sfp-weight 1
        --selection-exclude-corrective --learning-rate 5e-5
        --corrective-sfp-mass .1
        --updates 2000 --min-updates 2000 --budget-minutes 90 --seed 20261007)
for variant in control corrected; do
  extra=(--excluded-train-indices "$filter")
  [[ "$variant" == corrected ]] && extra=()
  printf '%q ' .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
      "${common[@]}" "${extra[@]}" --output "$out/$variant" > "$out/$variant.command.txt"
  printf '\n' >> "$out/$variant.command.txt"
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
    _ .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
    "${common[@]}" "${extra[@]}" --output "$out/$variant" \
    > "$out/$variant.log" 2>&1
  "$repo/.pixi/envs/default/bin/python" - "$out" "$variant" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);variant=sys.argv[2]
result=json.loads((out/variant/'summary.json').read_text())
p=out/'progress.json';progress=json.loads(p.read_text()) if p.exists() else {}
progress[variant]=dict(best_update=result['best_update'],
                       updates_completed=result['updates_completed'],
                       stop_reason=result['stop_reason'],checkpoint=str(out/variant/'best.pt'))
p.write_text(json.dumps(progress,indent=2)+'\n')
print(json.dumps({variant:progress[variant]}))
PY
done
