#!/usr/bin/env bash
# One preregistered matched data-only vs paired-contrast actor continuation.
set -euo pipefail
cache=$(realpath "${1:?route/handoff cache}")
pairs=$(realpath "${2:?feature-matched pair manifest}")
out=$(realpath -m "${3:?new result folder}")
repo=$(realpath "$(dirname "$0")/..")
[[ ! -e "$out" ]] || { echo "Preserve existing result folder: $out" >&2; exit 2; }
"$repo/.pixi/envs/default/bin/python" - "$pairs" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
train=len(d['train_pairs']);dev=len(d['validation_pairs'])
print(json.dumps({'audited_train_pairs':train,'audited_development_pairs':dev}))
if train<30 or dev<4 or not d['sufficient_for_fit']:
    raise SystemExit('Pair gate failed: require >=30 train and >=4 development pairs')
PY
mkdir -p "$out"
base="$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt"
init="$repo/outputs/experiments/2026-10-06_sfp_diverse_transport/diverse_matched_v1/best.pt"
common=(--cache "$cache" --checkpoint "$base" --full-state-skip --long-history
        --no-route-heads --balance-sfp-targets --paired-sfp-mass .5
        --init-temporal "$init" --updates 2000 --min-updates 2000
        --budget-minutes 120 --seed 20261007 --paired-contrast-manifest "$pairs")
printf '%q ' .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py "${common[@]}" \
  --paired-contrast-weight 0 --output "$out/control" > "$out/control.command.txt"
printf '\n' >> "$out/control.command.txt"
printf '%q ' .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py "${common[@]}" \
  --paired-contrast-weight .5 --output "$out/paired" > "$out/paired.command.txt"
printf '\n' >> "$out/paired.command.txt"
(
  cd "$repo"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 \
    "$repo/.pixi/envs/default/bin/python" scripts/train_temporal_transport_probe.py \
    "${common[@]}" --paired-contrast-weight 0 --output "$out/control" \
    > "$out/control.log" 2>&1
) & control=$!
(
  cd "$repo"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 \
    "$repo/.pixi/envs/default/bin/python" scripts/train_temporal_transport_probe.py \
    "${common[@]}" --paired-contrast-weight .5 --output "$out/paired" \
    > "$out/paired.log" 2>&1
) & candidate=$!
rc_control=0;rc_candidate=0
wait "$control" || rc_control=$?
wait "$candidate" || rc_candidate=$?
"$repo/.pixi/envs/default/bin/python" - "$out" "$rc_control" "$rc_candidate" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);result={'control_exit_code':int(sys.argv[2]),'candidate_exit_code':int(sys.argv[3])}
(out/'process_status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
(( rc_control == 0 && rc_candidate == 0 ))
