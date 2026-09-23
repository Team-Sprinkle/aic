#!/usr/bin/env bash
set -euo pipefail

source_suite="$(realpath "${1:?source suite directory}")"
output_root="${2:?follow-up output directory}"
shift 2
test "$#" -gt 0
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$output_root"

for trial in "$@"; do
  index="${trial#trial_}"
  index="${index%%_*}"
  out="$output_root/$trial"
  name="aic_ordinary_fresh_${index}_20260923"
  .pixi/envs/default/bin/python "$here/extract_ordinary_broad_trial.py" \
    "$source_suite" "$trial" "$out"
  if AIC_AUDIT_CONTAINER_NAME="$name" bash "$here/run_ordinary_broad_suite.sh" "$out"; then
    printf '%s\t%s\n' "$trial" "$(cat "$out/engine.exit")" >> "$output_root/run_status.tsv"
  else
    printf '%s\t%s\n' "$trial" "runner_error" >> "$output_root/run_status.tsv"
  fi
  docker rm "$name" >/dev/null 2>&1 || true
done
