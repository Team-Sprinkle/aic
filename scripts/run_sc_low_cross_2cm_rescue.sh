#!/usr/bin/env bash
# Repeat the fixed September 26 SC port-0 failure and vary only route margins.
set -euo pipefail

repo="$(realpath "$(dirname "$0")/..")"
source_dir=/var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926
root="${AIC_RESCUE_ROOT:-/var/tmp/chmin_aic_20260918_act/sc_low_cross_2cm_rescue_20260929}"
case_name="${1:?case: repeat_failure | behind_plus20 | lane_plus20 | behind_lane_plus20 | forward_plus20}"
replicate="${2:-1}"
[[ "$replicate" =~ ^[0-9]+$ ]] || { echo "Replicate must be numeric" >&2; exit 2; }
case "$case_name" in
  repeat_failure) behind=0; lane=0; forward=-0.01 ;;
  behind_plus20) behind=0.02; lane=0; forward=-0.01 ;;
  lane_plus20) behind=0; lane=0.02; forward=-0.01 ;;
  behind_lane_plus20) behind=0.02; lane=0.02; forward=-0.01 ;;
  forward_plus20) behind=0; lane=0; forward=0.01 ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac

dir="$root/${case_name}_r${replicate}"
if test -e "$dir/results/scoring.yaml"; then
  echo "Scored case already exists: $dir" >&2
  exit 1
fi
mkdir -p "$dir"
cp "$source_dir"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} "$dir/"
sha256sum "$dir"/{eval_config.yaml,world_audit.sdf,bridge_audit.yaml} > "$dir/scene_sha256.txt"
printf '{"case":"%s","route_variant":"outside_left_low_cross_return","behind_extra_m":%s,"lane_extra_m":%s,"forward_extra_m":%s,"source_failed_run":"%s"}\n' \
  "$case_name" "$behind" "$lane" "$forward" "$source_dir" > "$dir/variation.json"

export DOCKER_HOST=unix:///run/user/1008/docker.sock
export AIC_AUDIT_SMOOTH_CAPTURE=1
export AIC_AUDIT_DEADLINE_SEC=900
export AIC_CABLE_ROUTE_BEHIND_EXTRA_M="$behind"
export AIC_CABLE_ROUTE_LANE_EXTRA_M="$lane"
export AIC_CABLE_ROUTE_FORWARD_EXTRA_M="$forward"
echo "Running $case_name in $dir" >&2
set +e
bash "$repo/scripts/run_recovery_gazebo_diagnostic.sh" "$dir" outside_left_low_cross_return \
  > "$dir/launcher.log" 2>&1
runner_status=$?
set -e
printf '%s\n' "$runner_status" > "$dir/runner_status.txt"
if ! test -s "$dir/results/scoring.yaml"; then
  echo "No score for $case_name; see $dir/launcher.log" >&2
  exit 1
fi
"${AIC_RESCUE_PYTHON:-$repo/.pixi/envs/default/bin/python}" - "$dir" <<'PY'
import json, sys, yaml
from pathlib import Path
d = Path(sys.argv[1])
s = yaml.safe_load((d/'results/scoring.yaml').read_text())
t = s['trial_000001']
out = {'case': d.name, 'runner_status': int((d/'runner_status.txt').read_text()),
       'tier3': float(t['tier_3']['score']), 'tier3_message': t['tier_3']['message'],
       'total': float(s['total']), 'route_plan': str(d/'route_plan.json'),
       'scoring': str(d/'results/scoring.yaml')}
(d/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
PY
