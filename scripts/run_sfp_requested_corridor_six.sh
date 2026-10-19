#!/usr/bin/env bash
# Offline SFP requested-port corridor planning on saved nonfinal actor states.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
out=$(realpath -m "${1:?new output directory}")
[[ ! -e "$out" ]] || { echo "Preserve prior result: $out" >&2; exit 2; }
mkdir -p "$out"
set +u
source .pixi/envs/default/setup.bash
set -u
export PYTHONPATH="$repo/aic_teacher_official:$repo/scripts:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=95 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
early=/home/nas/chmin/aic_archives/2026-10-07_sfp_history_context_dev_takeover_v1/rollouts
pilot=/home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/rollouts
audits=(
  "$early/fresh_cards2_card3p1_n400__ec2_01_sfp_easy_trial_000122_actor5s_teacher_full"
  "$early/fresh_cards3_card2p0_n200__card2_port0_trial_000370_actor5s_teacher_full"
  "$early/fresh_cards5_card3p1_n400__ec2_02_sfp_hard_trial_000291_actor5s_teacher_full"
  "$pilot/episode2400089_card1p0_to_card4p1_actor5s_teacher_full"
  "$pilot/episode2400020_card2p1_to_card0p0_actor5s_teacher_full"
  "$pilot/episode2400110_card4p1_to_card0p0_actor5s_teacher_full"
)
for i in "${!audits[@]}"; do
  audit=${audits[$i]}
  [[ -f "$audit/scored_stream.json" ]] || { echo "Missing stream: $audit" >&2; exit 2; }
  command=(.pixi/envs/default/bin/python scripts/plan_sfp_actor_state_corridor.py
    --audit "$audit" --output "$out/case${i}.json" --buffer-mm 85)
  printf '%q ' "${command[@]}" > "$out/case${i}.command.sh"
  printf '\n' >> "$out/case${i}.command.sh"
  "${command[@]}" > "$out/case${i}.log" 2>&1 || {
    echo "Case $i failed: $out/case${i}.log" >&2; exit 1;
  }
  .pixi/envs/default/bin/python - "$out/case${i}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));print(json.dumps(dict(case=sys.argv[1],
    request=[d['request_card'],d['request_port']],fk_mm=d['start_fk_error_mm'],
    variants={k:v['success'] for k,v in d['route_variants'].items()})),flush=True)
PY
done
.pixi/envs/default/bin/python - "$out" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[json.loads((r/f'case{i}.json').read_text()) for i in range(6)]
if len({x['scene_group'] for x in rows})!=6:raise ValueError('Reset overlap')
names=('direct','align_then_approach','retreat_align_approach')
summary=dict(schema='aic_sfp_requested_port_corridor_six/v1',
  scene_groups=6,development_groups=3,train_groups=3,
  cases=[dict(case=i,split='development' if i<3 else 'train',scene=x['scene_group'],
              requested_card=x['request_card'],requested_port=x['request_port'],
              collision_objects=x['collision_objects_applied'],fk_error_mm=x['start_fk_error_mm'],
              variants={n:dict(success=x['route_variants'][n]['success'],
                               tcp_path_mm=x['route_variants'][n]['tcp_path_mm']) for n in names})
         for i,x in enumerate(rows)],
  successes={n:sum(x['route_variants'][n]['success'] for x in rows) for n in names},
  privileged_teacher_future_joint_goal_used=False,cable_collision_checked=False,
  gazebo_executed=False,full_route_gate_passed=False,final_scenes_opened=False)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))
PY
