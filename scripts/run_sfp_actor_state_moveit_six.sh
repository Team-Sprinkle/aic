#!/usr/bin/env bash
# Three development plus three train SFP actor takeovers. Offline planning only.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
out=$(realpath -m "${1:?new output directory}")
[[ ! -e "$out" ]] || { echo "Preserve prior result: $out" >&2; exit 2; }
mkdir -p "$out"
set +u
source .pixi/envs/default/setup.bash
set -u
export PYTHONPATH="$repo/aic_teacher_official:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=94 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
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
  [[ -f "$audit/scored_stream.json" && -f "$audit/eval_config.yaml" ]] || {
    echo "Missing saved evidence: $audit" >&2; exit 2;
  }
  command=(.pixi/envs/default/bin/python scripts/probe_sfp_actor_state_moveit_path.py
    --audit "$audit" --output "$out/case${i}.json" --net-mm 10 --max-lookahead-s 4)
  printf '%q ' "${command[@]}" > "$out/case${i}.command.sh"
  printf '\n' >> "$out/case${i}.command.sh"
  "${command[@]}" > "$out/case${i}.log" 2>&1 || {
    echo "Case $i failed; inspect $out/case${i}.log" >&2; exit 1;
  }
  .pixi/envs/default/bin/python - "$out/case${i}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));print(json.dumps(dict(case=sys.argv[1],success=d['moveit_success'],
    fk_mm=d['start_fk_error_mm'],objects=d['collision_objects_applied'],points=d['trajectory_points'])))
PY
done
.pixi/envs/default/bin/python - "$out" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[json.loads((r/f'case{i}.json').read_text()) for i in range(6)]
scenes=[x['scene_group'] for x in rows]
if len(set(scenes))!=6:raise ValueError('Development scene overlap')
summary=dict(schema='aic_sfp_actor_state_moveit_six/v1',
    cases=[dict(case=i,split='development' if i<3 else 'train',scene_group=x['scene_group'],request=x['request']['target_module_name']+'/'+x['request']['port_name'],
                success=x['moveit_success'],fk_mm=x['start_fk_error_mm'],
                objects=x['collision_objects_applied'],points=x['trajectory_points']) for i,x in enumerate(rows)],
    scene_groups=6,rigid_plan_successes=sum(x['moveit_success'] for x in rows),
    target_available=sum(x['target_available'] for x in rows),
    rigid_plan_gate_passed=(sum(x['moveit_success'] and x['start_fk_error_mm'] is not None and x['start_fk_error_mm']<=5 for x in rows)>=5 and
                            all(x['moveit_success'] and x['start_fk_error_mm'] is not None and x['start_fk_error_mm']<=5 for x in rows[:3])),
    cable_collision_checked=False,gazebo_executed=False,autonomous_success_claim=False,
    final_scenes_opened=False)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))
PY
