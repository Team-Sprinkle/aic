#!/usr/bin/env bash
# Extend the frozen legacy launcher without modifying an active comparison.
# The generated launcher and actual container settings are retained per run.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
export AIC_TRANSPORT_EVAL_REPO="$repo"
export AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT=1
exec "$repo/.pixi/envs/default/bin/python" - "$repo" "$@" <<'PY'
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
repo=Path(sys.argv[1]); source=repo/'scripts/run_clean_transport_eval.sh'
text=source.read_text()
keys=('AIC_SC_NATIVE_POSE_CHECKPOINT','AIC_SFP_NATIVE_POSE_MAX_REPROJECTION_PX',
      'AIC_SC_TRANSPORT_ROUTE', 'AIC_SC_EARLY_TILT', 'AIC_SFP_FALLBACK_LIFT_M',
      'AIC_SFP_REQUESTED_VIEW_THRESHOLD','AIC_SFP_GRASP_CHECKPOINT',
      'AIC_SFP_GRASP_APPROVAL','AIC_SHARED_RECOVERY_MODE',
      'AIC_SHARED_RECOVERY_SIM_BUDGET_S','AIC_SHARED_RECOVERY_EXPLORATION_SEED')
# Docker -e KEY inherits an existing host value; an absent optional variable
# remains absent, so Python policy defaults are preserved (not empty strings).
anchor=' -e AIC_RESULTS_DIR=/audit/results -e AIC_SHARED_PORT_TCP_CHECKPOINT="$checkpoint" \\\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+''.join(f' -e {k} \\\n' for k in keys))
anchor2="docker exec -d \"$name\" bash -lc '\n source /ws_aic/install/setup.bash\n export CUDA_VISIBLE_DEVICES=\"$AIC_ENGINE_CUDA_VISIBLE_DEVICE\""
assert text.count(anchor2)==1
text=text.replace(anchor2,
    '# Save the effective runtime environment, not just host intentions.\n'
    'docker inspect "$name" --format \'{{json .Config.Env}}\' > "$audit/container_environment.json"\n'
    '"$repo/.pixi/envs/default/bin/python" - "$audit/container_environment.json" <<\'RUNTIME_ENV_AUDIT\'\n'
    'import json,os,sys\n'
    'actual=dict(x.split("=",1) for x in json.load(open(sys.argv[1])) if "=" in x)\n'
    f'keys={keys!r}\n'
    'for key in keys:\n'
    ' if key in os.environ and actual.get(key)!=os.environ[key]:\n'
    '  raise RuntimeError("Container did not receive requested setting: "+key)\n'
    'print("Effective runtime settings verified before engine startup",flush=True)\n'
    'RUNTIME_ENV_AUDIT\n'+anchor2)
if sys.argv[2:]==['--print-generated-launcher']:
    print(text,end='');sys.exit(0)
fd,name=tempfile.mkstemp(prefix='aic_shared_recovery_launcher_',suffix='.sh')
os.close(fd);Path(name).write_text(text)
try:
    result=subprocess.run(['bash',name,*sys.argv[2:]],env=os.environ)
finally:
    Path(name).unlink(missing_ok=True)
sys.exit(result.returncode)
PY
