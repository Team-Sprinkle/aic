#!/usr/bin/env bash
# Isolated wrapper: preserve all active legacy launchers and their frozen hashes.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
export AIC_TRANSPORT_EVAL_REPO="$repo" AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT=1
exec "$repo/.pixi/envs/default/bin/python" - "$repo" "$@" <<'PY'
import os,subprocess,sys,tempfile
from pathlib import Path
repo=Path(sys.argv[1])
text=subprocess.check_output(['bash',str(repo/'scripts/run_chunked_insertion_eval.sh'),'--print-generated-launcher'],text=True)
keys=('AIC_SC_CLOSE_GRASP_CHECKPOINT','AIC_SC_CLOSE_GRASP_APPROVAL')
forwarded=(*keys,'AIC_SC_CLOSE_DIAGNOSTIC_TRIAL','AIC_SFP_ORIENTATION_REPAIR_CHECKPOINT','AIC_SFP_ORIENTATION_REPAIR_APPROVAL','AIC_EMPTY_ROW_CHECKPOINT','AIC_EMPTY_ROW_APPROVAL')
anchor=' -e AIC_RESULTS_DIR=/audit/results -e AIC_SHARED_PORT_TCP_CHECKPOINT="$checkpoint" \\\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+''.join(f' -e {key} \\\n' for key in forwarded))
anchor='for key in keys:\n'
assert text.count(anchor)==1
text=text.replace(anchor,'keys=(*keys,'+','.join(repr(k) for k in forwarded)+')\n'+anchor)
if sys.argv[2:]==['--print-generated-launcher']:
    print(text,end='');sys.exit(0)
for key in keys:
    if not Path(os.environ[key]).is_file():raise ValueError('Missing frozen close pose input: '+key)
fd,path=tempfile.mkstemp(prefix='aic_close_grasp_',suffix='.sh');os.close(fd)
Path(path).write_text(text)
try:result=subprocess.run(['bash',path,*sys.argv[2:]],env=os.environ)
finally:Path(path).unlink(missing_ok=True)
sys.exit(result.returncode)
PY
