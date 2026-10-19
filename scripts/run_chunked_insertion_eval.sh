#!/usr/bin/env bash
# Add the execution-count setting to the audited rootless launcher.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
export AIC_TRANSPORT_EVAL_REPO="$repo" AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT=1
exec "$repo/.pixi/envs/default/bin/python" - "$repo" "$@" <<'PY'
import os,subprocess,sys,tempfile
from pathlib import Path
repo=Path(sys.argv[1])
text=subprocess.check_output(['bash',str(repo/'scripts/run_shared_recovery_eval.sh'),'--print-generated-launcher'],text=True)
anchor=' -e AIC_RESULTS_DIR=/audit/results -e AIC_SHARED_PORT_TCP_CHECKPOINT="$checkpoint" \\\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+' -e AIC_INSERTION_EXECUTE_STEPS \\\n')
anchor='for key in keys:\n'
assert text.count(anchor)==1
text=text.replace(anchor,"keys=(*keys,'AIC_INSERTION_EXECUTE_STEPS')\n"+anchor)
if sys.argv[2:]==['--print-generated-launcher']:
 print(text,end='');sys.exit(0)
if 'AIC_INSERTION_EXECUTE_STEPS' not in os.environ:raise ValueError('Explicit execution count required')
fd,path=tempfile.mkstemp(prefix='aic_chunked_insertion_',suffix='.sh');os.close(fd)
Path(path).write_text(text)
try:result=subprocess.run(['bash',path,*sys.argv[2:]],env=os.environ)
finally:Path(path).unlink(missing_ok=True)
sys.exit(result.returncode)
PY
