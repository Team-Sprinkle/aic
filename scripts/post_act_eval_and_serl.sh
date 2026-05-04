#!/usr/bin/env bash
set -euo pipefail

cd /data1/chmin/yj/ws_aic/src/aic

RUN_DIR=/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/act/bc/20260502_act_20hz_chunk8_nact2_4gpu_eval_all_v1
DATASET_ROOT=/data1/chmin/yj/ws_aic/src/aic/outputs/hf_converted/sfp2nic_target_card0_port0_randomized_smoke/accepted_dataset_nominal_task_conditioned
FULL_EVAL_SUBDIR=runtime_eval_3trial
FULL_EVAL_CONFIG=/home/chmin/yj/ws_aic/src/aic/aic_engine/config/sample_config.yaml
BEST_JSON="$RUN_DIR/${FULL_EVAL_SUBDIR}_best_checkpoint.json"
BEST_CSV="$RUN_DIR/${FULL_EVAL_SUBDIR}_scores.csv"
SERL_OUTPUT=/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/serl/offline_adapter/20260504_from_best_act_3trial_200k

export PYGAME_HIDE_SUPPORT_PROMPT=1
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

echo "[post-act] Waiting for ACT training to finish..."
while pgrep -f "lerobot.scripts.lerobot_train .*20260502_act_20hz_chunk8_nact2_4gpu_eval_all_v1" >/dev/null; do
  date -u +"[post-act] %Y-%m-%dT%H:%M:%SZ ACT still running"
  sleep 120
done

echo "[post-act] ACT process finished. Checking target checkpoint."
if [ ! -f "$RUN_DIR/checkpoints/165000/pretrained_model/model.safetensors" ]; then
  echo "[post-act] ERROR: expected checkpoint missing: $RUN_DIR/checkpoints/165000/pretrained_model/model.safetensors" >&2
  exit 2
fi

echo "[post-act] Stopping lightweight ACT eval watcher if present."
if [ -f "$RUN_DIR/latest_act_eval_watcher_tmux_session.txt" ]; then
  tmux kill-session -t "$(cat "$RUN_DIR/latest_act_eval_watcher_tmux_session.txt")" 2>/dev/null || true
fi
mapfile -t eval_pids < <(
  pgrep -af "python .*scripts/evaluate_act_checkpoints_runtime.py .*--run-dir $RUN_DIR" \
    | awk '{print $1}' \
    || true
)
if [ "${#eval_pids[@]}" -gt 0 ]; then
  kill "${eval_pids[@]}" 2>/dev/null || true
fi
docker --host "$DOCKER_HOST" restart aic_eval >/dev/null || true

echo "[post-act] Running full official 3-trial eval for checkpoints."
pixi run python scripts/evaluate_act_checkpoints_runtime.py \
  --run-dir "$RUN_DIR" \
  --eval-subdir "$FULL_EVAL_SUBDIR" \
  --workspace-host /data1/chmin/yj/ws_aic/src/aic \
  --workspace-container /home/chmin/yj/ws_aic/src/aic \
  --engine-config "$FULL_EVAL_CONFIG" \
  --checkpoint-glob 'checkpoints/[0-9]*/pretrained_model' \
  --once-existing \
  --command-mode delta_pose \
  --command-frame gripper/tcp \
  --max-runtime-sec 180 \
  --start-delay-sec 1 \
  --control-hz 20 \
  --max-translation-delta 0.02 \
  --max-rotation-delta 0.2 \
  --sim-wait-sec 30 \
  --readiness-timeout-sec 240 \
  --engine-timeout-sec 900

echo "[post-act] Aggregating official scores and selecting best checkpoint."
pixi run python - <<'PY'
import csv
import json
from pathlib import Path

import yaml

run_dir = Path("/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/act/bc/20260502_act_20hz_chunk8_nact2_4gpu_eval_all_v1")
eval_root = run_dir / "runtime_eval_3trial"
rows = []
for summary_path in sorted(eval_root.glob("*/eval_summary.json")):
    step = summary_path.parent.name
    summary = json.loads(summary_path.read_text())
    scoring_path = summary_path.parent / "scoring.yaml"
    row = {
        "step": step,
        "checkpoint": str(run_dir / "checkpoints" / step / "pretrained_model"),
        "engine_returncode": summary.get("engine_returncode"),
        "policy_ready": summary.get("policy_ready"),
        "failure_reason": summary.get("failure_reason", ""),
        "total": None,
        "trial_1": None,
        "trial_2": None,
        "trial_3": None,
    }
    if scoring_path.exists():
        scoring = yaml.safe_load(scoring_path.read_text()) or {}
        row["total"] = scoring.get("total")
        for trial in ("trial_1", "trial_2", "trial_3"):
            value = scoring.get(trial, {})
            if isinstance(value, dict):
                row[trial] = (
                    (value.get("tier_1", {}) or {}).get("score", 0)
                    + (value.get("tier_2", {}) or {}).get("score", 0)
                    + (value.get("tier_3", {}) or {}).get("score", 0)
                )
    rows.append(row)

csv_path = run_dir / "runtime_eval_3trial_scores.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "checkpoint", "total", "trial_1", "trial_2", "trial_3", "engine_returncode", "policy_ready", "failure_reason"])
    writer.writeheader()
    writer.writerows(rows)

scored = [r for r in rows if isinstance(r["total"], (int, float))]
if not scored:
    raise SystemExit("No scored checkpoints found after full official evaluation")
best = max(scored, key=lambda r: (float(r["total"]), int(r["step"])))
best_path = run_dir / "runtime_eval_3trial_best_checkpoint.json"
best_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"scores_csv": str(csv_path), "best": best}, indent=2, sort_keys=True))
PY

ACT_CHECKPOINT=$(pixi run python - <<'PY'
import json
from pathlib import Path
best = json.loads(Path("/data1/chmin/yj/ws_aic/src/aic/outputs/train/sfp_to_nic/hf_sfp2nic_card0_port0_randomized/act/bc/20260502_act_20hz_chunk8_nact2_4gpu_eval_all_v1/runtime_eval_3trial_best_checkpoint.json").read_text())
print(best["checkpoint"])
PY
)

echo "[post-act] Starting ACT-backed offline SERL from best checkpoint: $ACT_CHECKPOINT"
mkdir -p "$SERL_OUTPUT"
cat > "$SERL_OUTPUT/launch_context.json" <<EOF
{
  "act_checkpoint": "$ACT_CHECKPOINT",
  "dataset_root": "$DATASET_ROOT",
  "source_act_run": "$RUN_DIR",
  "official_eval_scores_csv": "$BEST_CSV",
  "official_eval_best_json": "$BEST_JSON"
}
EOF

export CUDA_VISIBLE_DEVICES=2,3,4,5
pixi run python -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 4 \
  aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py \
  --dataset-root "$DATASET_ROOT" \
  --act-checkpoint "$ACT_CHECKPOINT" \
  --output-dir "$SERL_OUTPUT" \
  --job-name 20260504_from_best_act_3trial_200k \
  --steps 200000 \
  --batch-size 2 \
  --device cuda \
  --adapter-lr 5e-5 \
  --critic-lr 5e-5 \
  --lr 5e-5 \
  --act-lr 1e-6 \
  --gamma 0.99 \
  --tau 0.005 \
  --bc-weight 1.0 \
  --cql-weight 0.0 \
  --adapter-penalty-weight 0.05 \
  --act-preservation-weight 0.5 \
  --smoothness-weight 0.001 \
  --action-horizon 8 \
  --actor-mode act_adapter \
  --freeze-act \
  --adapter-hidden-dim 256 \
  --adapter-num-layers 2 \
  --adapter-scale 1.0 \
  --adapter-delta-clip 0.02 \
  --action-clip 0.05 \
  --reward-mode dataset \
  --dataset-video-backend pyav \
  --save-every 5000 \
  --val-fraction 0.1 \
  --val-every 5000 \
  --val-max-batches 32 \
  --early-stopping-metric bc_loss \
  --early-stopping-patience 0
