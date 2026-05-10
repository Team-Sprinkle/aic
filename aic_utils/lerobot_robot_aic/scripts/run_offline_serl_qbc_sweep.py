#!/usr/bin/env python3
"""Launch the q_bc vision offline SERL hyperparameter sweep."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "outputs/hf_combined/sfp2nic_card0_2_ports0_1_explicit_task_conditioned_contact_features"
DEFAULT_ACT = (
    REPO_ROOT
    / "outputs/train/sfp_to_nic/hf_sfp2nic_card0_card2_ports0_1_explicit_contact_features/act/bc/"
    / "20260509_act_contact_features_dec2_dropout0_bs8_250k_2gpu_v1/checkpoints/250000/pretrained_model"
)


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _run_one(
    *,
    run: dict[str, Any],
    gpu: int,
    args: argparse.Namespace,
    python: Path,
) -> dict[str, Any]:
    run_dir = args.output_root / run["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    eval_log = run_dir / "actor_eval.log"
    command_path = run_dir / "command.json"
    start = time.time()
    train_cmd = [
        str(python),
        "aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py",
        "--dataset-root",
        str(args.dataset_root),
        "--act-checkpoint",
        str(args.act_checkpoint),
        "--output-dir",
        str(run_dir),
        "--job-name",
        run["name"],
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda:0",
        "--num-workers",
        str(args.num_workers),
        "--action-horizon",
        "2",
        "--actor-mode",
        "act_adapter",
        "--actor-update-mode",
        "q_bc",
        "--freeze-act",
        "--adapter-arch",
        "mlp",
        "--adapter-activation",
        "gelu",
        "--adapter-delta-clip",
        str(run["adapter_delta_clip"]),
        "--adapter-lr",
        str(run["adapter_lr"]),
        "--bc-weight",
        str(run["bc_weight"]),
        "--critic-lr",
        str(run["critic_lr"]),
        "--critic-image-encoder",
        "small_conv",
        "--critic-arch",
        "multiplicative",
        "--critic-activation",
        "gelu",
        "--critic-feature-dim",
        "256",
        "--critic-hidden-dim",
        "256",
        "--critic-num-layers",
        "2",
        "--critic-per-camera-dim",
        "64",
        "--reward-mode",
        "dataset",
        "--state-encoding",
        "fourier",
        "--state-encoding-indices",
        "0",
        "1",
        "2",
        "13",
        "14",
        "15",
        "--state-encoding-num-bands",
        "4",
        "--state-encoding-max-freq",
        "8.0",
        "--state-encoding-scale",
        "10.0",
        "--cql-weight",
        "0.0",
        "--gamma",
        "0.99",
        "--tau",
        "0.005",
        "--adapter-penalty-weight",
        "0.001",
        "--act-preservation-weight",
        "1.0",
        "--smoothness-weight",
        "0.0",
        "--val-fraction",
        "0.05",
        "--val-every",
        "250",
        "--val-max-batches",
        "8",
        "--save-every",
        str(args.steps),
        "--camera-keys",
        "observation.images.center_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    command_path.write_text(
        json.dumps(
            {
                "gpu": gpu,
                "train_cmd": train_cmd,
                "env": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
                "params": run,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with train_log.open("w", encoding="utf-8") as log:
        train_proc = subprocess.run(train_cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    eval_result = None
    eval_rc = None
    if train_proc.returncode == 0:
        eval_cmd = [
            str(python),
            "aic_utils/lerobot_robot_aic/scripts/evaluate_vision_offline_serl_actor.py",
            "--dataset-root",
            str(args.dataset_root),
            "--checkpoint",
            str(run_dir / "checkpoint_latest.pt"),
            "--device",
            "cuda:0",
            "--batch-size",
            "8",
            "--max-batches",
            "64",
            "--num-workers",
            str(args.num_workers),
            "--output-json",
            str(run_dir / f"actor_eval_{args.steps:06d}.json"),
        ]
        with eval_log.open("w", encoding="utf-8") as log:
            eval_proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        eval_rc = eval_proc.returncode
        eval_result = _read_json(run_dir / f"actor_eval_{args.steps:06d}.json")
    duration = time.time() - start
    last_val = _last_jsonl(run_dir / "validation_metrics.jsonl")
    last_train = _last_jsonl(run_dir / "metrics.jsonl")
    result = {
        **run,
        "gpu": gpu,
        "output_dir": str(run_dir),
        "returncode": train_proc.returncode,
        "eval_returncode": eval_rc,
        "elapsed_sec": duration,
        "steps": args.steps,
        "last_validation": last_val,
        "last_train": last_train,
        "actor_eval": eval_result,
    }
    (run_dir / "sweep_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _score(result: dict[str, Any]) -> tuple[float, float, float]:
    if result["returncode"] != 0 or result.get("eval_returncode") not in (0, None):
        return (float("inf"), float("inf"), float("inf"))
    val = result.get("last_validation") or {}
    actor = result.get("actor_eval") or {}
    td = float(val.get("td_loss", float("inf")))
    final_l1 = float(actor.get("final_l1", float("inf")))
    q_abs = abs(float(val.get("q_mean", float("inf"))))
    return (td, final_l1, q_abs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--act-checkpoint", type=Path, default=DEFAULT_ACT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    python = REPO_ROOT / ".pixi/envs/default/bin/python"
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for clip, bc_weight, critic_lr in itertools.product([1e-6, 3e-6, 1e-5, 3e-5], [3.0, 5.0, 10.0], [3e-5, 1e-4]):
        runs.append(
            {
                "name": f"clip{clip:.0e}_bc{bc_weight:g}_alr1e-05_clr{critic_lr:.0e}".replace("+", ""),
                "adapter_delta_clip": clip,
                "bc_weight": bc_weight,
                "adapter_lr": 1e-5,
                "critic_lr": critic_lr,
            }
        )
    manifest = {
        "dataset_root": str(args.dataset_root),
        "act_checkpoint": str(args.act_checkpoint),
        "gpus": args.gpus,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "runs": runs,
    }
    (args.output_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = []
        for idx, run in enumerate(runs):
            gpu = args.gpus[idx % len(args.gpus)]
            futures.append(pool.submit(_run_one, run=run, gpu=gpu, args=args, python=python))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            ordered = sorted(results, key=_score)
            (args.output_root / "sweep_results_partial.json").write_text(
                json.dumps(ordered, indent=2, sort_keys=True) + "\n"
            )
            best = ordered[0]
            print(
                f"completed {len(results)}/{len(runs)}: {result['name']} rc={result['returncode']} "
                f"best={best['name']} score={_score(best)}",
                flush=True,
            )
    ordered = sorted(results, key=_score)
    (args.output_root / "sweep_results.json").write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n")
    if ordered:
        print(json.dumps({"best": ordered[0], "num_results": len(ordered)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
