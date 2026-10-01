#!/usr/bin/env python3
"""Run a scored SC LLB collection manifest, with a hard 30-attempt pilot gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923"
BASE_MEASURED_START = {
    "shoulder_pan_joint": -0.15803, "shoulder_lift_joint": -1.35146,
    "elbow_joint": -1.69620, "wrist_1_joint": -1.66472,
    "wrist_2_joint": 1.57073, "wrist_3_joint": 1.41262,
}
TEACHER_ENV = {
    "AIC_MOVEIT_PILOT_GPU": "7",
    "AIC_MOVEIT_PILOT_ACTION_MODE": "absolute_cartesian_pose_base_link",
    "AIC_MOVEIT_PILOT_ONLINE_INSERTION": "true",
    "AIC_MOVEIT_PILOT_TRACKING_GATE_FORCE_FRACTION": "6.0",
    "AIC_MOVEIT_PILOT_FORCE_CONFIRM_THRESHOLD_N": "2.0",
    "AIC_MOVEIT_PILOT_NOMINAL_INITIAL_LATERAL_GATE_M": ".012",
    "AIC_MOVEIT_PILOT_PRECONTACT_PORT_ALIGN_SEC": "1.2",
    "AIC_MOVEIT_PILOT_SC_TRACKING_GATE_SPEED_MPS": ".008",
    "AIC_MOVEIT_PILOT_PRECONTACT_SOFT_FORCE_N": "1.5",
    "AIC_MOVEIT_PILOT_PRECONTACT_FATAL_FORCE_N": "2.0",
    "AIC_MOVEIT_PILOT_SC_Z_GATE_MAX_HOLD_SEC": "1.5",
    "AIC_MOVEIT_PILOT_SC_CONTACT_RECOVERY": "true",
    "AIC_MOVEIT_PILOT_GUARDED_LATERAL_SERVO": "true",
    "AIC_MOVEIT_PILOT_SC_RECOVERY_MAX_BACKOFF_M": ".04",
    "AIC_MOVEIT_PILOT_SC_RECOVERY_BACKOFF_SEC": "1.2",
    "AIC_MOVEIT_PILOT_SC_RECOVERY_REQUIRED_MEASURED_M": ".015",
}


def run(command, logfile, *, timeout, env=None):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as output:
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=output,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            return {"returncode": result.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "timed_out": True}


def scored_result(audit):
    path = audit / "results/scoring.yaml"
    if not path.exists():
        return {"success": False, "score_available": False, "reason": "no scoring.yaml"}
    score = yaml.safe_load(path.read_text())
    trial = next((value for key, value in score.items() if key.startswith("trial_")), None)
    if not trial:
        return {"success": False, "score_available": False, "reason": "no trial score"}
    tier3 = trial.get("tier_3", {})
    value = float(tier3.get("score", 0))
    return {
        "success": value >= 75.0,
        "score_available": True,
        "tier3": value,
        "tier3_message": tier3.get("message"),
        "total": score.get("total"),
        "reason": "full insertion" if value >= 75.0 else "no scored full insertion",
    }


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_run(audit, archive_root, *, require_bag):
    """Copy first, verify retained files, then replace local with a link."""
    archive_root.mkdir(parents=True, exist_ok=True)
    final = archive_root / audit.name
    staging = archive_root / (audit.name + ".incomplete")
    if final.exists() or audit.is_symlink():
        raise FileExistsError(f"Archive target already exists: {final}")
    staging.mkdir(exist_ok=True)
    result = subprocess.run(["rsync", "-a", "--no-perms", "--no-owner", "--no-group",
                             "--", str(audit) + "/", str(staging) + "/"],
                            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, timeout=900)
    if result.returncode:
        raise RuntimeError(f"rsync exit {result.returncode}: {result.stderr[-500:]}")
    checks = ["results/scoring.yaml", "eval_config.yaml", "moveit_joint_trajectory.json"]
    if (audit / "runtime_trace.jsonl").exists():
        checks.append("runtime_trace.jsonl")
    if (audit / "start_joint_audit.json").exists():
        checks.append("start_joint_audit.json")
    for relative in checks:
        source, copy = audit / relative, staging / relative
        if not source.exists() or not copy.exists() or sha(source) != sha(copy):
            raise RuntimeError(f"Archive verification failed: {relative}")
    source_bags = list((audit / "results").glob("bag_*/metadata.yaml"))
    if (require_bag and not source_bags) or not all(
        (staging / path.relative_to(audit)).exists() for path in source_bags
    ):
        raise RuntimeError("Archive bag metadata missing")
    staging.rename(final)
    temporary = audit.with_name(audit.name + ".archived_local")
    audit.rename(temporary)
    audit.symlink_to(final, target_is_directory=True)
    import shutil
    shutil.rmtree(temporary)
    return str(final)


def audit_actual_start(scene, audit):
    """Verify Gazebo applied a varied robot start in the first scored trial."""
    bags = sorted((audit / "results").glob("bag_*/metadata.yaml"))
    if len(bags) != 1:
        raise RuntimeError(f"Expected one scored bag, found {len(bags)}")
    report = audit / "start_joint_audit.json"
    relative_bag = bags[0].parent.relative_to(ROOT)
    relative_report = report.relative_to(ROOT)
    command = ["docker", "run", "--rm", "--net=none", "-v", f"{ROOT}:/repo",
               "-w", "/repo", "--entrypoint", "bash", IMAGE, "-lc",
               "source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; "
               f"python3 scripts/audit_sc_llb_start_pose.py {relative_bag} {relative_report}"]
    env = dict(os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", "unix:///run/user/1008/docker.sock"))
    with (audit / "start_joint_audit.log").open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"Scored start bag audit failed with exit {result.returncode}")
    samples = json.loads(report.read_text())["samples"]
    if not samples:
        raise RuntimeError("No scored startup joint observation")
    actual = samples[0]["positions"]
    predicted_state = json.loads((scene / "measured_start_joint_state.json").read_text())
    predicted = dict(zip(predicted_state["name"], predicted_state["position"]))
    errors = {joint: abs(actual[joint] - predicted[joint]) for joint in predicted}
    deviation = {joint: abs(actual[joint] - BASE_MEASURED_START[joint]) for joint in predicted}
    valid = max(errors.values()) <= 0.015 and max(deviation.values()) >= 0.003
    return {"valid": valid, "actual": actual, "predicted": predicted,
            "absolute_error_rad": errors, "deviation_from_fixed_start_rad": deviation,
            "max_error_rad": max(errors.values()),
            "max_deviation_from_fixed_start_rad": max(deviation.values()),
            "scored_bag": str(bags[0].parent)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--passed-pilot-progress", type=Path,
                        help="Required to run a continuation manifest beginning after case 29")
    parser.add_argument("--archive-root", type=Path,
                        help="Stage scored run files to this NAS directory before removing local copies")
    args = parser.parse_args()
    collection = args.collection.resolve()
    manifest = json.loads((collection / "manifest.json").read_text())
    continuation = manifest["first_attempt_index"] >= 30
    if continuation:
        if args.pilot_only or args.passed_pilot_progress is None:
            parser.error("Continuation requires --passed-pilot-progress and no --pilot-only")
        pilot = json.loads(args.passed_pilot_progress.read_text())
        if pilot.get("pilot_gate") != "pass" or len(pilot.get("attempts", {})) != 30:
            parser.error("The first 30 attempts have not passed the 10-success gate")
        start_manifest = collection / "start_variant_manifest.json"
        if not start_manifest.exists() or len(json.loads(start_manifest.read_text()).get("cases", [])) != len(manifest["cases"]):
            parser.error("Actual Gazebo start-variant manifest is incomplete")
    elif args.passed_pilot_progress:
        parser.error("--passed-pilot-progress is only for post-pilot cases")
    cases = manifest["cases"][:30] if args.pilot_only else manifest["cases"]
    progress_path = collection / "collection_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "schema": "aic_sc_llb_collection_progress/v1",
        "pilot_requested": 30,
        "pilot_required_successes": 10,
        "attempts": {},
        "full_successes": 0,
        "pilot_gate": "pending",
        "final_scenes_opened": False,
    }
    for case in cases:
        name = case["name"]
        if name in progress["attempts"]:
            continue
        start = time.time()
        scene = collection / "scenes" / name
        plan = collection / "plans" / name
        audit = collection / "runs" / name
        outcome = {"case": name, "selected_port": case["selected_port"],
                   "card_count": case["card_count"], "scene": str(scene),
                   "plan": str(plan), "audit": str(audit)}
        try:
            if not (plan.with_name(name + "_insert") / "moveit_joint_trajectory.json").exists():
                orientation = ",".join(str(value) for value in case["aligned_orientation_xyzw"])
                outcome["planning"] = run(
                    ["bash", "scripts/run_moveit_transport_plan.sh", str(scene), str(plan), orientation],
                    scene / "plan_launcher.log", timeout=420,
                )
            else:
                outcome["planning"] = {"returncode": 0, "reused_existing": True}
            insertion_plan = plan.with_name(name + "_insert")
            if not (insertion_plan / "moveit_joint_trajectory.json").exists():
                outcome["success"] = False
                outcome["reason"] = "MoveIt plan or insertion handoff failed"
            else:
                env = dict(os.environ, **TEACHER_ENV,
                           AIC_MOVEIT_PILOT_SCENE_DIR=str(scene))
                if not (audit / "results/scoring.yaml").exists():
                    outcome["replay"] = run(
                        ["bash", "scripts/run_moveit_sc_route_pilot.sh", str(insertion_plan), str(audit)],
                        scene / "replay_launcher.log", timeout=1200, env=env,
                    )
                else:
                    outcome["replay"] = {"returncode": 0, "reused_existing": True}
                outcome.update(scored_result(audit))
                if continuation and outcome["score_available"]:
                    try:
                        outcome["start_contract"] = audit_actual_start(scene, audit)
                    except Exception as error:
                        outcome["start_contract"] = {"valid": False,
                                                     "error": f"{type(error).__name__}: {error}"}
                if args.archive_root and outcome["score_available"]:
                    try:
                        outcome["archive"] = archive_run(
                            audit, args.archive_root.resolve(), require_bag=outcome["success"]
                        )
                    except Exception as error:
                        outcome["archive_error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            outcome["success"] = False
            outcome["reason"] = f"collection error: {type(error).__name__}: {error}"
        outcome["wall_seconds"] = round(time.time() - start, 2)
        progress["attempts"][name] = outcome
        progress["full_successes"] = sum(bool(item.get("success")) for item in progress["attempts"].values())
        if not continuation and len(progress["attempts"]) >= 30:
            progress["pilot_gate"] = "pass" if progress["full_successes"] >= 10 else "fail"
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps({"case": name, "success": outcome["success"],
                          "tier3": outcome.get("tier3"),
                          "attempted": len(progress["attempts"]),
                          "full_successes": progress["full_successes"],
                          "pilot_gate": progress["pilot_gate"]}), flush=True)
        if outcome.get("archive_error"):
            print(f"Archive failed for {name}; stop to preserve local disk: {outcome['archive_error']}", flush=True)
            break
        if continuation and outcome.get("score_available") and not outcome.get("start_contract", {}).get("valid"):
            print(f"Gazebo start contract failed for {name}; stop before the next case.", flush=True)
            break
        if continuation and len(progress["attempts"]) == 30 and progress["full_successes"] < 10:
            print("Varied-start continuation pilot failed 10-of-30; stop and diagnose.", flush=True)
            break
        if not continuation and len(progress["attempts"]) == 30 and progress["pilot_gate"] != "pass":
            print("Pilot failed: stop before attempt 31 and diagnose saved traces/videos.", flush=True)
            break


if __name__ == "__main__":
    main()
