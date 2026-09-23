#!/usr/bin/env python3
"""Build causal future connector trajectories for PoseDP/RPDP training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rpdp_geometry", HERE / "rpdp_geometry.py")
geo = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = geo
spec.loader.exec_module(geo)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--perception-cache", type=Path, required=True)
    p.add_argument("--cable-templates", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--trajectory-semantics",choices=("microstep_chunk","macro_horizon"),default="microstep_chunk",
                   help="Use all four recorded 50 ms target commands, or legacy 200 ms endpoints across decisions.")
    p.add_argument("--temporal-context", action=argparse.BooleanOptionalAction, default=False,
                   help="Append causal state change, prior executed chunk, and observation-only phase.")
    p.add_argument("--max-target-force-n",type=float,default=float("inf"),
                   help="Reject target rows at or above this causal force norm; useful for bounded DAgger pre-contact ablations.")
    return p.parse_args()


def file_id(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20): h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def tensor(value): return torch.as_tensor(value, dtype=torch.float32)


def geometry(item, post=False):
    m = item.get("metadata") or {}
    if post and (m.get("terminated") or m.get("truncated")):
        # Isaac auto resets before the ordinary post step diagnostics are
        # serialized.  The explicitly retained terminal observation is the
        # only causal pre reset endpoint for a terminal transition.
        terminal = item.get("terminal_observation")
        if not terminal or not terminal.get("insertion_geometry"):
            raise KeyError("terminal transition has no retained terminal geometry")
        g = terminal["insertion_geometry"]
    else:
        key = "post_step_insertion_geometry" if post else "causal_insertion_geometry"
        g = m.get(key) or {}
    return {
        "body_p": tensor(g["body_world_env0"]),
        "body_q": tensor(g["body_orientation_wxyz_by_env"][0]),
        "port_p": tensor(g["entrance_world_env0"]),
        "port_q": tensor(g["target_orientation_wxyz_by_env"][0]),
        "target_p": tensor(g["target_world_env0"]),
        "depth": float(g["signed_depth_m_env0"]),
        "lateral": float(g["lateral_error_m_env0"]),
        "orientation": float(g["orientation_error_rad_env0"]),
        "success": bool(g["success_geometry_by_env"][0]),
    }


def tcp_pose(item, post=False):
    m = item.get("metadata") or {}
    terminal = item.get("terminal_observation") if post and (m.get("terminated") or m.get("truncated")) else None
    state_source = terminal if terminal is not None else (item["next_obs"] if post else item["obs"])
    state = state_source["state"].reshape(-1)
    # State translation is expressed in the robot root frame, whereas the
    # diagnostic connector geometry is world-frame.  Use the retained world
    # TCP position and the state quaternion (the configured robot root has no
    # rotational offset in these scenes).
    if terminal is not None:
        bodies = terminal.get("all_body_insertion_geometry") or {}
    else:
        key = "post_step_all_body_insertion_geometry" if post else "causal_all_body_insertion_geometry"
        bodies = m[key]
    p = tensor(bodies["gripper_tcp"]["world_env0"])
    return p, state[[6, 3, 4, 5]].float()


def force_phase(item, g):
    m = item.get("metadata") or {}
    force = tensor(m.get("causal_force_xyz_n") or item["obs"]["state"][26:29]).norm().item()
    if force >= 10: return "contact", force
    if g["depth"] >= 0: return "insertion", force
    if g["depth"] >= -.003: return "alignment", force
    return "approach", force


def sequence_ids(transitions):
    ids, occurrence, previous, previous_done = [], -1, None, True
    for item in transitions:
        m = item.get("metadata") or {}; ep = (m.get("causal_episode") or {}).get("episode_id")
        if previous_done or ep != previous: occurrence += 1
        ids.append(f"occ-{occurrence:06d}:{ep}")
        previous, previous_done = ep, bool(m.get("terminated") or m.get("truncated"))
    return ids


def observation_phase(pose_mm, force_xyz):
    phase=torch.zeros(4);distance=torch.linalg.norm(pose_mm);force=torch.linalg.norm(force_xyz)
    if float(force)>=5.:phase[3]=1
    elif float(distance)>3.:phase[0]=1
    elif float(distance)>.75:phase[1]=1
    else:phase[2]=1
    return phase


def recorded_target_action(item):
    """Return the privileged teacher command recorded at this decision.

    ``action`` is the command that was actually sent after optional actor/guide
    blending.  Recovery collection deliberately blends those commands, so it
    is not the supervised target.  ``guide_action`` is the teacher proposal in
    the configured TCP body frame.  Terminal macro transitions can contain
    fewer than four executed microsteps; the collector pads ``guide_action``
    by repeating its last command, which would teach an artificial overshoot.
    Replace that padding with zero motion.
    """
    value = item.get("guide_action")
    if value is None:
        raise KeyError("transition has no recorded guide_action target")
    action = tensor(value).reshape(4, 6).clone()
    metadata = item.get("metadata") or {}
    microsteps = int(metadata.get("macro_microsteps", 4))
    if not 1 <= microsteps <= 4:
        raise ValueError(f"invalid macro_microsteps={microsteps}")
    if microsteps < 4:
        action[microsteps:] = 0
    return action


def load_split(name, source, cached, horizon, adapter, temporal_context=False,
               max_target_force_n=float("inf"),trajectory_semantics="microstep_chunk"):
    payload = torch.load(source, map_location="cpu", weights_only=False)
    transitions = payload["transitions"]; sequence = sequence_ids(transitions)
    by_sequence = defaultdict(list)
    for i, sid in enumerate(sequence): by_sequence[sid].append(i)
    position = {index: j for indices in by_sequence.values() for j, index in enumerate(indices)}
    lookup = {int(index): row for row, index in enumerate(cached["transition_indices"])}
    rows, rejected = [], Counter()
    adapter_translation_error, adapter_orientation_error = [], []
    observed_current_translation_error, observed_current_orientation_error = [], []
    target_vs_executed_translation, target_vs_executed_orientation = [], []
    terminal_target_padding_removed = 0
    for index in cached["transition_indices"]:
        index = int(index); item = transitions[index]; m = item.get("metadata") or {}
        try: current = geometry(item)
        except (KeyError, IndexError): rejected["missing_current_geometry"] += 1; continue
        sid = sequence[index]; indices = by_sequence[sid]; pos = position[index]
        # The first saved observation after Isaac auto reset can still contain
        # the previous episode's rendered frame. Deployment overrides these
        # decisions with requested-tip restoration, so they are neither valid
        # visual supervision nor executable policy labels.
        if pos == 0:
            rejected["reset_transient_stale_render"] += 1
            continue
        oracle_future, mask = [], []
        for step in range(1, horizon + 1):
            j = min(pos + step - 1, len(indices) - 1)
            future_item = transitions[indices[j]]
            try: fg = geometry(future_item, post=True)
            except (KeyError, IndexError): break
            # Every horizon is expressed in the current decision's fixed port frame.
            fp, fq = geo.relative_pose(current["port_p"], current["port_q"], fg["body_p"], fg["body_q"])
            oracle_future.append(geo.pose9(fp, fq)); mask.append(float(pos + step - 1 < len(indices)))
        if len(oracle_future) != horizon: rejected["missing_future_geometry"] += 1; continue
        oracle_cp, oracle_cq = geo.relative_pose(current["port_p"], current["port_q"], current["body_p"], current["body_q"])
        phase, force = force_phase(item, current)
        if force >= float(max_target_force_n):
            rejected["target_force_at_or_above_limit"] += 1
            continue
        c = lookup[index]
        # Match deployment: translation comes from the frozen visual pose
        # estimator (connector minus seated target center, in target frame),
        # shifted to the opening origin. Orientation is observation-only TCP
        # proprioception plus the fixed calibrated grasp transform.
        target_from_entrance, _ = geo.relative_pose(
            current["port_p"], current["port_q"], current["target_p"], current["port_q"])
        observed_cp = cached["pose_mm"][c] / 1000. + target_from_entrance
        state = item["obs"]["state"].reshape(-1).float()
        tcp_q_root = state[[6, 3, 4, 5]]
        connector_q_root = geo.quat_mul(tcp_q_root, adapter.tcp_to_connector_q)
        root_q = tensor([0, 0, 0, 1])
        target_q_root = geo.quat_mul(geo.quat_conjugate(root_q), current["port_q"])
        observed_cq = geo.quat_mul(geo.quat_conjugate(target_q_root), connector_q_root)
        # Build command-consistent *absolute* connector waypoints. Starting
        # from the same noisy observation used online, compose each recorded
        # TCP-body-frame expert chunk and then map the resulting TCP pose back
        # to the connector frame. This keeps RPDP's full port-relative pose
        # output while ensuring that its deterministic adapter reproduces the
        # demonstrated command instead of inheriting pose-estimator bias.
        command_future=[]
        tcp_p,tcp_q=adapter.connector_to_tcp(observed_cp,observed_cq)
        if trajectory_semantics=="microstep_chunk":
            try:
                action=recorded_target_action(item)
            except (KeyError, ValueError):
                rejected["missing_or_invalid_recorded_target_action"] += 1
                continue
            if horizon != 4:
                raise ValueError("microstep_chunk requires horizon=4")
            for delta in action:
                tcp_p,tcp_q=geo.compose(tcp_p,tcp_q,delta[:3],geo.rotvec_to_quat(delta[3:]))
                connector_p,connector_q=geo.compose(tcp_p,tcp_q,adapter.tcp_to_connector_p,
                                                     adapter.tcp_to_connector_q)
                command_future.append(geo.pose9(connector_p,connector_q))
        else:
            for step in range(1,horizon+1):
                raw=pos+step-1
                if raw < len(indices):
                    future_transition=transitions[indices[raw]]
                    try:
                        action=recorded_target_action(future_transition)
                    except (KeyError, ValueError):
                        rejected["missing_or_invalid_recorded_target_action"] += 1
                        break
                    for delta in action:
                        tcp_p,tcp_q=geo.compose(tcp_p,tcp_q,delta[:3],geo.rotvec_to_quat(delta[3:]))
                connector_p,connector_q=geo.compose(tcp_p,tcp_q,adapter.tcp_to_connector_p,
                                                     adapter.tcp_to_connector_q)
                command_future.append(geo.pose9(connector_p,connector_q))
        if len(command_future) != horizon:
            continue
        try:
            target_action=recorded_target_action(item)
        except (KeyError, ValueError):
            rejected["missing_or_invalid_recorded_target_action"] += 1
            continue
        executed_action=item["action"].reshape(4,6).float()
        valid_microsteps=int(m.get("macro_microsteps",4))
        if valid_microsteps < 4:
            terminal_target_padding_removed += 4-valid_microsteps
        valid=slice(0,valid_microsteps)
        target_vs_executed_translation.extend(
            ((target_action[valid,:3]-executed_action[valid,:3]).norm(dim=-1)*1000).tolist())
        target_vs_executed_orientation.extend(
            ((target_action[valid,3:]-executed_action[valid,3:]).norm(dim=-1)*180/math.pi).tolist())
        condition_parts=[cached["visual"][c],cached["visibility"][c],
                         cached["pose_mm"][c]/10.,
                         torch.sqrt(cached["pose_variance_mm2"][c].clamp_min(0))/10.,state]
        if temporal_context:
            previous_item=transitions[indices[pos-1]] if pos>0 else None
            previous_state=(previous_item["obs"]["state"].reshape(-1).float()
                            if previous_item is not None else state)
            previous_action=(previous_item["action"].reshape(-1).float()
                             if previous_item is not None else torch.zeros(24))
            condition_parts.extend((state-previous_state,previous_action,
                                    observation_phase(cached["pose_mm"][c],state[26:29])))
        rows.append({
            "episode_id": (m.get("causal_episode") or {}).get("episode_id"),
            "sequence_id": sid, "transition_index": index,
            "condition": torch.cat(condition_parts),
            "current_pose9": geo.pose9(observed_cp, observed_cq),
            "oracle_current_pose9": geo.pose9(oracle_cp, oracle_cq),
            "future_pose9": torch.stack(command_future),
            "oracle_future_pose9": torch.stack(oracle_future), "future_mask": torch.tensor(mask),
            "target_action": target_action,
            "executed_action": executed_action,
            "target_action_source": "recorded_guide_action_tcp_body_frame",
            "phase": phase, "force_n": force,
            "depth_m": current["depth"], "lateral_m": current["lateral"],
            "orientation_rad": current["orientation"], "success": current["success"],
            "tcp_to_connector_p": adapter.tcp_to_connector_p,
            "tcp_to_connector_q": adapter.tcp_to_connector_q,
        })
        observed_current_translation_error.append(float((observed_cp-oracle_cp).norm()*1000))
        observed_current_orientation_error.append(float(
            geo.quat_to_rotvec(geo.quat_mul(geo.quat_conjugate(observed_cq),oracle_cq)).norm()*180/math.pi))
    if not rows: raise RuntimeError(f"No rows accepted for {name}")
    for row in rows:
        p0, q0 = geo.unpack_pose9(row["current_pose9"])
        p1, q1 = geo.unpack_pose9(row["future_pose9"][0])
        if trajectory_semantics=="microstep_chunk":
            waypoint_p,waypoint_q=geo.unpack_pose9(row["future_pose9"])
            predicted=adapter.waypoint_chunk(p0,q0,waypoint_p,waypoint_q)
            p1,q1=waypoint_p[-1],waypoint_q[-1]
        else:
            predicted = adapter.action_chunk(p0, q0, p1, q1)
        # Compare the command endpoint in SE(3); simulator tracking error is reported separately.
        pp, pq = adapter.connector_to_tcp(p0, q0)
        for a in predicted: pp, pq = geo.compose(pp, pq, a[:3], geo.rotvec_to_quat(a[3:]))
        tp, tq = adapter.connector_to_tcp(p1, q1)
        adapter_translation_error.append(float((pp-tp).norm()*1000))
        adapter_orientation_error.append(float(geo.quat_to_rotvec(geo.relative_pose(pp,pq,tp,tq)[1]).norm()*180/math.pi))
        row["adapter_action"] = predicted
    phase_counts = Counter(r["phase"] for r in rows)
    return rows, {
        "source": file_id(source), "transitions": len(transitions), "accepted": len(rows),
        "rejected": dict(rejected), "sequence_count": len(set(r["sequence_id"] for r in rows)),
        "episode_count": len(set(r["episode_id"] for r in rows)), "phase_counts": dict(phase_counts),
        "adapter_endpoint_translation_error_mm_max": max(adapter_translation_error),
        "adapter_endpoint_orientation_error_deg_max": max(adapter_orientation_error),
        "observed_current_translation_error_mm": {
            "median": float(torch.tensor(observed_current_translation_error).median()),
            "p95": float(torch.quantile(torch.tensor(observed_current_translation_error), .95)),
        },
        "observed_current_orientation_error_deg": {
            "median": float(torch.tensor(observed_current_orientation_error).median()),
            "p95": float(torch.quantile(torch.tensor(observed_current_orientation_error), .95)),
        },
        "target_vs_executed_translation_difference_mm": {
            "median": float(torch.tensor(target_vs_executed_translation).median()),
            "p95": float(torch.quantile(torch.tensor(target_vs_executed_translation), .95)),
            "max": max(target_vs_executed_translation),
        },
        "target_vs_executed_orientation_difference_deg": {
            "median": float(torch.tensor(target_vs_executed_orientation).median()),
            "p95": float(torch.quantile(torch.tensor(target_vs_executed_orientation), .95)),
            "max": max(target_vs_executed_orientation),
        },
        "terminal_padded_target_microsteps_zeroed": terminal_target_padding_removed,
        "tcp_to_connector": {"translation_m": adapter.tcp_to_connector_p.tolist(),
                             "orientation_wxyz": adapter.tcp_to_connector_q.tolist()},
    }


def save_visualization(splits, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"reset_transient":"tab:gray","approach": "tab:blue", "alignment": "tab:orange",
              "insertion":"tab:green", "contact": "tab:red"}
    for name, ax in zip(("fit", "calibration", "development"), axes):
        rows = splits[name]
        for row in rows[::max(1, len(rows)//70)]:
            trajectory = torch.cat((row["current_pose9"][None], row["future_pose9"]), 0)[:, :3].numpy()*1000
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=.45, color=colors[row["phase"]])
            ax.scatter(trajectory[0,0], trajectory[0,1], s=7, color="black")
        ax.axhline(0,color="gray",lw=.5); ax.axvline(0,color="gray",lw=.5)
        ax.set(title=f"{name}: port-frame lateral paths", xlabel="port x (mm)", ylabel="port y (mm)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def calibrate_grasp_transform(replay_path, transition_indices):
    """Calibrate the fixed TCP->connector transform on fit episodes only."""
    transitions=torch.load(replay_path,map_location="cpu",weights_only=False)["transitions"]
    # The robot asset is mounted with Isaac wxyz root orientation (0,0,0,1).
    root_q=tensor([0,0,0,1]);positions=[];orientations=[]
    for index in transition_indices:
        item=transitions[int(index)];m=item["metadata"];g=m["causal_insertion_geometry"]
        tcp_world=tensor(m["causal_all_body_insertion_geometry"]["gripper_tcp"]["world_env0"])
        connector_world=tensor(g["body_world_env0"]);connector_q_world=tensor(g["body_orientation_wxyz_by_env"][0])
        state=item["obs"]["state"].reshape(-1);tcp_q_root=state[[6,3,4,5]].float()
        delta_root=geo.quat_apply(geo.quat_conjugate(root_q),connector_world-tcp_world)
        positions.append(geo.quat_apply(geo.quat_conjugate(tcp_q_root),delta_root))
        connector_q_root=geo.quat_mul(geo.quat_conjugate(root_q),connector_q_world)
        orientations.append(geo.quat_mul(geo.quat_conjugate(tcp_q_root),connector_q_root))
    positions=torch.stack(positions);orientations=torch.stack(orientations)
    reference=orientations[0]
    orientations=torch.where((orientations*reference).sum(-1,keepdim=True)<0,-orientations,orientations)
    mean_q=geo.quat_normalize(orientations.mean(0));residual=torch.stack([
        geo.quat_to_rotvec(geo.quat_mul(geo.quat_conjugate(mean_q),q)).norm() for q in orientations])
    return geo.ConnectorTCPAdapter(positions.mean(0),mean_q),{
        "fit_rows":len(positions),"robot_root_orientation_wxyz":root_q.tolist(),
        "translation_std_mm":(positions.std(0)*1000).tolist(),
        "orientation_residual_median_deg":float(residual.median()*180/math.pi),
        "orientation_residual_p95_deg":float(torch.quantile(residual,.95)*180/math.pi),
    }


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.manifest.read_text()); cache = torch.load(a.perception_cache, map_location="cpu", weights_only=False)
    # The perception cache indices address the merged replay used by its
    # producer, rather than the original per-split replay indices.
    merged = Path(manifest["artifacts"]["filtered_replay"])
    adapter,grasp_audit=calibrate_grasp_transform(merged,cache["splits"]["fit"]["transition_indices"])
    paths = {"fit": merged, "calibration": merged, "development": merged}
    splits, audit = {}, {}
    for name in paths:
        splits[name], audit[name] = load_split(name, paths[name], cache["splits"][name], a.horizon, adapter,
                                               temporal_context=a.temporal_context,
                                               max_target_force_n=a.max_target_force_n,
                                               trajectory_semantics=a.trajectory_semantics)
    bundle = {"schema_version": 1,
              "cadence_ms":50 if a.trajectory_semantics=="microstep_chunk" else 200,
              "decision_cadence_ms":200,"horizon": a.horizon,
              "trajectory_semantics":a.trajectory_semantics,
              "pose_contract": "recorded teacher target TCP-body-frame commands composed into absolute future connector SE(3) in the current fixed port-opening frame; xyz metres + rotation 6D",
              "target_action_contract": "guide_action recorded before actor/teacher blending; TCP body frame; unexecuted terminal padding is zero motion",
              "oracle_pose_contract": "achieved future connector SE(3), retained for diagnostics only",
              "condition_contract": ("observation-only visual/visibility/predicted translation+uncertainty/state"
                                     + ("/state-delta/prior-executed-chunk/phase" if a.temporal_context else "")),
              "temporal_context":bool(a.temporal_context),
              "max_target_force_n":float(a.max_target_force_n),
              "splits": splits}
    torch.save(bundle, a.output_dir / "rpdp_dataset.pt")
    save_visualization(splits, a.output_dir / "future_trajectory_overview.png")
    summary = {"schema_version": 1, "status": "prepared",
               "cadence_ms":50 if a.trajectory_semantics=="microstep_chunk" else 200,
               "decision_cadence_ms":200,"trajectory_semantics":a.trajectory_semantics,
               "horizon": a.horizon, "audit": audit,
               "grasp_transform_audit": grasp_audit,
               "sources": {"manifest": file_id(a.manifest), "perception_cache": file_id(a.perception_cache),
                           "cable_templates": file_id(a.cable_templates)},
               "reserved_final_opened": False}
    (a.output_dir / "dataset_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
