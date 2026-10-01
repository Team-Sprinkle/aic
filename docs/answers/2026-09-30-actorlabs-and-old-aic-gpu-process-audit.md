# Actor Labs and older AIC GPU process audit

Snapshot: **2026-09-30 20:58 UTC**. This was a read-only process audit. No container or job was interrupted, following the earlier instruction not to interrupt active jobs.

## Actor Labs hackathon

**Yes: the Actor Labs hackathon is not using a GPU now.** The workspace is `/data1/chmin/yj/ws_actorlab/` (including `actathon/`). No live process had that workspace as its current directory or in its command line, no rootless Docker container was named for it, and none of the GPU process command lines pointed to it. This is a current-process observation; the separate HIGS training processes owned by `donggeunlim` on GPUs 6 and 7 are not under the Actor Labs workspace and should not be treated as our jobs.

## Older AIC GPU processes

| GPU | Process / container | Evidence | Recommendation |
| --- | --- | --- | --- |
| 0 | `probe_target_reward.py` in `aic_isaac_world_policy_20260920` | Started Sep 23 for a bounded **250-step** SC probe. Still consumes about one CPU core and holds 1.35 GB GPU memory after seven days. Its requested `sc_nic5_no_gripper_base_collision250.json` output does **not** exist in the mounted output directory. GPU utilization was 0% in the sampled snapshot. | **Retire as stalled.** Preserve its command and any log before stopping; it is not advancing the current collection. It is a live OS process, not a completed result. |
| 1 | `train_sc_pose_probe.py` in `aic_sc_fix_gpu1` | Started Sep 24. It has actually reached **epoch 11/12** and updated `metrics.json` at 16:44 UTC today. A `best.pt` checkpoint exists. Recent epochs took about **50,000 s (~14 h)** each. CPU use is high while sampled GPU utilization was 0%; best held-out near-lateral p95 is **1.49 mm**, above the 0.5 mm insertion corridor. | **Not dead, but low priority.** The current project direction has parked this probe and the best checkpoint is saved. Recommend stopping after preserving its current metrics/checkpoint unless a complete 12-epoch diagnostic is specifically needed. Letting it finish may take about another half day. |
| 2 | Isaac Sim `kit` streaming process in `aic_sc_native_gpu2` | Started Sep 24, holds ~0.48 GB GPU memory, sampled GPU utilization ~20%. The container is on bridge networking with no published host port. Its only two established TCP sockets are the two ends of a `127.0.0.1:49100` self-connection; no external client was visible. | **Likely idle/obsolete service.** Recommend retiring after confirming nobody is about to use that Isaac streaming session. It is not part of the current Gazebo collection. |
| 7 | `aic_moveit_sc_pilot_llb_*` | Current scored SC-to-SC collection; one Gazebo replay at a time. Shares GPU 7 with another user's unrelated training process. | **Keep** until the 200-setting collection and archive/finalization finish. |

The other AIC containers `aic_transport_dataset_tools_20260930`, `aic_shared_bc_gpu3`, `aic_sc_pose_gpu2`, and `aic_act_all_eval_20260918` showed only sleep shells and/or zombie workers in the sampled process lists, with no GPU process attached. They can be retired later if their interactive shells are no longer needed; their existing files are in mounted workspaces/volumes rather than being identified here as disposable data.

**GPU count:** AIC currently has contexts on GPUs **0, 1, 2, 7**, but only GPU 7 is needed for the active collection. The entire eight-GPU server has processes on **0, 1, 2, 6, 7**. Retiring the three older AIC GPU jobs would leave this project on GPU 7 while the other user's jobs remain on GPUs 6 and 7.
