# GPU 1 and GPU 2 process audit (2026-09-30)

I inspected the running processes, their rootless Docker containers, output files, and the experiment record. I did not stop or change any job.

## GPU 1

`aic_sc_fix_gpu1` is running `scripts/train_sc_pose_probe.py`. It started **September 24 at 14:32 UTC** and had run for about **6 days 6.5 hours** when checked on September 30 at 21:02 UTC. It is training a diagnostic visual pose estimator from older Gazebo SC connector waypoint data: 14 training episodes and 7 held-out validation episodes. Its inputs are three wrist-camera images, ordinary robot state, and the task/target encoding; privileged port geometry supplies training labels. This is **not** the BC actor, the current 200-episode SC transport collection, or an online control job.

It is related to our earlier question of whether vision can locate the SC port accurately enough for insertion. Eleven of twelve epochs had completed. The best saved checkpoint reports **1.49 mm near-port lateral error at the 95th percentile**, still above the 0.5 mm insertion corridor. The job is genuinely making progress, though each recent epoch took roughly 14 hours. At the observed rate, its last epoch might finish around October 1 morning UTC; this is only an estimate. It uses substantial CPU, about 1 GB of GPU memory, and showed 0% GPU compute in the sampled instant. The best checkpoint is already saved under `/var/tmp/aic_sc_fix_20260924/sc_pose_probe_visual/`.

**Recommendation:** let the last epoch finish if we want a complete diagnostic result. It is not needed to continue collecting SC teacher episodes, and it should not be treated as evidence that the current policy can insert autonomously.

## GPU 2

`aic_sc_native_gpu2` was created **September 24 at 15:11 UTC** under the project's rootless Docker account. It mounts this AIC repository and old AIC experiment files. The [SC native pose ablation record](../experiments/2026-09-24-sc-native-pose-ablation.md) explicitly names this container as the environment used for its spatial/keypoint experiments. That is strong evidence it came from our AIC work, rather than Actor Labs or another project.

The container currently runs an Isaac Sim `kit` streaming service, with **no training process** inside. It has occupied roughly 0.48 GB of GPU memory and showed about 20% GPU compute in one sample. Its streaming ports are not published to the host. The only established TCP connection visible in the container was a loopback connection from the service back to itself; I found no external viewer/client. Because the Unix account is shared, I cannot prove nobody else has used the container, but the recorded provenance and current process/network state do not suggest another user's live job.

**Recommendation:** this looks like an old AIC visualization container that could be retired to free GPU 2. I left it running because the standing instruction is not to interrupt active jobs, and the user asked for an audit rather than a shutdown.

## GPU 0

The previous assessment stands: its older bounded Isaac probe appears stalled, with its expected result missing. I made no change to it in this audit.
