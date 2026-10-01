# Rootless access diagnosis and SC cleanup continuation

I read [LOCAL_WORKFLOW.md](../LOCAL_WORKFLOW.md) and `/home/chmin/code/ws_aic/ROOTLESS_DOCKER_GUIDE.MD`. Their supported path is the **existing rootless Docker daemon**, without sudo. This session has the expected `DOCKER_HOST=unix:///run/user/1008/docker.sock`; the socket exists and is owned by `chmin`, the same user running this session. A direct Unix-socket connection nevertheless returns **`EPERM: Operation not permitted`**, Docker reports permission denied, and `/dev/nvidia*` is absent. Changing Docker context, socket mode, or restarting the daemon will not repair that combination. The current Codex `workspace-write` sandbox is the access boundary. [Official OpenAI Docs](https://learn.chatgpt.com/codex/sandboxing) describe selecting a different permissions profile through `/permissions` in Codex CLI or the permissions control in the desktop app. The environment may need to be reopened as a host-access local session if its managed profile cannot expose Docker and GPUs.

I added this diagnosis to `docs/LOCAL_WORKFLOW.md`. I also tightened the cleaned BC filter so a local anchor is never dropped merely because its first target repeats: **all four recorded future targets** must remain unchanged. On existing materialized SC data, the filter still finds both planned hold plateaus (282 repeated transport anchors total) and removes zero local samples that show motion/contact. All 104 original scored episodes remain unchanged; the `sc_llb_cleaned_v3` specification is still metadata-only.

**The Docker/GPU boundary cannot be changed from inside this session.** Bag extraction, the actual cleaned action dataset, BC training, autonomous rollouts and later RL remain pending. Once a host-access session passes `docker info` and `nvidia-smi`, resume with:

```bash
cd /data1/chmin/yj/ws_aic/src/aic
.pixi/envs/default/bin/python scripts/run_sc_llb_cleaned_materialization.py \
  --spec outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized
```

Then audit the extracted action chunks and continue the [supervised continuation](../experiments/2026-10-01-sc-llb-cleaned-bc.md), preserving the 21 development validation episodes and keeping reserved final scenes sealed.
