# Enable host access and resume AIC work

1. In this Codex session, enter **`/permissions`** and select **Full access**. If you are using the desktop app, use the permissions control below the message box and select **Full access** there. Change this session only; a global default is unnecessary. [Official OpenAI instructions](https://learn.chatgpt.com/codex/sandboxing).
2. Send **“Continue the SC cleaned-dataset and BC plan.”** I will first run `docker info` and `nvidia-smi`, inspect the other recording session's active containers/GPU use, and avoid its containers, ROS domains, GPU allocations, and output paths. I will use rootless Docker without sudo.
3. If Full access is unavailable or those checks still show `EPERM`/no GPU devices, run Codex from a host-access local environment at `/data1/chmin/yj/ws_aic/src/aic` and give it the same continuation request. A permission profile within this restricted session cannot be repaired by changing `DOCKER_HOST`, Docker context, or socket mode.

The original 104 scored episodes are preserved. The prepared resume command is in [the cleanup experiment record](../experiments/2026-10-01-sc-llb-cleaned-bc.md). The first executable stage is scored-bag extraction and validation; BC starts only after the derived action dataset passes its checks. RL and reserved final scenes stay gated.
