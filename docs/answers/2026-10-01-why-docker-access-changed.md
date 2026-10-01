# Why Docker worked earlier but fails in this session

Earlier in this conversation, the command environment had **full host access**. In the October 1 continuation, the session environment changed to **workspace-write with restricted device and socket access**. That change is visible in the session configuration; it did not come from the AIC repository or the rootless Docker workflow. I cannot tell from inside the session why that permission profile was selected or changed.

I rechecked at 2026-10-01 12:07 UTC: the process is still `uid=1008 (chmin)`, and `/run/user/1008/docker.sock` still exists, but a direct connection returns **`EPERM: Operation not permitted`**. No `/dev/nvidia*` devices are visible. This combination points to the current Codex sandbox. It is not the normal symptom of a busy GPU, a running second Docker client, or a container-name collision.

Your second Codex session may be running episode recording and using GPUs, storage, or container names; those resources should be checked before starting another experiment. It would not normally remove NVIDIA device nodes from this session or change this session's permission profile. It could be a separate session with host access while this one has workspace-write access.

No source episodes were lost. The cleaned 104-episode metadata manifest is ready, while the actual action-chunk dataset remains pending. When this session has host access again, verify `docker info` and `nvidia-smi`, check the other session's active containers/GPU use, and resume the [materialization command](../experiments/2026-10-01-sc-llb-cleaned-bc.md). [Official OpenAI sandbox guidance](https://learn.chatgpt.com/codex/sandboxing) explains the difference between workspace-write and full access.
