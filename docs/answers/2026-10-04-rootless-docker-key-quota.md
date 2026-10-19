# Why rootless Docker cannot start a container

This is a **Linux kernel keyring quota**, not free disk space and not a GPU limit. The account running rootless Docker is UID 1008 (`chmin`). The readings at the first failed launch were:

```text
kernel.keys.maxkeys  = 200
kernel.keys.maxbytes = 20000
/proc/key-users: 1008: 200 200/200 200/200 13636/20000
```

The third quota pair says this user owns **200 keys out of 200 allowed**; the payload-byte limit has not yet been reached. All 200 keys visible to this account are keyrings, and 195 have names of the form `_ses.<container-id>`. The active Isaac container's session keyring is among them. When `runc` starts another container, it tries to create/join a new session keyring. The kernel refuses with `EDQUOT`, which Docker reports as `unable to create session key: disk quota exceeded`. [Linux kernel key quota documentation](https://docs.kernel.org/security/keys/core.html), [runc report of this exact 200-container error](https://github.com/opencontainers/runc/issues/726), [runc source for joining a container session keyring](https://github.com/opencontainers/runc/blob/main/libcontainer/setns_init_linux.go).

## Safest fix

Ask the server administrator to raise the **non-root** key quotas for the host, for example:

```bash
sudo sysctl -w kernel.keys.maxkeys=1024
sudo sysctl -w kernel.keys.maxbytes=1048576
```

The second setting gives room for the additional keyring descriptions and payloads. These are host-wide limits for non-root users, so the administrator should choose values appropriate for the shared server. To make the change survive reboot, the administrator can place those two assignments in a file under `/etc/sysctl.d/` and apply it using their normal configuration process. The Linux kernel documents these sysctls as administrator-controlled. Raising them should allow a new rootless container without stopping the active Isaac container; after the change, verify `/proc/key-users` and run a tiny test container before another Gazebo rollout.

## Options without sudo

After this answer was written, the kernel count fell to 199/200. I then removed 25 audited stopped September AIC containers; the count stayed 199/200, and one minimal `docker run --rm` now succeeds. Removing stopped-container metadata did **not** lower the observed key count. The single free slot allows sequential launches for now, but it is not a durable repair. The current shell's session keyring is revoked, so `keyctl` cannot reach arbitrary old container session keys by ID. Revoking unknown keys or restarting the rootless daemon could affect the active container and should not be used as routine cleanup.

After all live containers and jobs have completed, a controlled logout/restart of the rootless Docker user session may let the old session keyrings disappear, but this has **not** been tested on this host and should be treated as maintenance, not an immediate remedy. `runc` has a `--no-new-keyring` option, but that makes the container inherit a caller's session keyring; changing Docker's runtime behavior to use it is a security tradeoff and is not the recommended fix. [runc option documentation](https://github.com/opencontainers/runc/blob/main/man/runc-run.8.md).

The next AIC experiment remains ten new non-final SFP resets with frozen models. It can start after a new rootless container passes the tiny startup check; the active Isaac job stays untouched.
