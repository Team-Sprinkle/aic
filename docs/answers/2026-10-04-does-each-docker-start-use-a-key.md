# Does each Docker startup use one key?

**Each new container start usually needs a new session keyring.** It is not one key for every start of the Docker daemon, and ordinary commands such as `docker ps` do not each consume a key. The account's 200-key limit includes all its kernel keys and keyrings, not just Docker containers.

A container's session keyring should eventually disappear after the container exits **and all references to that keyring are released**. Thus 200 is not meant to be a lifetime limit of 200 launches. On this host, many old `_ses.<container-id>` keyrings remain counted even though their containers have exited or been removed. That is why the count is still 200/200 and another container cannot start. Removing one old exited Docker container did not lower it; the cause of those remaining references has not yet been established.

The practical fix is to have the server administrator raise the non-root kernel key quotas, then investigate old keyring references during maintenance. Do not stop the active Isaac job or revoke unidentified keyrings just to free a slot. [Kernel key quota documentation](https://docs.kernel.org/security/keys/core.html), [runc session-keyring behavior](https://github.com/opencontainers/runc/blob/main/libcontainer/setns_init_linux.go).
