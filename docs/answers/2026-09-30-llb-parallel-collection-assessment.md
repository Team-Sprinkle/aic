# Can additional GPUs accelerate the SC trajectory collection?

**Decision: keep the current collection on one worker.** I inspected the active collector, its launchers, progress writer, archive step, and GPU use. I did not change or interrupt the running job.

At the check, the 30-case pilot had finished and the continuation had recorded **44/170** attempts (through `llb_0073`), including **25 continuation full insertions**. Recent cases took about **3–5 minutes each**, and the worker was archiving the latest run to NAS. GPU 7 hosted the collector, but the time also includes MoveIt planning, Gazebo simulation, scored-bag checks, and copying roughly large per-case outputs to NAS. The now-free GPUs alone will not make those serial stages faster.

The current implementation is not safe to duplicate on another GPU:

- Both MoveIt launchers use fixed ROS domain 93, and Gazebo replay uses fixed ROS domain 87 with host networking. Concurrent runs could discover or communicate with each other's nodes.
- `run_sc_llb_collection.py` reads and rewrites one `collection_progress.json` without a lock. Two workers could attempt the same case or overwrite each other's results.
- The supervisor expects its single child to complete all 170 continuation cases, and the finalizer waits for that supervisor status. A separate worker would not be counted correctly without revising this lifecycle.
- Runs are copied and verified on NAS before local files are replaced by links. Two workers would increase disk and NAS pressure while local `/data1` has about 93 GB free.

Parallel collection is possible with a deliberate redesign: partition the manifest into disjoint case ranges; assign distinct ROS domains and per-worker output/progress/archives; then merge results only after each scored bag and start audit passes. That is more invasive than the likely speed gain for the remaining roughly 126 cases. The user explicitly preferred the existing speed if parallelism could disturb results, so the safer choice is to let this run finish. The extra GPUs remain available for other work that does not share these simulator domains or files.
