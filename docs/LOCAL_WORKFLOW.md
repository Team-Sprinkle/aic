# Local testing and rootless operation

Updated 2026-09-18. Dedicated rootless containers ran official policy trials,
Gazebo control probes and Isaac camera/terminal probes. The earlier
[live validation](experiments/2026-09-17-live-validation.md) did not insert;
the later [verified-data ACT experiment](experiments/2026-09-17-act-verified-8h.md)
has learned insertions but has not established reliability. Examples using
other scenes or settings remain separate evaluations. The active September 18
ACT run uses **physical GPUs 0–1**. The user separately allowed GPUs 2–4 for
the proposed world-model experiment after confirming its plan, for a maximum
of five GPUs across both efforts. That world-model training is still pending
confirmation. The completed earlier ACT experiment used GPUs 0–3 and released
them; check current ownership before reusing a container.

## 1. Identify the workspace and runtime

Run from this repository on the **host**, in Bash:

```bash
export AIC_REPO="$(git rev-parse --show-toplevel)"
export PATH="$HOME/bin:$HOME/.pixi/bin:$PATH"
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
cd "$AIC_REPO"
git status --short
docker info --format '{{json .SecurityOptions}}'
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
nvidia-smi
```

`docker info` should include `rootless`. The context name alone is insufficient:
this machine reported context `default` while connecting to a rootless daemon.
If the provisioned user daemon is stopped, use `systemctl --user start docker`.

On this host, the checkout is `/data1/chmin/yj/ws_aic/src/aic`.
`/home/chmin/yj/ws_aic/src/aic` resolves to it, but
`~/code/ws_aic/src/aic` is a **different checkout**. Use the current checkout path
explicitly when mounting containers. Inspect mounts with:

```bash
docker inspect isaac-lab-base --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

The existing `isaac-lab-base` mounts this repository at
`/workspace/isaaclab/aic`. Check current container state before reuse. The
September ACT experiment used `aic_eval_validation_20260917` on physical GPU 0
plus `aic_collect_validation_20260918` on GPU 1 and
`aic_compare_validation_20260918` on GPU 2. The latter containers have separate
ROS domains (117 and 118), Gazebo partitions and bridge networks. Exact image
IDs and mounts are saved in the experiment’s `container_versions.json`. Each
evaluator restarts its named container; never launch competing jobs against
the same container. Inspect mounts, image identity and GPU restrictions first.
These three experiment containers were stopped after final evaluation on
September 18. Inspect current state and GPU availability before restarting one.

### Sudo-free boundary

The local reference is `~/code/ws_aic/ROOTLESS_DOCKER_GUIDE.MD`. Its useful
cluster workaround is **direct rootless Docker**: Distrobox previously failed
on `/sys` mount propagation. Follow the direct-container recipes here on this
host rather than old `--sim-distrobox` examples.

Routine work assumes the rootless daemon, user namespace prerequisites, host
GPU driver, and NVIDIA container integration are already provisioned. The old
guide's `/etc/nvidia-container-runtime/config.toml` change requires an
administrator; setting a container environment variable does not replace host
configuration. See [Docker rootless prerequisites](https://docs.docker.com/engine/security/rootless/)
and [NVIDIA rootless configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode).
If those prerequisites are missing, record the failure and use the CPU checks
below while the host setup is resolved. No host `sudo` is needed for the workflow
on an already provisioned machine.

## 2. Inspect a saved model without starting simulation

Use the existing environment for inspection, avoiding a dependency re-solve:

```bash
.pixi/envs/default/bin/python -c 'import torch; print(torch.__version__)'
.pixi/envs/default/bin/python scripts/evaluate_act_checkpoints_runtime.py --help
export AIC_ACT_RUN="$AIC_REPO/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k"
test -f "$AIC_ACT_RUN/checkpoints/175000/pretrained_model/model.safetensors"
test -f "$AIC_ACT_RUN/checkpoints/175000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors"
test -f "$AIC_ACT_RUN/act_policy_ts_175000_cuda0.pt"
cat "$AIC_ACT_RUN/act_policy_ts_175000_cuda0.json"
```

This particular export records state size 82, action size 6, chunk size 8, and
three cameras. Older nominal experiments used other dimensions. Use the matching
JSON sidecar and pretrained normalizer; a checkpoint filename is not a schema.
Use the separately saved CPU export when testing on CPU: the exporter documents
that TorchScript may specialize tensors to its export device.

### Optional ACT feature stride

`scripts/train_verified_act.py --backbone-output-stride 16` keeps the ResNet18
weights and channel sizes but changes its final-stage stride, producing 16×18
features per camera at the existing 256×288 image size. The default remains
stride 32 (8×9 features). Omitting this option during a warm start inherits the
checkpoint's recorded geometry; explicitly pass `32` to restore the default.

Keep `aic_backbone_config.json` beside `config.json`, model weights, and the
action configuration. Stride is not stored in weight tensors. Python consumers
must use `lerobot_robot_aic.act_backbone.load_act_policy`; the standard
TorchScript exporter reconstructs and records this geometry automatically.
The historical `RunACT` and `RunOurACT` runners reject custom geometry; use
`RunACTTorchScript` for these exports. The stride option does not increase the
number of model parameters, but increases image tokens fourfold. Measure GPU
memory, throughput, and inference latency before choosing a training batch.

If `.pixi` is absent, follow [getting started](getting_started.md) and the package
setup guides. `pixi.lock` is ignored here, so a fresh `pixi install` can resolve
different dependencies. Save the resolved environment with future experiments.

The following existing audit exercises reward formulas on CPU:

```bash
mkdir -p outputs/local_checks
.pixi/envs/default/bin/python \
  aic_utils/aic_isaac/scripts/audit_axial_first_inserted_start_reward.py \
  > outputs/local_checks/axial_reward_audit.json
```

Check the process exit and JSON `failures`. Passing verifies synthetic reward
cases, not reset physics, controller direction, policy learning, or insertion.

For the evaluation/accounting regression suite without simulation or training:

```bash
.pixi/envs/default/bin/python -m pytest -q \
  aic_utils/gazebo_rl/test/test_score_parser.py \
  aic_utils/gazebo_rl/test/test_serl_transfer_validate.py \
  aic_utils/gazebo_rl/test/test_runtime_evaluator.py \
  aic_utils/aic_isaac/test/test_stateful_curriculum_runtime.py \
  aic_utils/aic_isaac/test/test_stateful_insertion_curriculum.py \
  aic_utils/aic_isaac/test/test_isaac_actor_state_schema.py \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

The wrapper tests substitute fake processes and text checkpoints. The actor
schema tests exercise input validation on CPU; they do not load Isaac or update
model weights.

## 3. Test ACT through the Gazebo challenge runtime

Use a dedicated container: the evaluator **restarts the named container for each
checkpoint**. Select an available physical GPU after checking `nvidia-smi`.
This example maps it to `cuda:0` inside the container.

```bash
export AIC_GPU=0
export AIC_EVAL_CONTAINER=aic_eval_reentry
export AIC_EVAL_IMAGE=ghcr.io/intrinsic-dev/aic/aic_eval:latest
docker image inspect "$AIC_EVAL_IMAGE" --format '{{.Id}} {{json .RepoDigests}}'
docker run -d --name "$AIC_EVAL_CONTAINER" \
  --gpus "device=${AIC_GPU}" --network host \
  --entrypoint /bin/bash \
  --mount "type=bind,src=${AIC_REPO},dst=${AIC_REPO}" \
  -w "$AIC_REPO" \
  -e NVIDIA_VISIBLE_DEVICES="$AIC_GPU" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true \
  "$AIC_EVAL_IMAGE" -lc 'exec sleep infinity'
docker exec "$AIC_EVAL_CONTAINER" nvidia-smi
```

Create this container once; on later visits inspect it and use `docker start`
when stopped. Record the image ID/digest because `latest` is mutable. If the
image is absent, pull/build it using the [evaluation image guide](build_eval.md).
The bind mount exposes local Python policy code and models; compiled evaluation
packages still come from the image. Rebuild the image when testing changes to
those packages. Run one Gazebo evaluation at a time on the shared Zenoh network.

Run from the host with the variables established above:

```bash
export AIC_EVAL_SUBDIR="runtime_eval_reentry_$(date -u +%Y%m%d_%H%M%S)"
.pixi/envs/default/bin/python scripts/evaluate_act_checkpoints_runtime.py \
  --run-dir "$AIC_ACT_RUN" \
  --checkpoint-glob 'checkpoints/175000/pretrained_model' \
  --container "$AIC_EVAL_CONTAINER" \
  --docker-host "$DOCKER_HOST" \
  --workspace-host "$AIC_REPO" \
  --workspace-container "$AIC_REPO" \
  --engine-config "$AIC_REPO/aic_engine/config/sample_config.yaml" \
  --policy-module aic_example_policies.ros.RunACTTorchScript \
  --act-torchscript "$AIC_ACT_RUN/act_policy_ts_175000_cuda0.pt" \
  --policy-device cuda:0 \
  --command-mode delta_pose --command-frame gripper/tcp \
  --control-hz 20 --n-action-steps 4 \
  --max-translation-delta 0.02 --max-rotation-delta 0.2 \
  --max-runtime-sec 60 --engine-timeout-sec 300 \
  --eval-subdir "$AIC_EVAL_SUBDIR" --once-existing --record-rollout
```

This selects one checkpoint and the repository's three-trial sample config.
The September live check instead used the recorded `single_sfp.yaml` and a
25-second policy limit; its result does not cover all three sample trials.
The 60-second policy limit applies per insertion call. Four actions are executed
from each eight-action chunk; this is an explicit runtime choice, not necessarily
the training setting. Commands and clips above are part of the baseline record.

Results go to `$AIC_ACT_RUN/$AIC_EVAL_SUBDIR/175000/`. The top-level
`eval_summary.json` points to the latest attempt. Each `attempt_0001/`,
`attempt_0002/`, etc. keeps its own summary, logs, scores, and trial recordings.
Read the actual score path from the summary:

```bash
cat "$AIC_ACT_RUN/$AIC_EVAL_SUBDIR/175000/eval_summary.json"
python3 - "$AIC_ACT_RUN/$AIC_EVAL_SUBDIR/175000/eval_summary.json" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
score_path = summary.get("scoring_yaml")
print(Path(score_path).read_text() if score_path else "no_score")
PY
```

Require policy readiness, completed trials, and actual score files. Read each
trial's insertion result using the [scoring guide](scoring.md). Missing scores
mean `no_score`/runtime failure. Tier 1 is model validity. The repaired bridge
parser uses official Tier 3 score **75** for correct insertion, reports each
trial separately, and sets overall `insertion_success` only when every observed
trial succeeds. Check counts against the engine configuration as well.

With `--once-existing`, exit 0 means all selected evaluations are complete;
exit 1 means a runtime evaluation failed or was incomplete; exit 2 means invalid
inputs or no matching checkpoints. Completion is a valid measurement even when
insertion fails. An explicit `--command-mode` is required; `none` is labeled
`interface_smoke`. Repeating the command retries failed/legacy summaries and
preserves earlier attempts. A complete result is reused only when settings,
engine config, model file identity, and recorded runtime-source hashes match.
Use a fresh `--eval-subdir` to repeat a measurement or after changing the
container image; the image identity is not part of the reuse signature.

For controlled ACT comparisons, distinguish command pacing from episode length:
`--control-clock simulation` paces commands against camera timestamps, while
`--max-runtime-sec` is always a wall watchdog. The optional fixed-duration
protocol is:

```text
--control-clock simulation --max-simulation-sec 90 --max-runtime-sec 180
```

`--max-simulation-sec` is supported by `RunACTTorchScript` and measures from
the first camera observation in each task. Keep the engine task limit and
whole-batch wall timeout large enough for all trials and reset overhead. The
policy logs unique start/stop records; the evaluator matches them to engine
trial/task identities and records `simulation_duration_audit`. A wall watchdog
that cuts the requested simulated duration short makes the evaluation
incomplete, even if score files exist. Omitting the option retains legacy
wall-limited behavior. The duration code passed 80 related CPU checks; its
first nine-scene live comparisons are in the active September 18 record.

For later ACT comparisons, use a **fresh simulator per original single-trial
config** and require `aic_initial_state_v1` before counting the trial. The
first recorded observation is captured before the first command. Match its
named arm joints to `robot.home_joint_positions` with the fixed **0.05 rad**
maximum absolute error; require finite named gripper values without assuming a
target width. Keep the 90-simulation-second limit, 180-second wall watchdog and
trial/task identity checks. `outputs/experiments/2026-09-18_act_all_verified_6h50/initial_state_contract.json`
pins the threshold, calibration, recorder source, and auditor. The earlier
nine-trial batch exposed two displaced physical starts despite readiness logs.
The repository's engine source has a checked readiness repair, but the active
evaluation image still contains the old engine binary; fresh simulators plus
the recorded state audit are the verified operational procedure today.

For historical SERL checkpoints, the same evaluator supports
`aic_example_policies.ros.RunACTAdapterSERL` with a `.pt` checkpoint glob and ACT
sidecar. The separate [transfer validator](../aic_utils/gazebo_rl/scripts/serl_transfer_validate.py)
supports `--sim-docker-container`, `--docker-host`, and `--workspace-container`.
Use checkpoint-specific settings. The transfer validator now classifies success
from per-trial insertion results; its legacy `--success-threshold` option no
longer controls that label. Docker bridge cleanup now runs inside its dedicated
container. Local/Distrobox cleanup still uses process-name matching, so avoid
concurrent local evaluation sessions.

### Store large runs and evaluations on NVMe

Keep `--workspace-host` and `--workspace-container` pointing to the code
checkout. For artifacts elsewhere, supply **both** `--artifact-host-root` and
`--artifact-container-root`. These describe an existing bind mount; they do not
create one. Paths outside the declared workspace/artifact roots fail preflight.
The two September 18 evaluation containers already mount
`/var/tmp/chmin_aic_20260918_act` at that same container path.

For example, after training and exporting a run under that directory:

```bash
export AIC_ARTIFACT_ROOT=/var/tmp/chmin_aic_20260918_act
export AIC_ACT_RUN="$AIC_ARTIFACT_ROOT/act_all_mixed_stage2"
.pixi/envs/default/bin/python scripts/evaluate_act_checkpoints_runtime.py \
  --run-dir "$AIC_ACT_RUN" --checkpoint-glob 'checkpoints/002000/pretrained_model' \
  --act-torchscript "$AIC_ACT_RUN/act_step002000_cuda0.pt" \
  --container "$AIC_EVAL_CONTAINER" --docker-host "$DOCKER_HOST" \
  --workspace-host "$AIC_REPO" --workspace-container "$AIC_REPO" \
  --artifact-host-root "$AIC_ARTIFACT_ROOT" --artifact-container-root "$AIC_ARTIFACT_ROOT" \
  --engine-config "$AIC_REPO/outputs/experiments/2026-09-18_act_all_verified_6h50/evaluation_configs/development/development_001_sfp_to_nic_nic1_sc0.yaml" \
  --policy-module aic_example_policies.ros.RunACTTorchScript --policy-device cuda:0 \
  --command-mode absolute_pose --command-frame base_link --control-clock simulation \
  --control-hz 20 --n-action-steps 4 --image-channel-order rgb \
  --translation-deadband 0 --rotation-deadband 0 --translation-limit-mode norm \
  --max-translation-delta 0.1 --max-runtime-sec 90 --engine-timeout-sec 220 \
  --eval-subdir eval_dev_nic1_sc0 --record-rollout --once-existing
```

Checkpoints, the normalizer, engine configs, and evaluation outputs can live in
either declared mount. Engine configs accept host or container paths. The
evaluator maps the TorchScript normalizer explicitly through
`AIC_ACT_NORMALIZER_PATH`, so different host/container artifact paths also work
without editing export metadata. Python imports and runtime-source hashes still
come from the workspace. Select an existing checkpoint/export and a fresh
evaluation subdirectory; do not move files belonging to an active run.

### Record and inspect a rollout

`--record-rollout` writes three camera JPEGs approximately once per simulation
second to each attempt's `rollout/task_<timestamp>/`, with image timestamps and
TCP positions in `frames.jsonl`. Capture happens when a policy requests an
observation; TF-only expert policies do not use this callback. Images are raw
policy observations. Recording adds some I/O, so keep the flag consistent in
comparative evaluations. Use the Pixi interpreter: host `python3` lacks PyYAML.

Render a particular trial on CPU, replacing the task path with its actual name:

```bash
.pixi/envs/default/bin/python scripts/render_rollout_snapshots.py \
  path/to/attempt_0001/rollout/task_TIMESTAMP/frames.jsonl \
  outputs/local_checks/policy_review.mp4 --title 'Policy evaluation'
```

This creates an H.264 MP4, first/middle/last contact sheet, and timing metadata.
Each snapshot is held to the next recorded simulation timestamp; the last is
held for one second. It does not invent intermediate frames or rerun a model.
The [September gallery](../outputs/experiments/2026-09-17_live_validation/review/index.html)
is a completed example.

For a privileged control diagnostic with measured signed motion and dense
camera recording, use a fresh output directory and the dedicated container:

```bash
PYTHONPATH=aic_utils/gazebo_rl .pixi/envs/default/bin/python \
  aic_utils/gazebo_rl/scripts/validate_control_rollout.py \
  --container "$AIC_EVAL_CONTAINER" \
  --engine-config "$AIC_REPO/outputs/experiments/2026-09-17_live_validation/single_sfp.yaml" \
  --output-dir "outputs/experiments/$(date -u +%Y%m%d_%H%M%S)_control_probe"
```

The Docker bridge connects over a Unix socket under `outputs/.ipc/`, mapped
through the repository bind mount. Rootless Docker's host network namespace
did not provide access to host learner localhost on this machine. The bridge
loads current checkout Python modules and can use `.pixi/.../bin/ros2` when the
container has no `pixi` command. Do not change the bind-mount relationship.

## 4. Work with Isaac without losing artifacts

Use a dedicated container with explicitly restricted GPUs. Reuse the existing
base container's inspected mounts, without starting it:

```bash
export AIC_ISAAC_GPU=1
export AIC_ISAAC_CONTAINER=aic_isaac_reentry
docker run -d --name "$AIC_ISAAC_CONTAINER" \
  --gpus "device=${AIC_ISAAC_GPU}" --volumes-from isaac-lab-base \
  -e NVIDIA_VISIBLE_DEVICES="$AIC_ISAAC_GPU" -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  --entrypoint /bin/bash isaac-lab-base:latest -lc 'exec sleep infinity'
docker exec "$AIC_ISAAC_CONTAINER" nvidia-smi
docker exec -it "$AIC_ISAAC_CONTAINER" bash
```

Inside the container, work in `/workspace/isaaclab`; this checkout is `aic/`.
The Isaac runtime and assets are separate dependencies; use the
[Isaac package guide](../aic_utils/aic_isaac/README.md) when rebuilding them.

The September image needed its editable package paths refreshed inside the new
container:

```bash
cd /workspace/isaaclab
/isaac-sim/python.sh -m pip install --no-deps --no-build-isolation \
  -e source/isaaclab -e source/isaaclab_tasks -e source/isaaclab_assets \
  -e source/isaaclab_rl -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
/isaac-sim/python.sh -m pip install --no-deps flatdict==4.0.1 safetensors==0.6.2
```

Host driver 535.104.05 failed the RTX driver check. The successful camera probes
used the repository's existing **container-local compatibility workaround**,
inside the dedicated container:

```bash
AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 \
  bash aic/aic_utils/aic_isaac/aic_isaaclab/scripts/patch_isaac_rtx_driver_check.sh
```

This bypasses a version guard and saves a backup; it does not upgrade the host
driver. Record its use. The exact zero-action/camera and terminal-count recipes
are saved in the [live validation evidence](experiments/2026-09-17-live-validation.md#isaac-reset-and-terminal-checks).
For one-second images at 20 Hz use `--image_log_every 20 --save_step_images
--save_videos --video_fps 1`; the recorded simulation rate determines cadence.
These diagnostics found reset drift, so they are not validated curriculum starts.

To materialize the latest curriculum without running training, inside that
container:

```bash
cd /workspace/isaaclab
export AIC_STATEFUL_RUN_ROOT="/workspace/isaaclab/aic/outputs/experiments/$(date -u +%Y%m%d_%H%M%S)_axial_config_check"
AIC_STATEFUL_DRY_RUN=1 bash aic/tools/train_axial_first_inserted_start.sh
```

This writes configs/events and needs the generated base episodes referenced by
`configs/axial_first_inserted_start_curriculum.yaml`; they are ignored artifacts.
It does not validate checkpoint loading or simulator behavior. Keep the dry-run
flag for this check and choose a fresh run root for each invocation. Without the
flag the launcher starts training; do not use it for a saved-model evaluation.

The repaired stateful wrapper counts completed terminal episodes, enforces
configured promotion/demotion thresholds, and requires all fixed evaluation
configurations to finish. With vector environments the final step can exceed
the requested count by up to `num_envs - 1`; the actual count is recorded. A
wall-time limit can end a training segment before its episode budget is reached.
Incomplete evaluation, failed processes, and legacy sampled logs stop progression.
`AIC_STATEFUL_NO_PROGRESS_ASSESS_SECONDS` defaults to 7200: the wrapper checks
after each cycle and exits 3 if there has been no promotion. Setting it to 0
disables that stop rule.

Current SFP evaluation defaults require 0.5 mm axial/lateral error, 0.03 rad
orientation, and module consistency within 1.0 mm axial / 1.5 mm lateral. Module
consistency cannot be disabled for promotion. Evaluation uses zero updates and
exploration, with guide/guard/action overrides rejected. Modified SFP/NIC
collision geometry requires `AIC_STATEFUL_DIAGNOSTIC_COLLIDERS=1`; record it as
a diagnostic setting. These criteria differ from the June run.

For a saved Isaac run, use its `train_config.json` (`argv`, `args`, checkpoint,
episode config) to reconstruct the evaluation. Set a fresh output directory,
explicit checkpoint and seed, bounded steps, and disable updates/exploration
for evaluation (`--updates 0 --actor_exploration_noise_std 0`). Add
`--terminate_on_target_success` and an explicit `--max_completed_episodes` when
collecting terminal success statistics. Recheck all referenced files and record any changed settings;
old snippets often depend on container `/tmp`. The recovered final June
evaluation config is indexed in the [ledger](EXPERIMENTS.md).

The stateful and axial launchers now default to mounted `aic/outputs/experiments/`
and retain every cycle. Override `AIC_STATEFUL_RUN_ROOT` to choose a location
under `/workspace/isaaclab/aic/outputs/` that the host receives. Container `/tmp`
is a separate filesystem location and is lost when the
container is removed/recreated. Small files can be recovered even while stopped:

```bash
export AIC_OLD_RUN=/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937
mkdir -p outputs/recovered_june_metadata
docker cp "isaac-lab-base:${AIC_OLD_RUN}/events.jsonl" outputs/recovered_june_metadata/
```

Back up the chosen checkpoint, ACT/normalizer dependencies, configs, and metrics
together before removing a container. See [artifact storage](../outputs_README.md).
Stop dedicated containers after evaluation with `docker stop <container-name>`;
their writable layers and mounted artifacts remain available.

## 5. Audit demonstrations before training

This reads stored frames and source selection metadata on CPU, without changing
the dataset or running a model. Choose a fresh output directory:

```bash
.pixi/envs/default/bin/python \
  aic_utils/lerobot_robot_aic/scripts/audit_insertion_demonstrations.py \
  --dataset-root outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_h264 \
  --manifest outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_raw32/manifests/accepted.csv \
  --output-dir "outputs/experiments/$(date -u +%Y%m%d_%H%M%S)_dataset_audit"
```

Inspect `summary.json`, `episode_audit.csv`, and `task_balance.csv`. The current
audit expects the six-coordinate Cartesian action / named TCP-velocity schema;
its reference limits are 0.02 m / 0.2 rad. It checks the existing last-5%-episodes
split, without constructing a replacement split. Missing success provenance
stays unknown. See the [measured label and split problems](experiments/2026-09-17-live-validation.md#demonstration-audit)
before reusing this historical dataset for an encoder comparison.

## 6. Reproduce the verified-data ACT work

The [eight-hour ACT report](experiments/2026-09-17-act-verified-8h.md) records the
current dataset decisions, parameter comparisons and simulator outcomes. Its
artifact root is `outputs/experiments/2026-09-17_act_verified_8h/`.

1. **Choose episodes using provenance.** Read [DATASETS.md](DATASETS.md). Full
   official insertion scores and trustworthy image/action lineage are separate
   requirements. `verification/episodes.json` audits the historical inventory;
   each new collection has `verification/audit.json` with scores and visual
   reviews. The newest combined cache is `cache_cheat130_aligned87`.
2. **Keep the split fixed.** `cache.json` records source episodes and train/val
   assignments. Current ACT runs use 70 recent training episodes and 17 recent
   held-out episodes. Historical cache entries remain available but are not
   sampled by these continuations. Their overall validation list also retains
   seven legacy NIC-1 episodes; use `corrective_holdout` in `validation.jsonl`
   for errors on the 17 recent held-out episodes alone. Do not train on
   development/final configs.
3. **Inspect the exact training recipe.** Each `act_*/training_config.json`
   includes arguments, sampled episodes, normalization episodes and warm-start
   checkpoint. The root's `act_*_command.json` files store executable argument
   lists. `scripts/train_verified_act.py --help` describes the options. To rerun,
   choose a new output directory, free GPU and time limit; the saved commands
   contain this experiment's **expired absolute deadline** after it ends.
4. **Preserve deployment metadata.** Keep the `pretrained_model` directory,
   preprocessor/normalizer, TorchScript and JSON sidecar together. The recent
   absolute-target models use RGB, X-positive quaternions, a 33D state including
   simulation elapsed time, and full TCP targets in `base_link`. The fresh
   relative-command candidate instead uses its full observation-relative TCP
   commands and `--delta-pose-reference observation`; check the sidecar. The runtime
   reads these conventions from metadata. Earlier BGR/relative-action models
   need their own matching settings. Do not infer compatibility from action
   width alone.
5. **Export on the intended device.** For example, after checking GPU use:

   ```bash
   CUDA_VISIBLE_DEVICES="$AIC_GPU" .pixi/envs/default/bin/python \
     aic_utils/lerobot_robot_aic/scripts/export_act_torchscript.py \
     --act-checkpoint "$AIC_ACT_RUN/checkpoints/002500/pretrained_model" \
     --output "$AIC_ACT_RUN/act_002500_cuda0.pt" --device cuda:0
   ```

   This exports an existing checkpoint; it does not train. Avoid overwriting a
   frozen export. A CUDA trace targets `cuda:0` inside the container even when
   that container maps another physical GPU.
6. **Use the official evaluator above.** For a recent absolute-pose model,
   select `RunACTTorchScript`, `--command-mode absolute_pose --command-frame
   base_link --control-clock simulation --control-hz 20`, RGB, zero deadbands,
   and the execution horizon and translation limits from its saved development
   summary. Use a fresh evaluation subdirectory. `--max-runtime-sec` is a
   **wall-clock bound**; `--record-rollout` saves ordinary observation snapshots
   about once per simulation second. Keep `--diagnostic-ground-truth` absent.
7. **Read scores and video together.** A complete engine exit and nonempty
   `scoring.yaml` are required. Full insertion is official **Tier 3 = 75**;
   partial insertion is a failure for the reliability target. Render an
   individual `rollout/task_*/frames.jsonl` with
   `scripts/render_rollout_snapshots.py`. The experiment's `review/index.html`
   links selected successes and failures.

Warm-starting on changed normalization should use `--rebase-normalization`
when state/action semantics are unchanged. It adjusts ACT input/output
projections to preserve physical predictions. A warm start resets the optimizer
and schedule; it is not an exact interrupted-training resume.

Incremental image caches reference parent shards by absolute path. Keep those
parents or remap the shard roots when moving a cache. Forty-two completed simulation
bags were archived losslessly; `verification/archived_diagnostic_bags.json`,
`verification/archived_completed_bags.json` and
`verification/archived_pre_final_bags.json` contain checksums and restore
arguments. Scores and training images were not
archived. None of these ignored local artifacts is backed up by a Git commit.

### Test the frozen September ACT model

After selecting a free GPU and preparing the dedicated rootless evaluation
container from section 3, this command reproduces one **known** final-test scene:

```bash
export AIC_ACT_BUNDLE="$AIC_REPO/outputs/experiments/2026-09-17_act_verified_8h/selected_act_final"
export AIC_EVAL_SUBDIR="reproduce_$(date -u +%Y%m%d_%H%M%S)"
.pixi/envs/default/bin/python scripts/evaluate_act_checkpoints_runtime.py \
  --run-dir "$AIC_ACT_BUNDLE" \
  --checkpoint-glob 'checkpoints/002500/pretrained_model' \
  --act-torchscript "$AIC_ACT_BUNDLE/act_selected_cuda0.pt" \
  --container "$AIC_EVAL_CONTAINER" --docker-host "$DOCKER_HOST" \
  --workspace-host "$AIC_REPO" --workspace-container "$AIC_REPO" \
  --engine-config "$AIC_ACT_BUNDLE/evaluation_configs/trial_000001.yaml" \
  --policy-module aic_example_policies.ros.RunACTTorchScript \
  --policy-device cuda:0 --command-mode absolute_pose --command-frame base_link \
  --control-clock simulation --control-hz 20 --n-action-steps 4 \
  --image-channel-order rgb --translation-deadband 0 --rotation-deadband 0 \
  --translation-limit-mode norm --max-translation-delta 0.1 \
  --max-rotation-delta 0.2 --max-runtime-sec 90 --engine-timeout-sec 220 \
  --evaluation-purpose development --eval-subdir "$AIC_EVAL_SUBDIR" \
  --record-rollout --once-existing
```

Read `final_results.md` in the bundle before choosing it as a baseline. The
checkpoint is selected for evidence preservation; its existence does not imply
reliable insertion. Use new scene seeds for future reliability claims. The
original final results remain under `eval_final_trial_*`; reproduction gets a
new subdirectory. Keep the original frozen files unchanged.

To verify and summarize the original 20-scene assessment without simulation:

```bash
.pixi/envs/default/bin/python scripts/summarize_frozen_act_evaluation.py \
  "$AIC_ACT_BUNDLE"
```

This checks the frozen file hashes and every declared trial. Missing or incomplete
trials remain in the denominator and cannot count as success. The frozen queue
manifests have an expired absolute deadline after this experiment; do not reuse
those queues for new runs. For independent parallel simulators, preserve both
network isolation and distinct ROS domains/Gazebo partitions, as recorded in
`container_versions.json` and the container isolation scripts.
