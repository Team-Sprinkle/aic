# Reproduce the final cable-insertion controller

The deployable entry point is **`aic_example_policies.final.FinalInsertionPolicy`**. It uses three wrist RGB cameras, the public requested-task code, public TCP/robot state and wrist force. No true target pose, scoring event, oracle contact identity, BC actor or RL actor chooses its commands.

The sequence is rule transport → requested-port history/crop acquisition → calibrated native-image landmark fit → held-connector pose estimation → visual approach/alignment/insertion, with local measured-path backoff and sideways retry. The exact selected settings and artifact hashes are packaged under `aic_example_policies/aic_example_policies/final/config/`.

## 1. Prerequisites

Use Linux x86_64, rootless Docker with NVIDIA container support, and a compatible NVIDIA driver for CUDA 12.8. The evaluation below was validated on one RTX A6000. Inspect `docker ps` and `nvidia-smi`; select a free GPU and ROS domain. The launcher isolates its Docker network and Gazebo partition and removes only its own container. Its locks coordinate this launcher; they do not reserve devices against unrelated users.

The pinned simulator/interface image is:

```text
ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
```

No host sudo access is required. A provisioned rootless Docker daemon, GPU support and image registry/network access are prerequisites; the scripts do not change host permissions or restart shared services.

## 2. Obtain the model artifacts

Download the model archive and checksum from the [artifact release](https://github.com/Team-Sprinkle/aic/releases/tag/final-visual-controller-artifacts-v1). Unpack into a directory outside Git:

```bash
curl -fLO https://github.com/Team-Sprinkle/aic/releases/download/final-visual-controller-artifacts-v1/final-visual-controller-artifacts-v1.tar.gz
curl -fLO https://github.com/Team-Sprinkle/aic/releases/download/final-visual-controller-artifacts-v1/final-visual-controller-artifacts-v1.tar.gz.sha256
sha256sum -c final-visual-controller-artifacts-v1.tar.gz.sha256
mkdir -p "$HOME/aic-final-artifacts"
tar -xzf final-visual-controller-artifacts-v1.tar.gz -C "$HOME/aic-final-artifacts"
export AIC_ARTIFACT_ROOT="$HOME/aic-final-artifacts"
```

There are **18 files, about 294 MB unpacked**: port backbone/coarse/native-landmark models, plug position/orientation models, near-view components, empty-row perception and certificates/calibration. The runtime verifies every artifact against `config/artifacts.json` before loading it. Use only this trusted bundle: checkpoints use PyTorch pickle-based loading. No training or dataset download is needed.

SC's selected close-up component retains its original experimental-scope approval: it passed the selected end-to-end replay, but its earlier independent pose gate failed. Packaging has not converted that certificate into a general qualification claim.

## 3. Install the exact inference environment

This is separate from the large research Pixi environment. It contains no ACT/diffusion/world-model trainer, LeRobot dataset tooling or Isaac stack. Choose a new prefix on a disk with at least 15 GB free during installation. Use Python with pip >=22.3 to bootstrap the pinned wheels.

For example, obtain micromamba 2.3.3 and verify its archive:

```bash
curl -fL https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 -o micromamba.tar.bz2
printf '%s  %s\n' e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2 micromamba.tar.bz2 | sha256sum -c -
mkdir -p mamba-tool
tar -xjf micromamba.tar.bz2 -C mamba-tool bin/micromamba
bash tools/final_submission/prepare_runtime.sh "$HOME/aic-final-runtime" "$PWD/mamba-tool/bin/micromamba"
```

The conda explicit lock pins the numerical/image libraries, including OpenCV 4.13.0 and its JPEG implementation. The policy uses the pinned simulator's Python 3.12 interpreter so its system ROS dependencies remain available. The wheel lock pins Torch 2.9.0+cu128, torchvision 0.24.0+cu128 and their dependencies. Preserve the installed prefix; moving an environment manually can break native-library configuration.

## 4. Run all three trials and read the score

From the repository root, choose an unused output directory, free GPU and ROS domain. The following example uses GPU 0; change it after checking other jobs:

```bash
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
python3 tools/final_submission/evaluate.py \
  --artifacts "$AIC_ARTIFACT_ROOT" \
  --runtime-env "$HOME/aic-final-runtime" \
  --output "$HOME/aic-final-results/released-001" \
  --gpu 0 --domain 202 --config released
cat "$HOME/aic-final-results/released-001/summary.json"
cat "$HOME/aic-final-results/released-001/results/scoring.yaml"
```

`released` runs the exact published `eval_config.yaml`: SFP with one card, SFP with two cards, SC with three cards. `--config sample` instead runs the public predeadline sample: SFP one card, SFP one card, SC zero cards. Each command runs **all three trials**, not a near-port reset or transport-only test. The engine supplies its normal per-trial randomized starts. The included robot description preserves the toolkit's nominal joint start; the diagnostic world adds recording views without privileged runtime inputs.

| Config | SHA256 |
| --- | --- |
| Published qualification | `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5` |
| Public sample | `21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d` |

The launcher checks these hashes, pins the simulator image, records runtime source hashes, waits for toolkit scoring and retains raw logs/bags. It does not change the evaluation YAML or controller thresholds to obtain a pass. Default wall-clock timeout is 30 minutes; logs remain available on failure.

Outputs include `summary.json`, `manifest.json`, `policy.log`, `engine.log`, `runtime_trace.jsonl`, recorded trial bags and `results/scoring.yaml`. **The toolkit's tier-3 award establishes insertion**, rather than a controller return value. The summary reports each trial total and the number of 75-point insertion awards. Full source traces can contain thousands of events; start with the summary and error lines.

## 5. Container packaging

To bake the same runtime and models into a submission image, use a modern Docker/BuildKit version supporting named build contexts:

```bash
docker build -f docker/final_controller/Dockerfile \
  --build-context artifacts="$AIC_ARTIFACT_ROOT" \
  -t aic-final-controller:v1 .
```

Allow sufficient Docker storage for the CUDA runtime layers. The image starts the official `aic_model` lifecycle/action interface with the final policy selected. The evaluation network supplies `AIC_ROUTER_ADDR`; ACL mode additionally uses `AIC_ENABLE_ACL` and `AIC_MODEL_PASSWD`, following the official interface. Keep credentials in environment/secrets, never commit them. The all-trial runner above validates the clean source and pinned environment in the simulator container; image-build validation status is recorded separately in the validation record.

## Results and limitations

The clean runtime achieved **276.17/300** using the existing numerical environment, then **275.91/300** using a fresh installation of the supplied locks. Both runs received all three insertion awards, close to the earlier selected 276.34/300. The original frozen 19-replay batch had 15/19 complete passes and 53/57 trial insertions. Small score differences are expected from randomized starts and simulator timing; one passing replay is not a reliability estimate.

Published scenes were exposed and used for development with permission. These are local reproduction results, not a new official leaderboard score or unseen-test performance. The separate reserved final split remains sealed. No claim is made that this configuration generalizes to every possible board pose, grasp or cable snag.

[Validation results](validation/final_runtime/README.md) · [Illustrated project](robotic_cable_insertion_project.md) · [Measured limitations and future work](FUTURE_WORK.md)
