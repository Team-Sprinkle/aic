# Force Parity Setup (Gazebo vs Isaac Lab)

Because Isaac Lab runs from a separate checkout (`~/IsaacLab`) and integrates this repo as
`~/IsaacLab/aic`, force-parity scripts must exist in that runtime tree.

This requirement follows the Isaac integration flow in [README.md](./README.md): run Isaac commands
inside the Isaac Lab Docker/container context, from the Isaac Lab checkout.

## Why copying is required

- Development path may be: `~/projects/ws_aic/src/aic`
- Isaac runtime path is: `~/IsaacLab/aic`
- `isaaclab -p ...` resolves scripts from the runtime tree in `IsaacLab/aic`

If files are updated only in `ws_aic`, Isaac runs may still use stale scripts from `IsaacLab/aic`.

## Files to copy to Isaac runtime tree

Copy these from your development repo to your Isaac runtime repo:

- `aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/gazebo_force_logger.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py`
- `scripts/compare_force_parity.sh`
- `scripts/force_parity_config.env`
- `scripts/run_gazebo_force_parity.sh`
- `scripts/run_isaac_force_parity.sh`

## Example copy commands

From your dev repo root (`~/projects/ws_aic/src/aic`):

```bash
cp aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/
cp aic_utils/aic_isaac/aic_isaaclab/scripts/gazebo_force_logger.py \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/
cp aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/
cp scripts/compare_force_parity.sh scripts/force_parity_config.env \
   scripts/run_gazebo_force_parity.sh scripts/run_isaac_force_parity.sh \
  ~/IsaacLab/aic/scripts/
chmod +x ~/IsaacLab/aic/scripts/compare_force_parity.sh \
         ~/IsaacLab/aic/scripts/run_gazebo_force_parity.sh \
         ~/IsaacLab/aic/scripts/run_isaac_force_parity.sh
```

## Runtime requirement (Isaac)

`run_isaac_force_parity.sh` must be executed inside the Isaac Lab Docker container.
If run outside Docker, it exits with an error.

## Running `cheatcode_modified_eval.py`

Run from inside the Isaac Lab container, from the Isaac Lab checkout root
(`cd /workspace/isaaclab`):

```bash
./isaaclab.sh -p -u aic/aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py \
  --enable_cameras \
  --task AIC-Task-v0 \
  --gazebo_config aic/outputs/configs/fixed_1_trials_sfp2nic.yaml \
  --trial_name trial_1 \
  --task_name task_1 \
  --out aic/outputs/force_parity/output.csv
```

Notes:

- `--enable_cameras` is required for this env configuration. It defines tiled camera
  sensors and image-feature observations.
- If `--enable_cameras` is omitted, the app may shut down soon after startup with:
  `RuntimeError: A camera was spawned without the --enable_cameras flag.`
