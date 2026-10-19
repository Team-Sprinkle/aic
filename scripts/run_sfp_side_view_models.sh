#!/usr/bin/env bash
# Evaluate unchanged SFP perception checkpoints on the three measured view holds.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
root=/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1
labels="$root/event_labels_v1"
[[ -f "$labels/rows.jsonl" ]] || { echo 'Missing event labels' >&2; exit 2; }
for path in "$root/view_candidates_v1.jsonl" "$root/view_proposals_v1" "$root/view_model_scores_v1"; do
  [[ ! -e "$path" ]] || { echo "Preserve prior output: $path" >&2; exit 2; }
done
image='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock" docker run --rm \
  --gpus device=1 --network none --shm-size 8g --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_CONTAINER_CLI_NO_CGROUPS=true -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo:$repo" -v /home/nas/chmin/aic_archives:/home/nas/chmin/aic_archives \
  -w "$repo" "$image" -lc '
    set -euo pipefail
    export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16"
    export PYTHONPATH=scripts:aic_utils/lerobot_robot_aic
    root=/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1
    .pixi/envs/default/bin/python scripts/audit_sfp_card_row_association.py \
      --labels "$root/event_labels_v1" \
      --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt \
      --output "$root/view_candidates_v1.jsonl"
    .pixi/envs/default/bin/python scripts/audit_sfp_observed_rail_crop.py \
      --candidates "$root/view_candidates_v1.jsonl" \
      --labels "$root/event_labels_v1" \
      --output "$root/view_proposals_v1" --source-reprojection-px 12
    .pixi/envs/default/bin/python scripts/evaluate_sfp_fresh_anchor_models.py \
      --labels "$root/event_labels_v1" --proposals "$root/view_proposals_v1" \
      --output "$root/view_model_scores_v1" --expected-scenes 4
  ' > "$root/view_models_v1.log" 2>&1
