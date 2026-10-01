#!/usr/bin/env bash
# Sequential, frozen, observation-only Gazebo comparison with the recorded-action cap.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
export AIC_TRANSPORT_EVAL_GPU=${AIC_TRANSPORT_EVAL_GPU:-3}
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06
root=outputs/experiments/2026-10-01_sc_llb_cleaned_bc
for arm in warm clean motion_safe elapsed; do
  case "$arm" in
    warm) model=transport_mixed_warm_start ;;
    clean) model=transport_imagenet_clean ;;
    motion_safe) model=transport_imagenet_motion_safe_v2 ;;
    elapsed) model=transport_imagenet_elapsed_v1 ;;
  esac
  for case_name in port0_two port1_three; do
    if [[ "$case_name" == port0_two ]]; then
      scene=port0_two_mixed
    else
      scene=port1_three_mixed
    fi
    audit="$root/autonomous_transport/${arm}_cap60_${case_name}"
    if [[ -s "$audit/launcher.exit" && -s "$audit/results/scoring.yaml" ]]; then
      echo "already finished: $audit"
    else
      [[ ! -e "$audit" ]] || { echo "Partial existing audit: $audit" >&2; exit 1; }
      bash scripts/run_clean_transport_eval.sh "$scene" "$root/$model/best.pt" "$audit"
    fi
    .pixi/envs/default/bin/python scripts/analyze_clean_transport_rollout.py "$audit" \
      --plan "outputs/experiments/2026-09-30_transport_continuation/plans/$scene/moveit_joint_trajectory.json" \
      > "$audit/analysis_stdout.json"
    echo "completed: $audit"
  done
done
