#!/usr/bin/env bash
# Rootless Docker; GPU 1 only. Exact image digest is recorded in the official audit.
python3 /audit/make_targeted_failure_suite.py
/entrypoint.sh ground_truth:=true start_aic_engine:=true headless:=true \
  aic_engine_config_file:=/audit/eval_config.yaml
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
# For each bag, run analyze_sc_trial.py with port index 0 or 1, then:
python3 /audit/summarize_targeted_failure.py
