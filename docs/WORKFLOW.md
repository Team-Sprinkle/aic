# Running the project

The final demonstrations use rule-based transport, requested-port visual perception, and an explicit insertion controller with local recovery. Learned insertion actors remain available for comparisons.

## Rootless operation

Use a provisioned rootless Docker daemon and verify that `docker info` reports rootless operation. Inspect active jobs and available GPU memory before starting a run. Launchers isolate ROS domains, Docker networks and Gazebo partitions; do not bypass their cooperative locks or interrupt another job.

Machine-specific setup, storage locations and commands are maintained in an untracked local workflow. Choose free devices and fresh output directories for each experiment. Archived checkpoints and frozen runtime bundles are required to reproduce the demonstrations; the repository media alone are insufficient.

## Evaluation and artifacts

The released-config runner is `scripts/run_sc_sfp_repaired_released_eval.py`; the public-sample recording runner is `scripts/run_repaired_sample_video.py`. Review their arguments and artifact dependencies before use. Check configuration hashes rather than relying on filenames. See the [evaluation protocol](EVALUATION_PROTOCOL.md) and [completion notes](answers/2026-10-19-final-project-completion-notes.md) for provenance and limitations.

Retain runtime traces, frame manifests, effective configuration, scoring outputs and model identifiers. Keep datasets, checkpoints, raw bags, bulk frames and routine videos outside Git. Only curated project media should be versioned; see [final recordings](media/final_evaluation/README.md). Simulator geometry and contact diagnostics are evaluation tools, not deployable policy inputs.
