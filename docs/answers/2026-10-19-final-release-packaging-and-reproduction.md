# Focused final-runtime PR and reproduction

The final-submission branch starts at main and contains only the final runtime dependency closure, packaging/evaluation support, concise documentation and curated media. It does not merge the experimental branch history.

The dependency audit found 129 statically imported research Python files. Extraction retained selected controller/transport dispatch and inference-only definitions in a dedicated final package, removed unused BC model loading, and kept all weights in a checksum-verified external bundle. Private class layers preserve the selected method-resolution order; their historical names are provenance rather than separately selectable experimental policies. Training loops, dataset classes, sweeps, unused ACT/diffusion/world-model/RL implementations and collection/oracle tools were not copied. Existing main files were left intact.

Two extracted-runtime full qualification replays scored **276.17/300** and **275.91/300**, both **3/3 insertions**. The second used a newly installed locked inference environment. The original selected demonstration scored 276.34/300. All six new trials had no scored card contacts or excessive-force penalty. These are packaging checks on exposed scenes, not unseen evaluation or a new reliability estimate.

Recorded-input numerical comparisons for SFP and SC found zero difference in coarse pixels, refined landmarks and connector rotation. Three CPU frame/consensus tests passed. One early fresh-environment startup failure exposed a missing ROS YAML dependency; the repaired environment/interpreter wiring passed the full subsequent evaluation.

[Reproduction commands](../FINAL_SUBMISSION.md) cover artifact download/hash verification, dependency installation, all-trial execution, expected output paths and container packaging. [Validation](../validation/final_runtime/README.md) retains raw toolkit scores, source manifests and qualifications.

The self-contained image build remains unverified because the rootless Docker filesystem lacks space for CUDA/runtime layers. This is explicitly recorded rather than equating a mounted-runtime test with a completed image build. The PR should remain draft until that build gate is exercised on adequate storage.
