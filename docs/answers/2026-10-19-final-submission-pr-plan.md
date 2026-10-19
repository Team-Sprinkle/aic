# Plan for a focused final-submission PR to main

## Recommendation

Do not open a PR from `feat/hybrid-train` directly. At source commit `0d6f6a8`, it differs from current `origin/main` in approximately 2,556 files. Keep that branch as the complete research record. Create a new branch from the latest `origin/main`, then copy or extract only the final runtime and its supporting assets. The PR must not merge the experiment branch's history.

This is a packaging and behavior-preservation task, not another model experiment. No new training is required. No PR has been opened by this audit.

## Important dependency finding

The selected entry point is `RunVisualConnectorServoQuietContextV56`. A static scan of its explicit local Python imports reaches 129 files and approximately 13,671 lines, including policy ancestors, geometry helpers and training modules. This is a conservative dependency inventory, not a claim that every imported method or model is exercised by the final configuration. Dynamic imports, package resources, checkpoint/config dependencies and container wiring still need a separate audit.

Examples:

- `RunSharedPortTCP` imports `SharedPortTCPActor` from `train_shared_port_tcp_bc.py` and initializes a checkpoint-backed actor. Final actions may bypass the actor while inherited setup still depends on it.
- `SCNativeGraspObservation` imports its network definition from `train_sc_native_keypoints.py`.
- The released evaluation runner loads a frozen bundle, checkpoint/config locations and launch settings from local archives. Copying that runner alone would not make a portable submission.

Therefore, selecting only V56 and deleting its old versions or training files is unsafe. Conversely, merging all 129 files without inspecting inactive paths would preserve much of the experimentation the user wants excluded.

## What belongs in the PR

| Include | Reason |
| --- | --- |
| Final SFP and SC rule transport | The routes used in the demonstrated controller |
| Requested-target coarse perception and history | Port identity, acquisition and crop proposal |
| Native-resolution refinement and calibrated geometry | Port pose fitting and acceptance gates |
| Held-connector pose models and close-up consensus | Correct insertion-axis estimate and stable updates |
| Explicit insertion controller and local recovery | The action-producing path used in final videos |
| Required observation/action adapters | Preserve frame, timestamp and API contracts |
| Pinned selected configuration and artifact manifest | Reproduce the selected behavior and identify exact weights |
| Portable container entry point and dependency declarations | Run through the official policy interface without the original server |
| Focused validation and reproduction commands | Check installation, dependencies and behavior |
| Project writeup, concise experiment summary, results and curated media | Explain the outcome and retain evidence |
| One future-work document | Brief SERL and hierarchical recovery proposals |

The artifact manifest should specify role, expected schema, checksum and externally supplied location for every required model/calibration/config artifact. Do not commit bulk weights by default. Define an explicit artifact acquisition or mounting process rather than assuming server archive paths exist.

## What stays on the experiment branch

- ACT, diffusion, Dreamer/world-model and unused learned insertion implementations added by this branch.
- Training sweeps, failed-policy variants and superseded route/controller versions once their active behavior has been extracted.
- Isaac-only diagnostic/training environments, data generation, teacher collection and oracle tools that the final runtime does not need.
- Raw recordings, datasets, bags, checkpoints, caches and routine rollout videos.
- Long historical prompts/answers, obsolete plans, backup files and archived status documents.
- Server-specific workflow files, usernames, mount locations, device selections and local Docker configuration.

Keep concise failed-approach conclusions in EXPERIMENTS.md; they explain the project's decisions without importing the failed implementations. Preserve links to the experiment branch using immutable source commits where detailed historical records are useful. Repair links in the selected documentation so excluded files do not become dangling references.

Existing unrelated files already on `main` should not be deleted as part of this PR. A separate cleanup can address them if requested.

## Ordered implementation and gates

1. **Inventory the frozen selected runtime.** Explicit-import audit completed. Next inspect the selected frozen bundle/environment, checkpoint schemas, configuration and dynamic imports. Record exact artifact hashes and public-input boundaries. Do not change running jobs.
2. **Create an isolated release worktree from current main.** Keep the experimental checkout and all local workflow files intact. Use a new branch such as `release/final-visual-controller`; copy selected files rather than merge the experimental branch.
3. **Extract a stable runtime package.** Move model definitions into inference modules without training/data-loader imports. Consolidate the final policy's active inherited behavior into named transport, perception, insertion and recovery modules. Remove unused BC actor initialization only after confirming no required preprocessing, state or geometry buffers depend on it. Preserve checkpoint tensor names, transforms, thresholds, timing and selected defaults.
4. **Make submission packaging portable.** Add a clear final policy entry point, pinned configuration and artifact manifest. Replace host paths with explicit configuration. Build/install dependencies and include required package resources. Keep local evaluation/oracle diagnostics outside the deployable package.
5. **Validate equivalence and packaging.** Check fresh-worktree imports, checkpoint loading, model outputs on fixed recorded inputs, frame conversions, command continuity, history and recovery decisions. Confirm no simulator target transforms or contact identities enter control. An isolated short runtime smoke should precede a full three-trial evaluation of the extracted version. Report any score differences as packaging regressions to investigate; the old score is not automatic validation of refactored code. No retraining is needed for extraction.
6. **Add curated documentation/media.** Include the current illustrated writeup, concise EXPERIMENTS/STATUS, generic workflow, one future-work page, selected evaluation videos/GIFs and provenance. Keep technical notes directly supporting the final method; link other history to the research branch. Do not copy all of docs/answers or docs/experiments.
7. **Open a reviewable PR to main.** Use separate commits for runtime extraction, portable packaging/validation, and documentation/media. Review the complete changed-file list, file sizes and host-path exposure. The PR should state external artifact requirements, validation performed and known remaining failures. Open as draft until the clean-branch runtime gate passes; do not merge automatically.

## Stopping rule

Do not publish a supposedly ready final-submission PR that only works because the research checkout or private archive directory is mounted. If an artifact cannot be located or the extracted implementation does not reproduce the frozen runtime, keep the PR in draft and report the exact missing requirement. Keep all original code and assets on the experiment branch so extraction remains reversible.
