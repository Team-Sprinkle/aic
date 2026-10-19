# Focused final-runtime PR

Created **[draft PR #13](https://github.com/Team-Sprinkle/aic/pull/13)** from `release/final-visual-controller` to `main`.

The branch starts from main and changes **94 files in three commits**, rather than merging the approximately 2,556-file research branch:

1. `2dcc2d8` — extract the final visual controller and inference-only perception definitions; remove unused BC action-network initialization.
2. `4a66c5f` — package pinned dependencies, artifact/config checksums, rootless evaluation commands and container support.
3. `680ca65` — add reproduction documentation, raw scores, scoped validation and selected videos/GIFs.

Unused research implementations, data collection, sweeps, obsolete plans and raw datasets remain on the experiment branch. Pre-existing unrelated main files are retained. Private controller layers preserve the selected method dispatch; they are not separately exported alternative policies.

## Evaluation confirmed before PR creation

| Clean runtime test | Total /300 | Insertion awards |
| --- | ---: | ---: |
| Existing numerical environment | **276.17** | **3/3** |
| Newly installed locked environment | **275.91** | **3/3** |
| Earlier selected research recording | **276.34** | **3/3** |

The fresh-environment run scored 95.24 / 95.28 / 85.39 for the SFP-one-card, SFP-two-card and SC-three-card trials. Both new runs used only clean-branch runtime source and explicitly supplied artifacts; no experiment source checkout was mounted. All six trials received toolkit insertion awards without scored off-limit contacts or excessive-force penalties. Reserved final scenes were not opened. These are exposed-scene packaging checks, not a broad reliability estimate.

Two SFP/SC recorded-input comparisons exactly matched the frozen coarse locator, native landmarks and connector orientation in CPU float32. Three frame/consensus tests passed. A fresh-start ROS dependency failure was diagnosed and repaired before the successful full rerun.

## Reproduce

[Complete instructions](https://github.com/Team-Sprinkle/aic/blob/release/final-visual-controller/docs/FINAL_SUBMISSION.md) include dependency installation, hashes, all-trial commands and score paths. The [versioned artifact release](https://github.com/Team-Sprinkle/aic/releases/tag/final-visual-controller-artifacts-v1) contains the 18 required model/approval files (about 294 MB unpacked). GitHub's uploaded archive digest matches the local SHA256.

```bash
python3 tools/final_submission/evaluate.py \
  --artifacts "$AIC_ARTIFACT_ROOT" \
  --runtime-env "$HOME/aic-final-runtime" \
  --output "$HOME/aic-final-results/released-001" \
  --gpu 0 --domain 202 --config released
```

Choose a free GPU and domain. Read `summary.json` and `results/scoring.yaml` under the output directory. Only one GPU is used for the evaluation; no training was launched.

## Remaining gate

The PR remains **draft** because the fully self-contained Docker image build was not validated: the rootless Docker filesystem does not have sufficient space for the CUDA/runtime layers. The passing fresh-environment test mounts the explicit runtime and artifacts in the pinned simulator image; it is not mislabeled as a completed image-build test. The Dockerfile and build command are included for that remaining gate. No merge or auto-merge was performed.
