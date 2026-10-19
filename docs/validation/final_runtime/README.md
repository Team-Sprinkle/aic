# Final-runtime extraction validation

Validated October 19, 2026. The release branch is based on `main`; it does not merge the research branch. The final controller has no learned BC action network, diffusion sampler, world model, RL trainer or dependency on the experiment source checkout.

## Full three-trial reproduction

| Test | SFP, 1 card | SFP, 2 cards | SC, 3 cards | Total /300 | Insertions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extracted code, existing numerical environment | 95.1865 | 95.3171 | 85.6648 | **276.1684** | **3/3** |
| Extracted code, freshly installed locked environment | 95.2378 | 95.2843 | 85.3895 | **275.9116** | **3/3** |
| Self-contained model image, separate evaluator/model containers | 95.3217 | 95.1724 | 85.4201 | **275.9141** | **3/3** |
| Earlier selected research-branch recording | 95.2692 | 95.1533 | 85.9158 | **276.3383** | **3/3** |

All three new runs used the exact exposed published qualification YAML, hash `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5`. All nine new trials received 75-point insertion awards, no scored off-limit contacts and no excessive-force penalty. Variations in duration, path efficiency and smoothness explain the modest score differences; deterministic score equality is not claimed.

[First raw toolkit score](released_existing_environment_scoring.yaml) · [Fresh-environment raw score](released_clean_environment_scoring.yaml) · [Fresh summary](released_clean_environment_summary.json) · [Runtime source manifest](released_clean_environment_manifest.json)

The runtime source hashes in the fresh manifest identify the tested implementation. The frame/pose tests and code/artifact packaging did not tune routes, models or thresholds on these new runs. The separate internal final split was not opened. Three successful three-trial runs are packaging regression checks, not a broad reliability study; retain the earlier frozen-batch rate of 15/19 complete passes as separate evidence.

## Numerical and frame checks

On identical recorded SFP and SC triplets/public state/task codes, the extracted coarse locator, native landmark refiner and held-connector orientation predictor matched frozen-code outputs **exactly** in CPU float32. [Parity measurements](perception_parity.json). This checks two fixed inputs, not held-out model accuracy.

`tools/final_submission/check_contracts.py` passed three CPU tests: port/TCP/body-frame round trip under substantial rotations; rejection of a single 5 mm pose outlier; rejection of repeated observation timestamps. Checkpoint loading and full policy initialization also passed with no BC action model initialized.

## Packaging failure caught and repaired

The first fresh-environment launch failed before policy rollout because the environment's Python did not include a ROS-provided YAML dependency. The runner now uses the pinned simulator's Python 3.12 interpreter with the locked numerical environment on its import/library paths; PyYAML is also pinned explicitly. The successful fresh-environment run above followed that repair. This was an environment defect, not a policy failure hidden from the score table.

## Self-contained image gate: passed

Follow [FINAL_SUBMISSION.md](../../FINAL_SUBMISSION.md) for the downloadable artifact bundle, numerical environment and all-trial command. Model files are external release assets with checksums, not committed weights or private archive dependencies. Raw bags and full logs remain external; the compact score/provenance records are versioned here.

The self-contained image was subsequently built successfully after reclaiming unused Docker build cache. A generated colcon setup script was incompatible with shell `nounset`; startup now temporarily disables that option while sourcing ROS setup and restores it afterward. The Dockerfile also pins the micromamba base digest and excludes host bytecode from its build context.

The packaged policy was evaluated in a separate container beside the provided evaluator, sharing one physical GPU. Its only host bind mount was the output directory: **no host source, numerical environment or model directories were mounted into the model container**. All three trials inserted for **275.9141/300**, with no scored contacts or excessive-force penalties. The controller, weights, routes, scene and thresholds were unchanged.

[Build provenance](submission_image_build.json) · [Packaged-image raw score](released_self_contained_scoring.yaml) · [Summary](released_self_contained_summary.json) · [Run manifest](released_self_contained_manifest.json)

Both the mounted-runtime reproduction path and the self-contained model-image path are now verified. No image was uploaded to the competition registry or submitted to its portal; the PR remains a source-code review, not an official new competition result.
