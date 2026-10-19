# Final-runtime extraction validation

Validated October 19, 2026. The release branch is based on `main`; it does not merge the research branch. The final controller has no learned BC action network, diffusion sampler, world model, RL trainer or dependency on the experiment source checkout.

## Full three-trial reproduction

| Test | SFP, 1 card | SFP, 2 cards | SC, 3 cards | Total /300 | Insertions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extracted code, existing numerical environment | 95.1865 | 95.3171 | 85.6648 | **276.1684** | **3/3** |
| Extracted code, freshly installed locked environment | 95.2378 | 95.2843 | 85.3895 | **275.9116** | **3/3** |
| Earlier selected research-branch recording | 95.2692 | 95.1533 | 85.9158 | **276.3383** | **3/3** |

Both new runs used the exact exposed published qualification YAML, hash `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5`. All six new trials received 75-point insertion awards, no scored off-limit contacts and no excessive-force penalty. Variations in duration, path efficiency and smoothness explain the modest score differences; deterministic score equality is not claimed.

[First raw toolkit score](released_existing_environment_scoring.yaml) · [Fresh-environment raw score](released_clean_environment_scoring.yaml) · [Fresh summary](released_clean_environment_summary.json) · [Runtime source manifest](released_clean_environment_manifest.json)

The runtime source hashes in the fresh manifest identify the tested implementation. The frame/pose tests and code/artifact packaging did not tune routes, models or thresholds on these new runs. The separate internal final split was not opened. Two successful three-trial runs are packaging regression checks, not a broad reliability study; retain the earlier frozen-batch rate of 15/19 complete passes as separate evidence.

## Numerical and frame checks

On identical recorded SFP and SC triplets/public state/task codes, the extracted coarse locator, native landmark refiner and held-connector orientation predictor matched frozen-code outputs **exactly** in CPU float32. [Parity measurements](perception_parity.json). This checks two fixed inputs, not held-out model accuracy.

`tools/final_submission/check_contracts.py` passed three CPU tests: port/TCP/body-frame round trip under substantial rotations; rejection of a single 5 mm pose outlier; rejection of repeated observation timestamps. Checkpoint loading and full policy initialization also passed with no BC action model initialized.

## Packaging failure caught and repaired

The first fresh-environment launch failed before policy rollout because the environment's Python did not include a ROS-provided YAML dependency. The runner now uses the pinned simulator's Python 3.12 interpreter with the locked numerical environment on its import/library paths; PyYAML is also pinned explicitly. The successful fresh-environment run above followed that repair. This was an environment defect, not a policy failure hidden from the score table.

## Reproduction and remaining packaging gate

Follow [FINAL_SUBMISSION.md](../../FINAL_SUBMISSION.md) for the downloadable artifact bundle, numerical environment and all-trial command. Model files are external release assets with checksums, not committed weights or private archive dependencies. Raw bags and full logs remain external; the compact score/provenance records are versioned here.

The self-contained Dockerfile is supplied, but its full image build was **not validated** in this session: the rootless Docker storage filesystem has insufficient free space for the CUDA/runtime layers. The tested launcher runs the clean code and freshly installed pinned dependencies in the pinned simulator container with explicit read-only mounts. Keep the PR in draft until a build with adequate Docker storage validates the self-contained image as well; the passing mounted-runtime evaluation must not be represented as a completed image-build test.
