# Plan for high-quality coarse camera-check data

## Goal

Train the coarse camera check to answer three different questions about the **requested** port:

1. **Which opening is it, and where is it in each image?**
2. **Can the current images actually identify that opening?** It may answer “uncertain.”
3. **Is this a useful, safe place to pause for precise inspection?** This is a transport handoff proposal, not permission to insert.

Keeping these labels separate matters. A far but clearly visible port is a positive location example and a negative pause example. A neighboring visible port is a positive example *when that neighbor is requested*, but a hard negative for the current request. A mathematically projected yet plug-covered opening is not a clear-visibility positive.

## Phase 1: Reuse and audit existing recordings

Start with the original 1152×1024 center/left/right images, synchronized measured TCP and task configuration from the existing **66 SFP training reset groups** (1,739 image triplets), ten grouped development episodes, the already inspected ten harder scenes, recent fully inserting teacher routes, and saved autonomous failures. Keep the existing development and inspected scenes out of fitting unless they are explicitly retired from evaluation; create fresh nonfinal development starts for any final comparison. Keep the four reserved final IK configurations sealed.

One triplet can be paired with each of the **ten possible SFP card/port requests**. That gives up to **17,390 request–triplet examples** from the existing 1,739 triplets: 15,788 requests for installed ports and 1,602 for physically absent slots. These are useful *training labels*, not 17,390 independent scenes. The same-card other port, adjacent card, sparse-card gap and absent slot should be deliberately represented, not left to their natural frequency. SC-to-SC needs the same protocol for its two requested ports, especially one-port and occluded examples, after the SFP label pipeline is sound.

Before using any image, verify its camera timestamp, measured TCP sample, reset identity, task code and post-reset boundary. Reject stale or unmatched triplets. Use complete reset groups for splits; never split adjacent frames of one run across train and evaluation. Retain native images for a high-resolution model; the older 256×288 transport cache cannot replace them.

## Phase 2: Generate cheap candidate labels automatically

For every synchronized triplet and every possible request, derive from *training-only simulator state*:

| Field | How to obtain it | Important limitation |
| --- | --- | --- |
| Physical card/port present | Scene configuration | Says nothing about RGB visibility. |
| Requested opening 3D pose and camera projection | Scene geometry, measured TCP and fixed calibration | A projected point may be hidden by cable, plug, gripper or another object. |
| Distance, lateral/axial offset and orientation to requested port | Measured TCP/plug pose and scene geometry | Training/scoring label only; never an actor or runtime crop input. |
| Contact/force and motion progress | Causal measured streams | Must be joined before reset; contact alone does not identify the requested port. |

This stage is feasible with existing label builders and should take hours of engineering and CPU processing, not a new large simulation campaign. Save exact source hashes and a manifest showing each request–triplet's reset group and label provenance.

## Phase 3: Add the labels geometry cannot provide

Review a **stratified set of about 500–1,000 triplets** rather than thousands of near-identical consecutive frames. Sample by card count (one to five), requested card/port, early versus near-port phase, target distance, and the following difficult cases:

- Correct port clearly visible; both same-card ports visible.
- Another card or port visible while the requested one is out of frame.
- Requested port geometrically in frame but partly/fully covered by plug, gripper, cable or card.
- Board end or slot-zero reference clipped, so the physical card number is ambiguous.
- Sparse-card layouts with empty slots, off-route actor views, and force/contact pauses.

Review the **unmarked original RGB first**: for each camera mark `clear`, `partly identifiable`, `not identifiable`, or `uncertain`, and whether the reviewer can name the correct requested slot. After that blind review, reveal the simulator-projected point to reconcile mistakes. Keep uncertainty; do not force an ambiguous frame into positive or negative. Use a second reviewer or spot-check a meaningful subset of hard/uncertain cases, adjudicate disagreements, and record agreement by category. The older auxiliary depth experiment produced invalid depth/gray RGB after motion, so depth or segmentation should only automate visibility after it is verified against these original RGB reviews.

This is the main labor cost. At roughly 20–40 seconds to examine a three-camera triplet, 500–1,000 examples mean approximately **3–11 hours of first-pass review**, plus quality checks and adjudication. That is a planning estimate, not a measured throughput. A 100–200-triplet pilot should measure the real rate and reveal whether the requested slot can be identified at all from these views. If many middle-card views are genuinely ambiguous, pause labeling and test a safe wider observation posture instead.

## Phase 4: Build distinct training targets

For each `(current triplet, distance-sampled history, requested port code)` training example, retain:

1. **Location:** per-camera requested-port keypoint or crop, when physically present and visually identifiable.
2. **Identity/visibility:** per-camera and three-camera `clear / partial / invalid / uncertain`. The model should abstain on invalid or uncertain images instead of guessing a neighbor.
3. **Pause opportunity:** whether the measured plug/TCP is in a *predefined safe inspection region* for this requested port, with acceptable orientation, force/contact and route state. Fit the region from successful teacher approaches and freeze its thresholds before evaluating on new starts. “Close in base coordinates” alone is insufficient.

Use matched same-image request swaps for hard negatives. Balance clear positives against absent, offscreen, occluded, neighboring-port, far-but-visible and wrong-route examples **per episode/request category**, rather than oversampling repeated frames from one run. Do not let the teacher's selected request or route shortcut the identity label. Keep the image-derived crop proposal in the training path; do not select training/evaluation crops using the true projected location at inference.

## Phase 5: Train, test, and collect only if a gap remains

First fit one bounded original-image, request-conditioned full-row locator with short movement-sampled history against the frozen locator. Fit separate visibility/identity and pause heads, or a shared visual backbone with distinct heads; keep their metrics separate. A sensible first cap is **2,000 updates on one GPU**. Select checkpoint and thresholds only on complete development groups.

Before any live connection, freeze the model and run a new nonfinal development set covering **all ten SFP card/port requests** across card counts, sparse gaps and poses. Report by episode and category: wrong accepted ports, correct early acquisitions, abstentions, visibility precision/recall, false and missed pause opportunities, and complete inference p50/p95/p99. The identity gate remains **zero accepted wrong ports and at least 9/10 early correct acquisitions**; p95 complete inference must stay below **300 ms**. A camera-check “yes” would authorize a held precise inspection only, not insertion.

If a category is missing or fails, collect **only that category**. For example, if five-card middle-slot identity fails despite clear images, gather varied five-card positions and viewpoints; if early target visibility is genuinely absent, test a safe view change; if off-route negatives are missing, use actor-generated failed routes. Preserve all episodes and failure labels. A small first tranche of **10–20 targeted new Gazebo resets** is feasible with the existing rootless workflow, but each run is sequential in the current ROS domain. Recent full runs occupied roughly **0.3–0.8 GB each**, so archive bags and original images to NAS as they are produced; `/data1` has only about **8.6 GB free**. If the existing recordings already cover the category, collect none.

## Feasibility and risks

| Part | Feasibility | Main constraint |
| --- | --- | --- |
| Reindex existing images and make all-request counterfactual labels | High | Verify frame/reset joins and avoid false independence from repeated frames. |
| Geometry, presence, projection, distance and force labels | High | Simulator data is for training/scoring only; stale or post-reset joins must be excluded. |
| True RGB identifiability labels | Moderate | Human review or validated rendering masks are needed; middle-card ordinal ambiguity may be intrinsic to some views. |
| Safe pause labels | Moderate | Requires a physically checked standoff/orientation/force window and off-route negatives, not a distance bit alone. |
| One bounded model comparison | High technically | Generalization depends on reset diversity and label quality; no promise that 2,000 updates will pass. |
| New targeted Gazebo scenes | Feasible but slower | Rootless sequential simulation, review time and NAS storage; do not collect routinely. |

The plan is practical because most raw ingredients already exist. Its main uncertainty is **whether the camera images contain enough information to identify the numbered middle slot in difficult views**. That is why the small blind-review pilot comes before another large training job. If the information is present, better labels and balanced hard examples are a plausible repair. If it is absent, the right fix is an observation or transport change, not more labels of the same view.

See [existing coarse training and failures](../experiments/2026-10-04-coarse-shared-history-execution.md), [recent spatial diagnosis](../experiments/2026-10-05-spatial-transport-prerequisite.md), and [local workflow](../LOCAL_WORKFLOW.md).
