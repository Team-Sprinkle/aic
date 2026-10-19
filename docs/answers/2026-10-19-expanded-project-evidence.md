# Expanded project writeup and evidence

The workflow privacy changes were pushed first; the requested project expansion had remained unfinished. The existing project page is now expanded using all eight experiment themes from EXPERIMENTS.md, condensed into a single illustrated account. It covers dataset defects, ACT/world-model limitations, transformer/diffusion transport experiments, rule routes, geometric orientation, requested-port history, held-connector pose, explicit insertion, recovery and released-config repair.

## Added evidence and scope

- **Perception:** a new six-panel Gazebo montage shows saved SFP and SC model predictions across center/left/right cameras, with predicted landmarks and projected reference axes. It uses original-image crops centered on predictions. No ground-truth points are drawn. The displayed predictions visually align with the requested fixtures, but the montage alone is not an accuracy measurement.
- **Historical perception accuracy:** 0.278 mm median / 0.670 mm p95 lateral error and 94.1% correction-sign accuracy describe 173 decisions from six held-out Isaac development episodes. They are not claimed as current Gazebo accuracy across all settings. Source: [perception record](../experiments/2026-09-20-perception-supervised-rl.md).
- **Orientation:** the 1.37° to 0.017° board-yaw repair is one difficult control plus reused checks, not a population-wide error reduction. The final port estimator uses five native-image landmarks; the useful board-rail edge experiment and rejected dense held-plug edge fitting are distinct.
- **Connector pose:** the close-up update comparison changes SC error from 2.27° to 0.69° and depth from 9.30 to 15.52 mm on one matched development start. Source: [SC controller record](../experiments/2026-10-18-sc-explicit-controller-continuation.md).
- **Recovery:** the bright Gazebo clip shows one local recovery with 18.41 mm measured net TCP retreat and a 3.16 mm retry, then an insertion award. The motion labels come from trace events. No RL actor or hierarchical route reset is involved. The source has one brief camera gap; held frames are disclosed in the media README.
- **Recovery comparison:** 1/8 versus 3/8 local seats is an earlier eight-start Isaac recording screen using the same RL actor with recovery disabled/enabled. It is not a matched Gazebo result or proof of hierarchical recovery. Source: [historical screen](../experiments/2026-09-22-serl-mixture-recovery.md).
- **Final scores:** best exposed qualification replay 276.34/300; the frozen batch has 53/57 trial insertions and 15/19 complete passes. Current sample 287.90/300 and earlier selected sample 290.19/300 are separate configurations. Reserved final scenes remain sealed.

## Files and validation

[Project writeup](../robotic_cable_insertion_project.md) · [Experiments](../EXPERIMENTS.md) · [Recovery video and provenance](../media/recovery_demo/README.md) · [Prediction montage](../images/cable_insertion/gazebo_predicted_port_landmarks.png)

Only existing trace/frame artifacts were processed; no simulation, data collection or training was launched. The exports were checked for readable H.264/yuv420p playback, expected duration, prediction-overlay placement, recovery phase views and repository-relative links. The export script and manifest preserve reproducibility without publishing a server-specific source path.
