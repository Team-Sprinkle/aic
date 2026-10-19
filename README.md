> **October 18 score target achieved:** **290.19/300**, all three public-sample insertions pass independent depth/lateral/orientation checks; all used-stage p95 inference is below300ms. The analyze/repair/evaluate loop stops after18 completed full samples. V56 preserves three quiet SCcontext frames while sharing initial grasp observations, and seats1.5s above the public1s requirement. Rule transport and explicit RGB/public-state insertion use unchanged perception weights; no learned BCactor or RLactions. Failures and unqualified fits are preserved. This is one selected public-sample pass, with small SCdepth margin and unresolved unusual-SFP-grasp coverage; final scenes stay sealed. Earlier sections are historical. [Results, commands and videos](docs/experiments/2026-10-18-five-repair-cycles.md).

## October 18: gated close-up pose update and SC insertion result

V34 passes **4/4 matched SC development cases** (zero/five obstacle cards, both ports), followed by a passing SC round in the unchanged public sample: **15.510 mm depth, 0.141 mm lateral error, 0.469° orientation**, toolkit completion, no harm. Five fresh agreeing images support one bounded plug-pose update at 25 mm; the port frame stays fixed. Tighter lateral centering and stronger lateral tracking are also needed. All used-stage p95 timings pass; these are explicit RGB/public-state control results, not BC or RL.

The complete sample is **170.57/300, 2/3 strict** because SFP trial 2 has a 3.53° biased grasp estimate before contact. Its actual roll exceeds the original SFP training range; close-up predictions remain wrong. Repeats stop, and the next targeted repair is SFP orientation coverage rather than a stiffness-only change. No new training or collection is running. The SC model's original 1.026° qualification failure remains explicit; four successful development cases do not establish general reliability. Final scenes remain sealed.

[Detailed evidence, commands, images and next repair](docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

# AI for Industry Challenge Toolkit

> **October 17 current pipeline:** rule-based SC/SFP transport → requested-port coarse/native-image refinement and visual acquisition → explicit visual/proprioceptive approach to about **25 mm** → shared full-connector-pose BC insertion actor. Frozen sample-derived transport/handoff: **119/124** (SC22/24, SFP97/100); separate exposed released scenes **2/6**. Four-rollout insertion diagnostic: **SFP1/2, SC0/2**; subsequent complete sample **165.42/300 points but0/3strict insertions**. No recovery or newRL in either test. The actor no longer owns the entire150mm approach in this diagnostic. See [milestones and remaining gaps](docs/STATUS.md#major-milestones--transport-handoff-and-insertion-october-17). Earlier dated running/queued statements are historical; step5 remains open.


> October 1 supervised continuation: the project is testing a three-model system: one shared SFP/SC temporal transport actor, a separate task-conditioned selected-port perception model, and a separate port-relative insertion actor. The earlier four single-observation BC transport variants reached 0/16 stable near-port handoffs; the new bounded temporal comparison also failed its offline route-subgoal gate. Perception, insertion, backtracking, and RL remain gated. See [current status](docs/STATUS.md) and the [temporal continuation record](docs/experiments/2026-10-01-temporal-three-model-continuation.md).

> October 1 experiment status: the SC Low Lateral Bypass teacher data has an audited motion-safe cleaned BC derivative. Four supervised transport checkpoints completed 16 matched Gazebo development rollouts; none passed insertion or the stable transport handoff gate. See [current status](docs/STATUS.md) and the [SC cleaned BC record](docs/experiments/2026-10-01-sc-llb-cleaned-bc.md) before using a checkpoint.

**September 30 latest:** teacher transport gate accepted. Two same-world Cartesian SC teachers fully inserted, providing v9 with 1,751 transport and 503 local chunks, plus the verified SFP source. The balanced-port BC retry finished 1,500 updates and failed both autonomous development starts: final planned-standoff errors 103 and 603 mm versus 76 and 74 mm for the prior masked actor; neither reached handoff or inserted. A bounded trace/video audit found early wrong-route commands without sustained cable snag. Stop routine feed-forward BC. A checkpoint-provenance audit found that this run reused the full action head from an older mixed-SC-route model and had only two new SC episodes; it was not a clean test of the new Low Lateral Bypass route. Audit clean route data and initialization before deciding on a route-subgoal model. No new training was started for this audit. True visibility, autonomous two-actor insertion and RL remain gated; reserved final scenes stay sealed. [Evidence and exact artifacts](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

> **Transport continuation:** the user accepted the transport gate; four changed-card-layout routes reached standoff/handoff. Verified successful SC command targets are prepared separately for transport and local BC. True-visibility label checks are ongoing; autonomous two-actor insertion and RL gates remain open. [Current execution](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

> **2026-09-30 update:** [Oracle force diagnosis](docs/experiments/2026-09-30-force-oracle.md) identified plug/SC-port contact in a reproduced old-route failure, corrected a Gazebo contact-reporting bug, and validated the explicit tilt-before-return route with collisions intact. These are privileged development trials; autonomous policy and new-start generality gates remain open.

> **Returning to `feat/hybrid-train`?** Start with the [working documentation](docs/README.md):
> [current status](docs/STATUS.md), [experiment history](docs/EXPERIMENTS.md),
> [rootless setup and local model testing](docs/LOCAL_WORKFLOW.md), and
> [artifact locations](outputs_README.md). Reviewed 2026-09-24. The frozen
> natural-cable pose estimate reached 0.322/0.626 mm lateral median/p95. Generic
> pose concatenation and an additive correction both failed their control
> gates. The repaired controller now predicts four complete connector
> waypoints at 50 ms intervals relative to the port, learned from the recorded
> teacher target commands. Fixes cover blended-versus-target labels, lost
> within-chunk paths, checkpoint selection, and live clipping. A deterministic
> 2.00M-parameter AIC-RPDP BC arm achieved **6/6 local 8 mm seating events** on
> the fixed near-port development set and **8/8** on eight additional near-port
> starts, at 19.86 ms p95 inference. A later recorded five-start check reached
> **4/5** at 20.12 ms p95. These configs override the physical approximately
> 45.8--48.72 mm SFP depth with an 8 mm surrogate and use privileged near-mouth
> resets, so they are local-control results rather than normal full-task
> insertions. The
> matched diffusion arm reached 2/6; the earlier valid DPPO arm
> remains at 1/6 and was not continued. The supervised candidate passes its
> development gate, while the reserved final split remains sealed. See the
> [RPDP/DPPO execution record](docs/experiments/2026-09-22-rpdp-dppo.md),
> [generic controller record](docs/experiments/2026-09-22-pose-conditioned-gru-policy.md),
> and [explicit-correction record](docs/experiments/2026-09-22-explicit-pose-correction.md).
> The subsequent four-mode SERL continuation exposed policy-mode collapse,
> repaired full-mixture anchoring and optimizer learning-rate restoration, and
> recovered the frozen **3/8** metrics-only result without improving it. A new
> three-camera rerun recorded **1/8 BC, 1/8 selected RL, and 3/8 selected RL
> plus measured-path backtracking**. Per-episode review shows mainly lateral
> divergence and port-lip stalls; it does not establish cable-snag recovery.
> The selected policy remains an SFP-to-NIC near-port diagnostic. SC-to-SC and
> multi-card cable routing have not been validated. See the
> [SERL experiment](docs/experiments/2026-09-22-serl-mixture-recovery.md),
> [video failure analysis](docs/experiments/2026-09-23-serl-video-failure-analysis.md),
> and [ordered SC cable-snag plan](docs/SC_CABLE_SNAG_RECOVERY_PLAN.md).
> A later five-card Gazebo route probe produced an abrupt high-force approach
> stall, but did not identify a named contact pair. The proposed core
> continuation is [hierarchical recovery](docs/HIERARCHICAL_RECOVERY.md):
> measured-pose local retreat, then an earlier route checkpoint and a different
> cable corridor after repeated blockage. This is a research hypothesis, not
> a validated insertion gain. The first 17-trace force/TCP audit and two fresh
> five-card Gazebo controls are diagnostic only. The later Isaac mechanics
> repair and SC label audit passed their bounded checks. The active
> [shared SFP/SC execution plan](docs/experiments/2026-09-24-shared-actor-sc-serl-execution.md)
> is testing native-image SC perception on new resets before collecting the
> missing successful SC demonstrations and training one full-task actor. A
> frozen eight-scene pose check is running; no shared BC or SC RL result exists.
> Superseded plans, reports, and handoffs are in the [historical archive](obsolete/README.md).

[![build](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml)
[![style](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml)

![](../media/aic_banner.png)

The **AI for Industry Challenge** is an open competition for developers and roboticists aimed at solving some of the hardest, high-impact problems in robotics and manufacturing.

This repository contains the official toolkit to help participants start developing their solutions. For registration details, official rules, and FAQs, please visit the [AI for Industry Challenge event page](https://www.intrinsic.ai/events/ai-for-industry-challenge).

---

## Toolkit Guide

Welcome to the AIC toolkit documentation. This guide walks you through the complete workflow for participating in the challenge — from understanding the requirements to submitting your solution.

Follow the sections below to navigate through each phase of the process.

1. **📖 Understand the Challenge**
   - Read the [Challenge Overview](./docs/overview.md) to understand the goals.
   - Review the [Qualification Phase](./docs/phases.md#qualification-phase-train-your-model) to understand what you'll be building.
   - Review the [Scoring Guide](./docs/scoring.md) to understand how you'll be scored.

2. **🔧 Set Up Your Environment**
   - Follow the [Getting Started](./docs/getting_started.md) guide to set up and validate your development environment.
   - Run the evaluation container and set up your local workspace with Pixi.

3. **💻 Develop Your Policy**
   - Explore the [Scene Description](./docs/scene_description.md) to learn how to customize and explore the environment.
   - Review [AIC Interfaces](./docs/aic_interfaces.md) to understand available interfaces to communicate with sensors and actuators.
   - Consult [AIC Controller](./docs/aic_controller.md) to learn about controlling the robot.
   - Consult the [Challenge Rules](./docs/challenge_rules.md) to ensure compliance.
   - Start with the [Policy Integration Guide](./docs/policy.md) to implement your solution.
   - See [Participant Utilities](./docs/participant_utilities.md) for a list of helpful tools.

4. **🧪 Test Your Solution**
   - Use the provided simulation environment to test your policy.
   - Run `aic_engine` with the `sample_config` in [`aic_engine/config/`](./aic_engine/config/) to test different scenarios. For more information on running the `aic_engine` with different configs, see the [aic_engine README file](./aic_engine/README.md).
   - Create your own test scenarios by following the configuration example in [`aic_engine/config/`](./aic_engine/config/) to run with `aic_engine`.
   - Refer to [Troubleshooting](./docs/troubleshooting.md) if you encounter issues.

5. **📦 Submit Your Entry**
   - Package your solution following the [Submission Guidelines](./docs/submission.md).
   - Test your container locally before submitting following [these instructions](./docs/submission.md#verify-locally).
   - Submit through the official portal following [these instructions](./docs/submission.md#2-upload-your-image-to-our-registry).

---

## Toolkit Architecture

![AIC Competition Components](../media/aic_competition_components.png)

The AI for Industry Challenge toolkit is divided into **two main components**:

### 1. Evaluation Component (Provided - Run by Organizers)

This component provides the complete evaluation infrastructure:
- **`aic_engine`** - Orchestrates trials and computes scores.
- **`aic_bringup`** - Launches simulation environment (Gazebo, robot, sensors).
- **`aic_controller`** - Low-level robot control with force management.
- **`aic_adapter`** - Sensor fusion and data synchronization.

**What you receive:** Standard ROS sensor topics providing camera images, joint states, force/torque measurements, and TF frames.

### 2. Participant Model Component (Your Implementation - What You Submit)

This is what you develop and submit:
- **A ROS 2 node** that follows the behavioral requirements defined in [Challenge Rules](./docs/challenge_rules.md).
- **Your custom logic** - Code to process sensor data and command the robot to insert cables.

**What you provide:** A container with a ROS 2 Lifecycle node named `aic_model` that responds to the `/insert_cable` action and outputs robot motion commands via standard ROS topics/services.

**Convenient Entry Point:** We provide an `aic_model` framework that handles all the ROS 2 boilerplate and lifecycle management. You simply implement a Python policy class that gets dynamically loaded at runtime. See the [Policy Integration Guide](./docs/policy.md) for details.

### Development and Submission Workflow

> [!IMPORTANT]
> **ROS 2 Distribution:** The official evaluation of all submissions will be conducted using **ROS 2 Kilted Kaiju**. If you choose to develop or test your policy using a different ROS 2 distribution (e.g., Humble or Jazzy), it is entirely your responsibility to ensure compatibility and support. Please note that **inter-distro communication is not guaranteed and not officially supported**.

**Development Options:**
- Develop inside a container (recommended - matches evaluation environment).
- OR develop in native Ubuntu 24.04 environment (requires all dependencies).

**Submission Requirements:**
- Package your solution using the provided `aic_model` Dockerfile.
- Submit your container - it must respond to standard ROS inputs and command the robot to insert cables.
- Your container interfaces with the evaluation component via ROS topics.

---
## Repository Structure

```
aic/
├── aic_adapter/          # Adapter for interfacing between model and controller
├── aic_assets/           # 3D models and simulation assets
├── aic_bringup/          # Launch files for starting the challenge environment
├── aic_controller/       # Robot controller implementation
├── aic_description/      # Robot and environment URDF/SDF descriptions
├── aic_engine/           # Trial orchestration and validation engine
├── aic_example_policies/ # Example policy implementations
├── aic_gazebo/           # Gazebo-specific plugins and configurations
├── aic_interfaces/       # ROS 2 message, service, and action definitions
├── aic_model/            # Template for participant policy implementation
├── aic_scoring/          # Scoring system implementation
├── aic_utils/            # Utility packages and tools
├── docker/               # Docker container definitions
└── docs/                 # Comprehensive documentation
```

---

## Key Packages for Participants

### `aic_model` - Convenient Policy Framework (Recommended)
This package provides a ready-to-use ROS 2 Lifecycle node that dynamically loads and executes your Python policy implementation. It handles all ROS 2 boilerplate, lifecycle management, and challenge rule compliance, allowing you to focus on implementing your policy logic.
- **Location**: `aic_model/`.
- **Documentation**: [Policy Integration Guide](./docs/policy.md).
- **Tutorial**: [Creating a New Policy Node](./docs/policy.md#tutorial-creating-a-new-policy-node).

> **Note:** While we recommend using this framework, you may implement your own ROS 2 node from scratch as long as it adheres to the [Challenge Rules](./docs/challenge_rules.md).

### `aic_interfaces` - Communication Protocols
Defines all ROS 2 messages, services, and actions used in the challenge.
- **Location**: `aic_interfaces/`.
- **Documentation**: [AIC Interfaces](./docs/aic_interfaces.md).

### `aic_example_policies` - Reference Implementations
Example policies demonstrating different approaches and techniques.
- **Location**: `aic_example_policies/`.
- **README**: [aic_example_policies/README.md](./aic_example_policies/README.md).

### `aic_bringup` - Launch the Environment
Launch files to start the simulation, robot, and scoring systems.
- **Location**: `aic_bringup/`.
- **README**: [aic_bringup/README.md](./aic_bringup/README.md).

### `aic_engine` - Trial Orchestrator
Manages trial execution, validates participant models, and collects scoring data.
- **Location**: `aic_engine/`.
- **README**: [aic_engine/README.md](./aic_engine/README.md).

---

## Additional Documentation

### Challenge Information

* **[Challenge Overview](./docs/overview.md):** High-level summary of the competition goals and structure.
* **[Competition Phases](./docs/phases.md):** Details on Qualification, Phase 1, and Phase 2.
* **[Qualification Phase](./docs/qualification_phase.md):** Detailed technical overview of the qualification phase trials and scoring.
* **[Challenge Rules](./docs/challenge_rules.md):** Required behavior for participant models.
* **[Scoring](./docs/scoring.md):** Metrics and methods used to evaluate performance.
* **[Scoring Test Examples](./docs/scoring_tests.md):** Reproducible examples exercising each scoring tier with exact commands.

### Technical Documentation

* **[Getting Started](./docs/getting_started.md):** How to set up your local development environment.
* **[Policy Integration](./docs/policy.md):** Guide to implementing your policy in the `aic_model` framework.
* **[AIC Interfaces](./docs/aic_interfaces.md):** ROS 2 topics, services, and actions available to your policy.
* **[AIC Controller](./docs/aic_controller.md):** Understanding the robot controller and motion commands.
* **[Scene Description](./docs/scene_description.md):** Technical details of the simulation environment.
* **[Task Board Description](./docs/task_board_description.md):** Physical layout and specifications of the task board.
* **[Troubleshooting](./docs/troubleshooting.md):** Common issues and debugging strategies.

### Reference Materials

* **[Glossary](./docs/glossary.md):** Terminology and definitions used throughout the AI for Industry Challenge

### Submission

* **[Submission Guidelines](./docs/submission.md):** How to package and submit your final model.

---


## Support and Resources

- **Discussions**: Engage in conversations and ask questions about the challenge on [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/). The community is encouraged to participate in discussions and assist each other.
- **Issues**: Report any bugs or technical issues via [GitHub Issues](https://github.com/intrinsic-dev/aic/issues). Please refrain from using the Issue tracker for general questions about the challenge.
  - **Note:**: Review the list of [known issues](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22known%20issue%22) and [bugs](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) before opening a new ticket.
- **Event Page**: Visit the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) for official updates.

---

## License

This project is licensed under the Apache License 2.0 - see the individual package files for details.
The [aic_isaac](./aic_utils/aic_isaac/) folder contains files licensed under BSD-3 - see [aic_isaac/LICENSE](./aic_utils/aic_isaac/LICENSE).
