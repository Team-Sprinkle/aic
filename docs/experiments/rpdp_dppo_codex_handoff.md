# RPDP + DPPO: conversation summary and Codex handoff

Date: 2026-09-22

Scope: summarizes the discussion beginning with **“explain poseinsert in easy words”**, including the detailed explanation, publication/model-size/runtime questions, noisy-reference-frame concern, and BC-to-RL options. Minimal earlier project context is included so this document is self-contained.

## 1. Final decision from the user

**Use RPDP, train its diffusion policy with behavior cloning, then fine-tune the diffusion policy directly with DPPO.**

This supersedes the earlier suggestion to start with a Gaussian residual over a frozen diffusion policy. Residual RL and Gaussian-actor distillation were discussed as alternatives; they are **not the selected implementation direction**.

The intended pipeline is:

1. Estimate connector/port geometry and maintain a consistent target estimate.
2. Use RPDP-style relative-pose and visual conditioning, adapted to the available sensors.
3. Train a conditional diffusion action model with BC.
4. Fine-tune that diffusion policy with DPPO, including force-sensitive corrections and recovery.
5. Convert policy outputs into the robot's existing command interface using verified geometry.

No separate Gaussian robot-action head is required merely to enable RL. DPPO requires an appropriate stochastic denoising training process and its likelihood calculations.

## 2. Project context and intended behavior

- Task: Intrinsic AIC wire/cable connector insertion into a port on a card.
- Existing model: approximately **40M parameters**, with **three camera images, force/torque data, and proprioception**.
- The user has successfully estimated the port opening pose. Connector pose or a reliable connector-to-gripper transform is also needed for object-relative geometry.
- Existing actions are described as **body-frame delta poses**. Their exact translation/rotation convention has not been inspected.
- The user's cameras move with the robot hand. The target is expected to remain approximately fixed, though contact could cause movement or flex.
- Obstacles can require indirect approaches. Force can require retreat, search, or reorientation even when the geometric goal is forward.
- The user wants learned action selection, with BC followed by RL. Avoid replacing it with a fixed pose servo or a threshold-driven insertion/recovery state machine.
- Target hardware: a single NVIDIA RTX A6000 for deployment.
- A Dreamer 4-style system was also discussed. DPPO is now the selected actor-update approach; combining it with an imagined environment is a separate integration question.

This conversation inspected the public PoseInsert code and papers, not the user's current repository or controller. Confirm repository interfaces before implementation.

## 3. Naming: PoseInsert, PoseDP, and RPDP

**PoseInsert is the paper/project name.** Its two policy variants are:

| Variant | Conditioning |
| --- | --- |
| PoseDP | Connector pose relative to the target |
| RPDP | Relative pose plus goal-conditioned RGBD features, combined with pose-guided residual gated fusion |

The selected direction is **RPDP**, extended for the user's sensing and RL requirements. The original paper does not evaluate force-aware RL. Its robot provides passive compliance. [PoseInsert paper](https://arxiv.org/html/2505.09424v2)

### Why RPDP was preferred

The paper reports the following averages over six tasks:

| Variant | Standard conditions | Out-of-distribution conditions |
| --- | ---: | ---: |
| PoseDP | 81.7% | 76.7% |
| RPDP | 91.7% | 78.3% |

RPDP particularly helped the noisy USB insertion case. PoseDP remained competitive on several other tasks. There were ten trials per task and condition; these results do not establish a universal ranking. [Results](https://arxiv.org/html/2505.09424v2)

For this project, multimodal conditioning is appropriate because geometry alone does not reveal all obstacles or contact conditions. Retaining global visual information and force history is a proposed adaptation, not a demonstrated capability of the original object-focused RGBD pipeline.

## 4. Geometry: what is estimated, and what is predicted?

There are two different prediction problems:

| Component | Meaning of its prediction |
| --- | --- |
| Perception/pose estimator | Where the connector and port are **now** |
| Diffusion policy | Where the connector should be at **future trajectory steps** |

The estimated **current port pose** supplies the reference frame. The policy's predicted future connector pose does not define that frame.

### Frame convention

Write \({}^A T_B\) for the pose of frame B expressed in frame A; it maps coordinates from B into A.

| Symbol | Frame |
| --- | --- |
| W | Fixed robot-base/world frame |
| E | End-effector control frame/TCP |
| C | Connector mating/reference frame |
| P | Port reference frame |
| K | Camera frame |

If perception supplies both objects in camera coordinates:

\[
{}^P T_C=({}^K T_P)^{-1}\,{}^K T_C.
\]

The position and orientation parts are:

\[
{}^P p_C=({}^K R_P)^\top({}^K p_C-{}^K p_P),
\qquad
{}^P R_C=({}^K R_P)^\top{}^K R_C.
\]

Interpretation: subtract the positions, then express the offset and orientation along the port's axes. [Rigid-transform reference](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/)

An illustrative one-dimensional example:

| Quantity | Setup A | Setup B |
| --- | ---: | ---: |
| Port world position | 500 mm | 700 mm |
| Connector world position | 495 mm | 695 mm |
| Connector relative to port | −5 mm | −5 mm |

The local insertion relationship is identical. This removes a source of variation the network would otherwise need to learn.

The final correct relative transform need not be the identity. It depends on the frame definitions, desired mating orientation, and insertion depth. Successful trajectories establish that relationship.

## 5. RPDP architecture and learning

### 5.1 Pose representation and encoder

The released configuration uses nine pose values: three translation values and a continuous six-dimensional rotation representation. Its pose encoder has these dimensions:

| Path | Dimensions |
| --- | --- |
| Translation MLP | 3 → 128 → 128 → 64 |
| Rotation MLP | 6 → 128 → 128 → 64 |
| Combined pose feature | Concatenate to 128, then transformer encoder layer and projection |

The branches use normalization and nonlinearities. Their output is a learned 128-dimensional feature vector. [Encoder implementation](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/cnn.py)

A six-value rotation representation still has only three physical degrees of freedom. Two rotation-matrix columns can encode the orientation; decoding orthogonalizes the vectors and reconstructs the third axis. This representation addresses continuity issues, not obstacle avoidance or force control. [Rotation-representation paper](https://arxiv.org/abs/1812.07035)

Do not infer the actual tensor layout from “nine values.” The released wrapper packs/unpacks translation and rotation explicitly; inspect it before building a compatible dataset.

### 5.2 Current and goal visual observations

The RGBD branch compares current and goal observations. Crops include both objects and are resized to 320 × 320. This provides visual information about discrepancies that a noisy pose estimate may miss. [Visual branch description](https://arxiv.org/html/2505.09424v2)

A goal image is an additional input requirement. A port pose alone does not supply it. The receiving agent must establish how a suitable goal reference is provided for the actual task.

The code uses a shared image encoder, combines the two observations' features, and produces a 1200-dimensional visual feature before projection for fusion. [Image encoder and policy wrapper](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/policy.py)

### 5.3 Exact direction of the gate

Let p be the 128-dimensional pose embedding and v the visual feature projected to 128 dimensions:

\[
g=\sigma(Wp+b),\qquad f=p+g\odot v.
\]

The gate is computed from **pose features** and regulates **visual corrections**. It has per-feature weights, not one physically interpreted confidence value. The pose path remains directly present. The original gate is not explicitly conditioned on force or a pose-estimator confidence score. [Fusion implementation](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/cnn.py)

Our interpretation: this favors building predictions around geometry while allowing visual corrections. It does not guarantee that the gate detects unreliable estimates or identifies task phases. A multimodal/force-conditioned gate would be an extension and should be identified as such.

### 5.4 What the diffusion output means

The policy predicts a trajectory of future connector poses relative to the port:

\[
{}^P T_{C,t}
\longrightarrow
\left[{}^P\widehat T_{C,t+1},\ldots,{}^P\widehat T_{C,t+H}\right].
\]

This describes desired future placements, rather than directly emitting the user's body-frame deltas. [PoseInsert formulation](https://arxiv.org/html/2505.09424v2)

The released pose training script configures one observation, a 20-step prediction horizon, nine action values per step, and a 128-dimensional observation embedding. These are code settings, not necessary properties of the method. [Training configuration](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/train_pose.py)

Port-relative trajectories can describe alignment, insertion, sideways search, or retreat. The coordinate choice does not impose straight-line or monotonic movement toward the goal.

### 5.5 BC objective and denoising

Given a demonstrated clean future sequence \(A_0\), choose a noise level k and create:

\[
A_k=\alpha_k A_0+\beta_k\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Train a conditional network to recover the clean sequence:

\[
\widehat A_0=f_\theta(A_k,k,f),\qquad
L_{\rm BC}=\|\widehat A_0-A_0\|^2.
\]

The released code uses a conditional 1D U-Net with DDIM and selects `prediction="sample"`, meaning the target is the clean trajectory. Its default inference sampler uses 20 denoising iterations. [Diffusion implementation](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/diffusion.py)

The paper shows a noise-prediction equation but also states sample prediction; the explicit training configuration corroborates the latter. Preserve scheduler/prediction-type consistency when adapting the implementation.

Distinguish two different axes:

- **Trajectory time:** future moments at which commands will be executed.
- **Denoising iteration:** internal computation refining the proposed sequence.

The robot does not move after every denoising iteration. At deployment, only a chosen part of a predicted sequence is executed before replanning.

## 6. Converting connector poses into robot commands

The gripper must be positioned according to how it holds the connector. With a grasp relationship \({}^C T_E\):

\[
{}^W T_E^{\rm desired}
= {}^W T_P\;{}^P\widehat T_C\;{}^C T_E.
\]

Interpretation: locate the port, locate the desired connector relative to the port, then locate the gripper relative to the connector.

The current observed connector and hand poses can define the grasp relationship:

\[
{}^C T_E=({}^W T_C)^{-1}\,{}^W T_E.
\]

For the user's body-frame delta interface:

\[
\Delta T_b=({}^W T_E^{\rm current})^{-1}\,{}^W T_E^{\rm desired}.
\]

These equations are geometry; the policy chooses the desired trajectory. Conversion to the controller's six numbers still requires checking whether it uses separate Cartesian translation/rotation-vector increments or full Lie-algebra exponential coordinates. These are not interchangeable under finite rotation.

For a faithful RPDP implementation, retain port-relative trajectory outputs and use a deterministic adapter to the body-frame command interface. Retaining direct body-delta outputs inside the learned policy was also discussed, but would be an RPDP-inspired adaptation. Do not silently switch between these formulations or use mismatched BC labels.

Confirm TCP, grasp offset, insertion depth, timestamp, normalization, clipping, and action-chunk frame semantics. If the connector slips or flexes relative to the gripper, a fixed grasp transform may be insufficient.

## 7. Noisy reference frames: the user's concern

The user correctly observed that a noisy port estimate can cause the reference frame to wobble over time, affecting both observations and commands.

This does not mean every reference-frame error becomes an equal action error. A consistent input/output transformation can cancel some coordinate effects. It cannot generally correct a mistaken destination.

### One-dimensional example

Assume connector position is known exactly:

- True port: 0 mm.
- Connector: 10 mm.
- Estimated port: +2 mm.
- Relative input: 10 − 2 = 8 mm.

| Intended behavior | Predicted relative coordinate | Converted physical coordinate |
| --- | ---: | ---: |
| Hold current position | 8 mm | 2 + 8 = 10 mm: correct |
| Reach estimated port center | 0 mm | 2 + 0 = 2 mm: wrong by 2 mm |

If the estimate alternates between +2 and −2 mm, goal-directed commands can alternate too. Orientation noise can change the inferred insertion direction. Relative coordinates provide structure; they do not solve state estimation.

### Camera setup correction

The paper describes a front Orbbec Dabai RGBD camera observing the tabletop. Do not assume the main experimental camera follows the gripper like the user's wrist cameras. [Experimental setup](https://arxiv.org/html/2505.09424v2)

For a moving wrist camera and an approximately stationary target:

\[
{}^W\widehat T_P(t)
= {}^W T_E(t)\;{}^E T_K\;{}^K\widehat T_P(t).
\]

The port's camera-frame coordinates should change with camera motion. After accounting for that motion, its base-frame pose should stay approximately constant. Extra variation can come from perception, calibration, timing, or real target movement.

### Proposed adaptation for this task

These are engineering recommendations from the discussion, not a claim that the original RPDP implements them:

1. Transform detections into a fixed robot-base frame using FK at the image capture time.
2. Maintain a persistent target estimate using observations over time, and across cameras when calibration permits.
3. Track uncertainty and handle implausible jumps. Use rotation-aware estimation rather than independently averaging Euler angles.
4. Derive the policy's current relative geometry from the consistent estimate.
5. Preserve visual and force feedback because filtering cannot remove every systematic bias or grasp error.

If the card can shift or flex, allow target-state updates rather than assuming permanent immobility. Estimation does not dictate actions; the policy remains learned. [Lie-group state-estimation reference](https://arxiv.org/html/1812.01537v9)

An implementation consequence is to associate each trajectory chunk with the reference estimate used to generate it. Do not accidentally reinterpret an old relative trajectory in a newly shifted reference frame.

## 8. Publication, size, latency, and A6000 feasibility

### Publication history

- First arXiv release: **May 14, 2025**.
- Revised version discussed: **March 23, 2026**.
- These are preprint dates; this discussion did not establish a separate conference/journal publication date. [arXiv history](https://arxiv.org/abs/2505.09424)

### Model size and runtime

| Variant | Calculated policy parameters | Reported RTX 4060 prediction latency | Reciprocal prediction rate |
| --- | ---: | ---: | ---: |
| PoseDP | Approximately 17.38M | Approximately 70 ms | Approximately 14 Hz |
| RPDP | Approximately 36M | Approximately 140 ms | Approximately 7 Hz |

**Evidence distinction:** latency is reported in the paper. Parameter counts were calculated in this conversation from the released layer definitions and training settings; the paper does not state these counts. [Runtime](https://arxiv.org/html/2505.09424v2), [released architecture](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/policy.py)

The inspected source revision was `24641dd3c6cc96fccd1e932a61a150bbb7a115c0`.

Counting details retained for reproducibility:

- Diffusion decoder: 17,214,729 parameters under the inspected training settings.
- Pose encoder: 168,064 parameters.
- Registered readout embedding: 512 parameters.
- PoseDP total: 17,383,305.
- RPDP total: roughly 35.95–35.97M for the encoder configuration variations examined.
- Counts include registered modules retained in the code even when unused by the forward path.
- Counts exclude the separate perception/pose-estimation system.
- These were static architecture counts, not a checkpoint audit or a GPU execution benchmark. RPDP loads an external visual-encoder configuration, so exact reproduction must inspect that configuration.

The A6000 has **48 GB VRAM**. Based on these policy sizes and reported operation on an RTX 4060, single-A6000 policy deployment should be practical. No A6000 latency was measured or found in this discussion. [NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)

Do not assume the paper's timings include every perception, preprocessing, communication, or controller component. Benchmark the complete application. Executing an action chunk at 20 Hz is also different from incorporating fresh force observations into every 50 ms decision.

## 9. Why diffusion can be trained with RL

A diffusion policy is already probabilistic. Conceptually:

\[
a=F_\theta(o,\epsilon),\qquad \epsilon\sim\mathcal N(0,I).
\]

Different initial noise can produce different trajectories. Even a deterministic denoising procedure can define a stochastic policy when initialized from random noise.

A Gaussian actor instead explicitly produces a mean and scale:

\[
a=\mu_\theta(o)+\sigma_\theta(o)\odot\epsilon.
\]

The practical difference is likelihood evaluation. Gaussian action densities are easy to compute. A diffusion policy's final action density is generally harder to evaluate, even though sampling is straightforward.

Ordinary PPO uses new/old policy likelihoods. Pretending the final denoised action came from a single Gaussian does not generally produce the correct diffusion-policy update.

RL as a whole does not require stochastic actors. TD3 is an example with a deterministic actor and exploration noise. That was a conceptual clarification, not the selected algorithm. [TD3](https://arxiv.org/abs/1802.09477)

## 10. Selected BC-to-RL route: diffusion + DPPO

DPPO treats denoising transitions as part of an augmented decision process. It uses their tractable Gaussian conditional likelihoods for policy-gradient updates, rather than requiring a simple density for the final robot action. It directly fine-tunes a BC-pretrained diffusion policy. [DPPO paper](https://arxiv.org/html/2409.00588v3)

The chosen workflow is:

1. **BC:** train RPDP's conditioned diffusion policy on demonstrations.
2. **RL rollout:** use a DPPO-compatible stochastic denoising sampler to generate trajectory proposals and execute commands.
3. **Feedback:** collect rewards and new observations, including force and achieved motion.
4. **Update:** apply DPPO to the diffusion policy and its supported trainable conditioning components, with an appropriate value model.
5. **Evaluation:** assess completion, excessive forces, obstacles, recovery, and retained approach competence.

DPPO discusses stochastic DDIM during training, including nonzero transition noise so denoising-step likelihoods are meaningful. Deterministic evaluation sampling can be considered separately. Its implementation also supports choices such as fine-tuning a subset of denoising steps. Do not assume an unchanged deterministic PoseInsert sampler is ready for DPPO. [DPPO sampling discussion](https://arxiv.org/html/2409.00588v3)

### Integration implications for the receiving agent

These are consequences of the chosen design to verify against the code:

- Preserve the BC prediction convention (`sample` versus `epsilon`) when constructing denoising means and scheduler updates. Clean-sample prediction is not a reason to add a Gaussian action head.
- Keep normalization, action representation, sampling noise, and recorded likelihoods consistent across rollout and update.
- Retain the denoising information required by the selected DPPO implementation. Final robot commands alone are not enough to reconstruct its update naively.
- Distinguish the horizon predicted by diffusion from the number of actions executed before obtaining new observations. Long open-loop chunks delay force reaction.
- Treat conversion from port-relative trajectories to controller commands as a documented environment/action adapter. Preserve the reference state needed for that conversion.
- Add force history, achieved motion, and recent commands to the actor's available conditioning, and train with those signals. They are extensions to the original RPDP setup.
- Allow successful behavior to temporarily increase goal distance. Recovery and obstacle avoidance should not be suppressed by an overly dominant pose-distance objective.
- Evaluate whether updating only the diffusion head is enough. If the pose/visual/force features need to adapt, explicitly decide which conditioning modules receive RL gradients.

The exact rewards, execution horizon, force encoder, and three-camera fusion implementation were not finalized in the conversation.

## 11. Alternatives discussed, retained for context

These are not the current plan:

| Alternative | Discussion outcome |
| --- | --- |
| Frozen diffusion BC + Gaussian residual | ResiP trains per-step PPO corrections conditioned on the base action. We proposed force-conditioned corrections, but the user selected direct diffusion RL instead. |
| Diffusion teacher → Gaussian student → RL | Possible, but requires supervised transfer and can lose multimodal behavior. A new Gaussian head does not automatically inherit the diffusion policy. |
| Deterministic actor RL | Demonstrates that stochastic architecture is not universally required; not selected. |
| Original Dreamer 4 PMPO actor update | Uses policy log-probabilities and a behavioral-prior KL. It is a different update from DPPO. |

References: [ResiP](https://arxiv.org/html/2407.16677v1), [Dreamer 4](https://arxiv.org/html/2509.24527v1).

If a Dreamer-style world model is used as the environment for DPPO, that is an additional adaptation. It must provide action-consistent future force, pose, and proprioception information. Do not substitute recorded future force under different actions or integrate commanded deltas as if they were always achieved during contact.

## 12. What Codex should do next

1. **Honor the selected direction: RPDP + diffusion BC + DPPO.** Do not default back to residual RL or Gaussian distillation based on an earlier recommendation.
2. Inspect repository instructions, model components, data, controller conventions, and existing RL infrastructure.
3. Establish connector/port/TCP frames, current and goal pose definitions, calibration, timestamps, and grasp assumptions.
4. Define how stationary-target estimation works with moving wrist cameras; validate it on recorded observations.
5. Map the RPDP pose encoder, goal-conditioned visual branch, and fusion into the available three-camera pipeline. Preserve obstacle-relevant information.
6. Specify the goal visual reference and force-history conditioning. Identify every departure from the published RPDP architecture.
7. Choose and document the learned trajectory representation and its deterministic conversion to body-frame commands. Keep BC labels and RL execution consistent.
8. Establish a BC baseline, then integrate DPPO with the correct denoising likelihoods, exploration, rollout storage, and execution horizon.
9. Measure the actual parameter count, GPU memory, and end-to-end latency in the user's environment. The numbers above are references, not acceptance results.
10. Evaluate noisy/stale pose estimates, held-out placements, obstacles, contact jams, and recovery. Separate confirmed results from assumptions.

The purpose of the geometric structure is to make the goal easier for the network to use. The policy must still learn action selection from demonstrations and rewards, including when to search or move away from the goal.
