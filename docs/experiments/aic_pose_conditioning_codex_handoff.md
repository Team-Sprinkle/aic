# Intrinsic AIC insertion: pose conditioning, geometry, and IL-to-RL handoff

Prepared 2026-09-21. This document transfers the design discussion into an existing Codex session. It contains the task constraints, recommendations, implementation guidance, and all research references retained from the discussion, with two additional references for the recommended conditioning layers.

**Read sections 1–3 first.** Sections 4–10 specify the implementation and validation details. Section 11 is the complete paper catalog; section 12 contains robotics background references.

This is a proposed design, not a report of completed experiments. No repository, controller wrapper, dataset schema, or model implementation was inspected for this handoff. Verify those interfaces before changing code. Paper findings and our proposed adaptations are identified separately.

## 1. Task and constraints

- Task: Intrinsic AIC wire/cable connector insertion into a port on a card.
- The user already has a working pose estimator for the port opening.
- Existing policy: approximately **40 million parameters**, consuming **three camera images, force/torque sensing, and proprioception**.
- Policy outputs are described as **delta poses in the body frame**. The precise controller convention still needs verification.
- Simply feeding the pose into the network did not work. The cause has not been established.
- There can be obstacles between the current hand position and the port. Alignment with the port is especially useful near insertion, while approach requires scene understanding and obstacle avoidance.
- Training should start with imitation learning (IL), followed by reinforcement learning (RL) to improve force-sensitive behavior, corrections, and recovery.
- The user wants a learned policy with flexibility to override an inaccurate pose estimate, retreat, search, and retry. A fixed pose servo or a rigid pose-triggered state machine is not the intended solution.
- A transformer policy is a likely architecture. Another possibility is an actor attached to separately pretrained tokenizer/dynamics components in a Dreamer 4-style system.
- Model additions should be small and compatible with existing training and action interfaces.

Do not assume the port opening pose is already the desired gripper pose. Desired mating orientation, insertion depth, connector geometry, and the connector-to-gripper transform must be accounted for.

## 2. Recommended starting design

**Compute the desired gripper pose relative to the current gripper; encode that relative geometry; repeatedly inject it into the learned action network with feature modulation; allow multimodal context to regulate that influence; keep the existing body-frame action interface.**

In plain language:

1. Convert “the opening is here in the camera image/frame” into “the desired hand pose is this far and this orientation from the current hand.”
2. Give this goal information a small dedicated encoder.
3. Use the resulting embedding to modify features inside the action network, so the policy has several opportunities to use the goal.
4. Let vision, force history, proprioception, and recent actions determine how strongly the goal should influence those features.
5. Train the resulting policy with demonstrations, then improve it with RL.

The geometry tells the policy where the destination is. The policy learns the next useful action, including movements away from the destination when contact or obstacles require it.

### 2.1 What creates the inductive bias?

| Component | Bias introduced | What it does not establish |
| --- | --- | --- |
| Relative goal geometry | Removes the need to learn camera/base-to-hand geometric subtraction from examples | Collision avoidance, correct grasp calibration, or contact state |
| Smooth rotation representation | Avoids some discontinuities of Euler angles and quaternion signs | A physically correct control law |
| Dedicated pose encoder | Makes geometry a distinct, compact modality | That the policy will use it |
| Repeated FiLM/adaptive normalization | Makes goal information directly affect intermediate action features | That it will always dominate, or be useful under every condition |
| Learned context gate | Allows pose influence to vary with scene, contact, uncertainty, and history | A calibrated confidence score or a guaranteed phase classifier |
| Action-conditioned auxiliary predictions | Encourages features to connect geometry and contact to the consequences of actions | A guarantee that the final actor uses those features |
| Diverse target positions and recoveries | Makes task success depend on the goal and feedback | Generalization outside the represented situations |

**No architecture alone guarantees useful pose utilization.** Correct geometry, informative training variation, and controlled intervention tests are necessary. A gate can collapse; auxiliary heads can learn shortcuts; an actor can ignore even well-encoded pose features.

### 2.2 Ranked methods for this project

This is an implementation priority ranking for the user's constraints, not a ranking of published success rates. The methods can be combined.

| Rank | Method | How to use it here | IL → RL flexibility | Main limitation |
| --- | --- | --- | --- | --- |
| 1 | Relative-pose conditioning with repeated FiLM/adaptive normalization and an optional learned context gate | Add a small pose encoder to the existing multimodal action network; compare ungated and gated conditioning | High; retain the action head and fine-tune trainable policy components | Still needs data variation and utilization checks |
| 2 | Learned BC policy plus force-aware residual RL | Keep a competent BC base; train a per-step correction policy using pose, force history, and the proposed base action | High for local corrections; can preserve existing approach skill | Small residuals can prevent large retreats or rerouting |
| 3 | Action-conditioned geometry/contact auxiliary learning | Predict achieved relative motion, next contact, or wrench from policy features and executed actions | High; use during IL and/or RL | Supervision quality and shortcut learning |
| 4 | Small feature experts with learned routing | Share encoders; use a few small expert feature branches with context-dependent routing and a shared action head | High | More complexity and possible expert collapse |
| 5 | Target-relative learned future poses/waypoints | Predict connector poses relative to the port, then convert them to robot commands | Compatible in principle | Pose/grasp errors enter command conversion; changes labels and possibly the dynamics interface |
| 6 | Spatial 3D relative attention | Attach geometry to visual/action tokens when reliable depth or 3D correspondence is available | Compatible with a suitable actor | Larger representation/calibration change than needed for a first experiment |
| 7 | Explicit equivariant policy architecture | Use the actual task symmetries to restrict the network | Possible, but involves architectural work | Physical gravity, cable anchoring, robot reachability, and image formation complicate the symmetry assumptions |

**Implementation choice:** start with rank 1. Rank 3 is a useful follow-up if representations remain weak. Rank 2 is especially attractive when the current BC policy already navigates well or uses a diffusion head that is inconvenient to fine-tune directly with RL.

An additional cheap representation experiment is to supply a few connector/port keypoint errors or insertion-axis features. These are suggested extensions, not prerequisites or claims from a particular paper; details appear in section 4.5.

## 3. What to borrow from PoseInsert, and what to change

The relevant paper is [PoseInsert: Exploring Pose-Guided Imitation Learning for Robotic Precise Insertion](https://arxiv.org/html/2505.09424v2), summarized in section 11.1.

For this project, borrow the idea of describing insertion through relative object geometry. Our proposed adaptation retains the user's multimodal policy and body-delta action interface, adding pose as a learned conditioning pathway.

Pose information should be allowed to become less influential when it is stale, occluded, inconsistent with vision, or contradicted by contact. Global visual information must remain available for navigating around the card and other objects.

We have not established that a complete replacement with PoseInsert would outperform this smaller modification. Its reported results should not be treated as evidence for this proposed force-aware IL-to-RL adaptation.

## 4. Geometry and controller contracts

### 4.1 Define frames before choosing network layers

Use the convention:

\[
{}^A T_B = \begin{bmatrix}{}^A R_B & {}^A p_B \\ 0 & 1\end{bmatrix}
\]

This is the pose of frame B expressed in frame A, and maps B-coordinate points into A coordinates.

| Symbol | Frame |
| --- | --- |
| W | Robot base or a fixed world frame; choose one explicitly |
| E | End-effector control frame/TCP used by the action wrapper |
| C | Connector mating frame, ideally at a mechanically meaningful tip/reference point |
| P | Port reference frame |
| K | Camera frame |
| S | Force/torque sensor frame |

Composition and inverse:

\[
{}^A T_C = {}^A T_B\,{}^B T_C,
\qquad
T^{-1}=\begin{bmatrix}R^\top & -R^\top p\\0&1\end{bmatrix}.
\]

If the camera moves with the hand:

\[
{}^W T_P = {}^W T_E(q_t)\,{}^E T_K\,{}^K T_P.
\]

Use the robot state corresponding to the camera capture time. A correct extrinsic transform combined with the wrong timestamp can still produce a bad relative pose.

Background: [Modern Robotics, homogeneous transforms](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/).

### 4.2 Opening pose → desired hand pose → relative goal

Let:

- \({}^E T_C\): connector pose relative to the control frame, assumed fixed only if the grasp is rigid.
- \({}^P T_C^*\): desired seated connector pose in the port frame, including mating orientation and insertion depth.

Then:

\[
{}^W T_E^*
= {}^W T_P\,{}^P T_C^*\,({}^E T_C)^{-1}.
\]

Compute the goal expressed relative to the current hand:

\[
E_b = ({}^W T_E)^{-1}\,{}^W T_E^*
= \begin{bmatrix}R_{\rm err}&p_{\rm err}\\0&1\end{bmatrix}.
\]

Explicitly:

\[
p_{\rm err}=R_E^\top(p_E^*-p_E),
\qquad R_{\rm err}=R_E^\top R_E^*.
\]

This aligns the input geometry with the body-frame action vocabulary. A goal to the hand's local right remains a local-right error when the whole coordinate description changes.

Alternatively, the connector expressed in the port frame is:

\[
{}^P T_C = ({}^W T_P)^{-1}\,{}^W T_E\,{}^E T_C.
\]

That representation is useful for lateral alignment and depth. The two representations are related, but their numerical values and meanings differ. Name tensors accordingly.

If the connector slips, bends relative to the gripper, or has an uncertain grasp, FK plus a fixed offset is insufficient. Track or estimate the connector pose, or represent uncertainty/history. Treat the connector's rigid mating part separately from the deformable cable.

### 4.3 Recommended input representation

Start with:

\[
z_{\rm pose}=
[p_{\rm err}/s_p,\;\operatorname{rot6d}(R_{\rm err}),\;c,\;m,\;\Delta t/s_t].
\]

Here c is an available confidence feature, m is a validity mask, and Δt is pose age. Do not invent a calibrated confidence score if the estimator does not provide one. Use only available metadata and document its meaning.

- The base geometry is **3 translation + 6 rotation = 9 numbers**.
- Six rotation numbers can be the first two columns of a rotation matrix, concatenated in an explicitly documented order.
- Six degrees of freedom do not require storing exactly six scalars.
- Translation and sensor quantities need meaningful scaling. Compute statistics on the training split only; preserve sensitivity at insertion-scale distances.
- If using multiscale distance features, retain metric displacement as well so the network can distinguish coarse approach from fine correction.
- Handle invalid estimates with a mask and a defined placeholder. A zero translation vector alone is ambiguous because it can also mean alignment.

An alternative is a local six-vector \(\log(E_b)^\vee\). It is convenient near alignment but has rotation branch issues near π, and its translation coordinates are not generally the Cartesian displacement. The 9-number representation is the suggested first experiment.

Background: [continuous rotation representations](https://arxiv.org/abs/1812.07035), [micro Lie theory](https://arxiv.org/html/1812.01537v9).

### 4.4 SE(3), se(3), and the meaning of the action

- **SE(3)** is the space of rigid poses/transforms.
- **se(3)** is the associated Lie algebra; a six-vector can represent a local rigid-motion increment through the exponential map.
- Neither construction encodes obstacle avoidance, insertion success, or a requirement to move toward the target.

This document orders six-vectors as translation first, rotation second. Some robotics libraries use rotation first.

Two common body-delta action conventions are different:

**A. Separate Cartesian displacement and rotation vector**

\[
a=[\delta p,\delta\phi],\quad
\Delta T_b=\begin{bmatrix}\exp([\delta\phi]_\times)&\delta p\\0&1\end{bmatrix},
\quad T_{\rm cmd}=T_t\Delta T_b.
\]

Thus \(p_{\rm cmd}=p_t+R_t\delta p\), and \(R_{\rm cmd}=R_t\exp([\delta\phi]_\times)\).

**B. Full Lie-algebra exponential coordinates**

\[
\xi=[\rho,\phi],\qquad
\widehat\xi=\begin{bmatrix}[\phi]_\times&\rho\\0&0\end{bmatrix},
\qquad T_{\rm cmd}=T_t\exp(\widehat\xi).
\]

The resulting translation is \(J_l(\phi)\rho\), where \(J_l\) is the SO(3) left Jacobian. It is generally different from ρ.

**Preserve the existing action contract.** If the wrapper implements A, do not silently reinterpret its six numbers as B. For small increments, A may already be entirely adequate.

Right multiplication applies a body increment. Left multiplication applies a spatial increment; a pure rotation by left multiplication can also move the hand origin around the world origin. Verify the wrapper instead of inferring its semantics from a variable named `delta_pose`.

Also establish whether actions are displacements per step or velocities, whether scaling/clipping occurs, and which body frame each element of an action chunk uses. Later chunk actions might refer to the chunk's initial frame or successive updated frames.

Background: [rigid-motion exponential coordinates](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-3-exponential-coordinates-of-rigid-body-motion/).

### 4.5 Optional task geometry features

These are proposed small additions if the basic relative pose proves insufficient:

1. **Insertion axis:** express the port's insertion axis in the current hand frame. Together with signed depth and lateral offset in the port frame, this makes the mechanically relevant directions explicit.
2. **Corresponding keypoint errors:** define a few non-collinear points on the rigid connector, and compare their current positions with their desired mating positions, expressed in a common frame. This makes rotation errors appear as physically meaningful point displacements.
3. **Connector dimensions or clearance metadata:** useful if multiple connector types have different scales, provided those values are genuinely available.

Keypoints can supply metric scaling for orientation: the same angular error produces a larger displacement at a more distant corner. They also preserve keyed orientation if the points are chosen appropriately. An insertion axis alone cannot distinguish roll about that axis.

These features condition the policy; they do not prescribe its action or establish collision-free motion.

## 5. Compact architecture proposal

### 5.1 Main pathways

Keep the existing camera encoders and global scene information. Add or retain:

- A small relative-pose MLP, e.g. embedding width 128.
- A temporal force/proprioception encoder: a small causal transformer, GRU, or temporal convolution is sufficient to test the idea.
- Recent actions and achieved motion in the temporal state.
- Learned pose modulation in a few action-network blocks.
- A gate conditioned on multimodal context and pose metadata.

Do not crop away the surrounding scene as the only visual input. A local connector/port crop can supplement global vision if the existing architecture supports it.

### 5.2 Learned modulation

Let x be an action-network feature, p the pose embedding, and c the current multimodal context. One simple residual adaptation is:

\[
g=\sigma(G([c,p])),\quad (\gamma_\ell,\beta_\ell)=M_\ell(p),
\]

\[
x_\ell' = x_\ell + g\odot
\left[\gamma_\ell\odot\operatorname{LN}(x_\ell)+\beta_\ell\right].
\]

This uses FiLM-style feature scaling/shifting inside an identity-preserving residual adapter. Conventional adaptive LayerNorm inside the block is another implementation. Select one integration point and document it.

The gate should see vision/context, force history, and proprioception. Force onset alone does not identify insertion: the robot could have hit an obstacle. Distance alone does not identify a clear path. Pose confidence alone does not describe the phase of the task.

Begin by comparing an ungated modulation variant and a learned-gate variant. A gate is a hypothesis worth testing, not a mandatory improvement.

Conditioning references: [FiLM](https://arxiv.org/abs/1709.07871), [Diffusion Transformers](https://arxiv.org/abs/2212.09748). The combined pose/force gate above is our proposed adaptation.

### 5.3 Illustrative PyTorch module

This is an integration sketch, not code tested against the user's repository. It preserves the incoming feature at initialization. `context` must contain current/past information only. The pose feature dimension depends on available metadata.

```python
import torch
from torch import nn


class PoseConditioner(nn.Module):
    def __init__(self, pose_dim=12, context_dim=256, width=256,
                 pose_width=128, blocks=4):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_width), nn.SiLU(),
            nn.Linear(pose_width, pose_width), nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(context_dim + pose_width, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.norms = nn.ModuleList([
            nn.LayerNorm(width, elementwise_affine=False)
            for _ in range(blocks)
        ])
        self.modulations = nn.ModuleList([
            nn.Linear(pose_width, 2 * width) for _ in range(blocks)
        ])
        # Non-saturated initial gate; avoid disabling all learning paths.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # Identity adapter on the first forward pass.
        for layer in self.modulations:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(self, pose_features, context, valid):
        p = self.pose_encoder(pose_features)          # [B, P]
        g = torch.sigmoid(self.gate(
            torch.cat([context, p], dim=-1)))        # [B, 1]
        g = g * valid.to(g.dtype).reshape(-1, 1)
        return p, g

    def apply_block(self, block_id, x, p, g):
        # x: [B, N, D], p: [B, P], g: [B, 1]
        gamma, beta = self.modulations[block_id](p).chunk(2, dim=-1)
        delta = (gamma[:, None, :] * self.norms[block_id](x)
                 + beta[:, None, :])
        return x + g[:, None, :] * delta
```

Call `encode` once per observation and `apply_block` at selected action-network blocks. Because the modulation output starts at zero, its projection learns first; gradients subsequently reach the pose encoder and gate. Do not also initialize a multiplicative gate to a hard zero.

For the displayed dimensions, the module has approximately **0.31M parameters**, under 1% of a 40M policy. This excludes the force encoder and any new visual components. Actual runtime and memory depend on the backbone, token count, and integration location; profile the real model.

### 5.4 Force and history design

Use a causal history with timestamps. A tunable short window, for example 0.25–1 second, is a starting engineering choice rather than a literature requirement. Preserve high-frequency force information if available; a single downsampled force sample can miss important transients.

Useful inputs include:

- Compensated force and torque, with the compensation procedure documented.
- Actual end-effector motion, joint state, and any relevant velocity estimates.
- Previous commands and their execution times.
- Pose validity/confidence/age over time.

A blocked connector can receive a forward command while moving almost nowhere and producing rising force. The combination of command, achieved motion, and force history reveals this situation.

Do not infer sensor frequency or history alignment from nominal policy frequency. Inspect the logging and controller timestamps.

## 6. Training so the policy actually uses pose

### 6.1 First investigate why concatenation failed

Possible causes are hypotheses, not diagnosed facts:

- Wrong transform direction, TCP, rotation convention, or timestamp.
- Translation scales that hide millimeter-level errors.
- Training with ground-truth pose and deploying with noisy estimates.
- Insufficient target variation; the robot can imitate a fixed trajectory without using the pose.
- A dominant visual/proprioceptive shortcut and weak optimization through the pose branch.
- Nearly identical images paired with conflicting or misaligned pose/action labels.
- Missing grasp offset, insertion depth, or connector slip.
- Evaluation failures caused by control/contact behavior rather than target localization.

Architectural changes should follow basic geometric and dataset checks.

### 6.2 Imitation learning

Train the existing action objective with the new conditioning path. Retain all phases, including obstacle-rich approaches and recoveries, in the training distribution.

Where possible:

- Run the actual pose estimator on recorded observations, using a causal pipeline.
- Match deployment noise, bias, dropout, latency, and occasional estimator failures.
- Include varied port poses and initial hand configurations so the desired action depends on the goal.
- Include examples with different contact outcomes under similar geometric errors.
- Balance sampling so long, easy approaches do not overwhelm a small number of difficult insertion/recovery steps.
- Keep the final goal available during approach; the learned network decides its influence. Do not require a hand-coded insertion switch.

**Augmentation consistency matters.** Corrupting the observed pose while keeping the true scene and action target fixed simulates estimator noise. Physically changing the desired target while keeping the old action label generally creates incorrect supervision. A passive coordinate change must transform every affected geometric quantity consistently.

Modality dropout is optional. Pose dropout trains fallback behavior and can also make ignoring pose easier. Visual dropout can make obstacle reasoning worse. Use limited, realistic corruption and evaluate its effect rather than relying on dropout to force utilization.

### 6.3 Optional auxiliary objectives

Use auxiliary heads on features shared with the actor, and test whether they improve closed-loop behavior. Candidate targets:

1. **Next achieved relative pose or motion**, conditioned on the action actually executed.
2. **Future contact probability/type**, if meaningful labels can be obtained.
3. **Future wrench or wrench change**, aligned with the command interval.
4. **Achieved-motion versus commanded-motion discrepancy**, useful for blocked contact.

A generic objective is:

\[
L=L_{\rm BC}+\lambda_p L_{\rm motion}
+\lambda_c L_{\rm contact}+\lambda_f L_{\rm wrench}.
\]

Do not choose all objectives automatically. Start with one that addresses an observed failure and whose labels are reliable. Normalize targets and tune weights against action quality.

For geometry, use separate translation and rotation losses with explicit units/scales. One rotation discrepancy is:

\[
d_R(\hat R,R)=\|\log(\hat R^\top R)^\vee\|_2.
\]

Handle numerical behavior near zero and π with a tested rotation implementation. For small local action vectors, the existing well-scaled action loss may suffice.

Predicting the goal from a goal input is trivial. Predicting the next state while copying the current state can also score well at high frame rates. Compare auxiliary heads against persistence and action-independent baselines before interpreting low loss as useful understanding.

### 6.4 Avoid a misleading goal-progress loss

Do not require every demonstrated or predicted action to reduce Euclidean pose error. Successful navigation, contact search, and recovery can temporarily increase that error.

A terminal mating goal does not define a valid path. Any geometric regularizer should be evaluated for whether it suppresses retreat or drives the robot into nearby obstacles.

## 7. RL options

### 7.1 Fine-tune the conditioned actor

For a tractable stochastic continuous-action actor, use the existing compatible RL algorithm, such as PPO or a SAC-style method. A deterministic BC head needs a properly defined exploration distribution and likelihood where the algorithm requires one.

Continue to provide pose, force history, visual context, and achieved motion. Fine-tune the pose adapters/gate as well as the action head when those components are intended to learn from RL. A BC regularizer or behavioral-prior KL can help retain useful approach behavior; its weight must still allow recovery improvements.

For diffusion policies, do not apply a Gaussian log-probability formula to the final denoised action and call it PPO. Use a diffusion-policy RL method, a suitable policy distillation approach, or residual RL. See section 11.8.

### 7.2 Residual RL over a learned BC policy

An alternative is:

\[
a_t=a_t^{\rm BC}+S\,r_\theta(o_{\le t},E_b,a_t^{\rm BC}).
\]

S sets per-component correction scales in the existing small-delta action parameterization. Inputs to the residual should include the base action, force history, and current geometry.

- The base is a learned BC policy, consistent with the user's preference.
- Per-step residuals can react inside a longer base action chunk if the execution system supports fresh feedback.
- The residual must have enough authority to cancel an inappropriate forward action and initiate useful retreat.
- If large rerouting is required, a tightly bounded local residual may be insufficient; relax it or fine-tune the full actor.
- Addition of small action coordinates is a parameterization choice. For larger rotations, exact transform composition may be appropriate, but the residual's reference frame must be defined.
- Log and supply the **total executed command** to the environment model/critic, including scaling and clipping. Also retain the original sampled action where the RL likelihood calculation requires it.

This adapts the residual learning idea; adding force conditioning to the particular ResiP setup is our proposal.

### 7.3 Rewards and recovery coverage

Favor task success, meaningful insertion progress, and appropriate penalties for excessive contact loads or damaging behavior. Reward design must distinguish desired insertion contact from harmful collisions when the available state permits it.

Avoid a dominant distance penalty that makes useful retreats too expensive. Ground-truth simulation state can support rewards or an asymmetric critic without leaking unavailable state into the deployed actor.

Randomize perception errors, calibration offsets, contact/friction properties, and grasp variation within plausible ranges. Include failed alignments and recoveries. Force-aware IL can already learn from force-labeled demonstrations; RL is useful for improving behavior beyond their coverage, not a prerequisite for accepting force as input.

## 8. Dreamer 4 adaptation with pretrained tokenizer and dynamics

The following is a proposed robotics adaptation. The published algorithm details are summarized in section 11.18.

### 8.1 Put goal conditioning on the actor side

If the tokenizer/dynamics must remain frozen, attach a trainable actor-side adapter or small action decoder to the available frozen features. Feed current relative pose, force-history features, proprioception, and prior executed actions into this trainable component.

Do not silently change the action representation consumed by pretrained dynamics. The actor can use relative geometry internally while still emitting the original body-frame deltas.

A change in the desired goal should influence future physics through the selected actions. It should not make the dynamics predict that insertion happens simply because the actor has been told to insert. If an adapter changes shared world-model features, the dynamics are no longer functionally unchanged even if their original weights are frozen.

### 8.2 Future actor inputs must exist during imagination

Real execution can observe current pose and force. An imagined rollout needs their future counterparts, consistent with the imagined actions.

| Actor input | What an imagined rollout needs |
| --- | --- |
| Relative goal pose | Predicted achieved hand/connector state and the relevant target state |
| Force/torque history | Action-dependent predicted contact forces or an equivalent predictive state |
| Proprioception | Predicted robot motion/joint state, including imperfect execution |
| Pose validity/confidence/age | A model or approximation of the observation process consistent with visibility and time |
| Previous action | The command actually supplied to the dynamics |
| Visual context | World-model latent/observation features for the imagined state |

Possible approaches:

1. **Readouts from frozen latent features.** Fit pose, motion, and force heads if the latent already preserves the required information. Check held-out and action-dependent predictions.
2. **An added predictive state model.** Model contact/proprioception and their interaction with the visual state when those signals are missing from the frozen model.
3. **Extend or fine-tune dynamics.** Necessary if frozen representations/predictions cannot capture the contact behavior the policy needs to learn.
4. **Use environment-based RL for contact refinement.** Prefer this if the imagined contact model is inadequate for the intended improvements.

A readout cannot recover information that has been discarded or predict action effects missing from the dynamics. Merely feeding force to the actor during real execution does not make imagined force feedback valid.

Do not feed recorded future force measurements to trajectories generated under different actions. Do not update the hand pose by integrating commanded deltas as if every command were achieved during contact.

### 8.3 If using a residual actor

Both the base actor and the residual need valid observations along the imagined trajectory. Feed the combined executed body-delta command into the world model. If the base requires camera inputs, establish how it obtains suitable imagined visual features or decoded images; it cannot rely on unavailable real future frames.

### 8.4 RL objective distinction

Dreamer 4's default imagination update freezes the transformer and updates policy/value heads with a PMPO-based policy objective. It should not be described as necessarily backpropagating actor gradients through predicted dynamics, as in earlier differentiable-dynamics explanations. Its agent adaptation stage precedes imagination training. See the [original paper](https://arxiv.org/html/2509.24527v1).

A continuous robotic actor or trainable adapter is an adaptation; do not assume every action-distribution detail is identical to the paper's Minecraft implementation.

## 9. Evaluation: demonstrate useful pose utilization

### 9.1 Minimal sequence of ablations

Keep demonstrations, visual backbone, action semantics, training budget, and evaluation initial conditions comparable.

| Variant | Purpose |
| --- | --- |
| A0: current policy | Establish the existing failure modes |
| A1: correctly normalized relative pose concatenation | Separate geometry/scale fixes from fusion changes |
| A2: relative pose with repeated modulation | Measure the conditioning architecture's contribution |
| A3: A2 plus learned context gate | Determine whether gating helps or suppresses pose use |
| A4: best prior variant plus one auxiliary target | Test representation supervision only if needed |
| RL: full-actor fine-tuning or residual RL | Measure force-sensitive improvements and retained approach performance |

### 9.2 Behavioral checks

1. **Geometric sanity:** identity, known translation, known rotation, TCP/grasp offset, transform inverse, and multiplication direction.
2. **Passive coordinate consistency:** a common change of global reference should leave the computed body-relative error unchanged:

   \[
   (HT_E)^{-1}(HT_E^*)=T_E^{-1}T_E^*.
   \]

3. **Held-out target variation:** evaluate novel target configurations supported by a consistent scene, not merely changed pose numbers with stale images.
4. **Missing/noisy/stale pose:** measure robustness and whether the policy makes sensible use of the remaining modalities.
5. **Local pose intervention:** small plausible perturbations can reveal action sensitivity. Inconsistent or shuffled pose inputs measure dependence but do not prove correct use.
6. **Blocked approach:** place an obstacle on the direct route and measure successful avoidance.
7. **Alignment versus jam:** evaluate similar pose errors with different contact histories; the action should depend on the contact situation.
8. **Recovery authority:** check whether the actor can cancel forward motion, retreat, reorient, and resume.

### 9.3 Metrics and interpretation

Report completion rate, contact peaks, unintended collisions, time to completion, alignment quality, recovery success, and inference latency. Use multiple seeds/trials and uncertainty estimates appropriate to the evaluation size.

Inspect gate distributions by situation and pose quality, but do not interpret a large gate as proof of useful conditioning. Measure output sensitivity and closed-loop outcomes. A policy can use pose appropriately even when a gate is not easily interpretable as a named task phase.

Relative geometry is invariant to a passive global frame change. Physically rotating the whole task is different: gravity, cable anchoring, reachability, and camera observations may change. Do not infer full task SE(3) symmetry from the coordinate identity above.

## 10. Instructions for the receiving Codex agent

Use this as design context and adapt it to the actual repository. Recommended next steps:

1. Read applicable repository instructions and inspect the current policy, observation preprocessing, action wrapper, dataset labels, and RL training code.
2. Document the frame/units/timestamp contract, including rotation order, TCP, grasp transform, insertion goal, velocity versus displacement, action clipping, and action chunk semantics.
3. Check the pose estimator's output convention, calibration, latency, confidence availability, and behavior under occlusion.
4. Implement or locate tested transform utilities, then verify the relative-goal calculation on a few interpretable examples.
5. Add the smallest pose encoder/modulation change behind a configuration flag. Retain a relative-concatenation baseline.
6. Preserve causal force/action/motion history and existing visual coverage.
7. Run a focused IL comparison before adding multiple auxiliary heads, experts, or a new action representation.
8. Decide between full-actor RL and residual RL based on the actual action distribution, base competence, and needed recovery range.
9. If using imagination, establish that all future actor inputs and rewards have action-consistent predictions before interpreting RL gains.
10. Report observed results separately from assumptions and paper claims.

Do not assume file names, module paths, supported action formats, or unverified simulation rates from this document. The existing code and user instructions determine those details.

## 11. Paper catalog: summaries, relevance, and limits

The catalog covers the research works retained from the discussion. Entries 11.19–11.20 are additional foundational citations for the proposed feature-conditioning implementation. Summaries describe the linked versions; comparisons across papers are qualitative because tasks, data, robots, and metrics differ.

### 11.1 PoseInsert — Exploring Pose-Guided Imitation Learning for Robotic Precise Insertion

**Source:** [paper, v2](https://arxiv.org/html/2505.09424v2).

**Published method:** Represents the source object relative to the target, predicts future relative object-pose trajectories with a diffusion policy, and converts them to end-effector commands. Translation and rotation have separate small encoders. Rotation actions use a continuous 6D representation. An RGBD extension uses current/goal visual patches and pose-primary residual gated fusion: pose features form the main pathway and gated visual features provide corrections.

**Evidence and limits:** Evaluates six real-robot insertion tasks with 7–10 demonstrations per task. It does not evaluate force sensing or RL and uses passive compliance. SE(3) poses here do not imply an se(3) action parameterization or an SE(3)-equivariant network. Tight task clearance is not a measurement of pose-estimation accuracy. Total model parameter count was not established in this review.

**Use here:** Relative geometry and deliberate multimodal fusion are relevant. The user's force-aware learned gate and retained body-delta outputs are adaptations.

### 11.2 ResiP — From Imitation to Refinement: Residual RL for Precise Visual Assembly

**Source:** [paper](https://arxiv.org/html/2407.16677v1).

**Published method:** Trains a Gaussian MLP residual with PPO to correct actions from a frozen BC diffusion policy. The residual receives the proposed base action and can correct each timestep within an action chunk. The work also uses teacher/student distillation and visual domain randomization for RGB-based deployment.

**Use here:** A practical way to add corrective RL while preserving a competent learned approach policy. Condition the proposed residual on relative pose, force history, proprioception, and the base action.

**Limits:** Force conditioning is our extension. Results involving simulated state-based learning and visual distillation should not be described as direct proof of end-to-end force-aware robot training. Local residuals can struggle with large initial errors or behaviors requiring major rerouting.

### 11.3 Residual Reinforcement Learning for Robot Control

**Source:** [paper](https://arxiv.org/abs/1812.03201).

**Published method:** Combines a conventional controller with a learned RL residual, allowing learning to compensate for effects the controller does not handle well.

**Use here:** Establishes the general residual-learning idea. For this user's preference, choose a learned BC base rather than a fixed geometric servo.

**Limits:** The choice of base policy and correction authority determines which behaviors remain reachable. A small correction cannot overcome an arbitrarily bad base command. The paper does not by itself specify the recommended multimodal conditioning architecture.

### 11.4 FORGE — Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty

**Source:** [paper](https://arxiv.org/html/2408.04587v1).

**Published method:** Uses force feedback, pose/dynamics uncertainty, force-related training penalties, and success prediction for assembly. Training uses recurrent PPO with an asymmetric actor-critic. Observations/actions are defined relative to the fixed part; the evaluated control formulation assumes upright parts and uses four pose dimensions.

**Use here:** Supports learning contact-aware behavior from noisy geometry and temporal force information. Its randomization and recurrent-policy ideas are relevant to IL-to-RL refinement.

**Limits:** Four-dimensional upright assembly is not full six-dimensional wire insertion. Its target-relative output conversion is not required to borrow the training ideas. The user should retain the existing action interface for the first implementation.

### 11.5 FoAR — Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation

**Source:** [paper](https://arxiv.org/html/2411.15753v1).

**Published method:** Integrates visual/3D scene features and temporal force/torque features for imitation learning. A future-contact predictor helps regulate modality fusion. The temporal encoder uses high-frequency force history. Deployment also includes threshold-based action corrections and contact-dependent execution logic.

**Use here:** Borrow the temporal force representation and the idea of predicting upcoming contact as an auxiliary signal.

**Limits:** The complete method contains deployment rules, which do not match the user's desired fully learned corrective behavior. Its threshold-based action correction should not be silently copied. Contact prediction also cannot, by itself, distinguish intended insertion from an obstacle collision. RL compatibility of our proposed adaptation remains to be evaluated.

### 11.6 ForceVLA — Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation

**Source:** [paper, v3](https://arxiv.org/html/2505.22159v3).

**Published method:** Adds force-aware multimodal fusion to a VLA action pathway using a mixture of experts. Its reported fusion uses four MLP experts with top-1 routing and residual integration. The evaluation includes contact-rich manipulation tasks such as plug and USB insertion.

**Use here:** Inspiration for a compact, learned routing mechanism when different sensory cues matter in different situations. A small shared-encoder feature MoE can be tried without adopting a full VLA.

**Limits:** The published setup is not a demonstrated IL-to-RL solution for the user's small policy. Learned experts need not correspond cleanly to named phases. Avoid averaging opposing action controllers merely to reproduce an MoE label.

### 11.7 Diffusion Policy — Visuomotor Policy Learning via Action Diffusion

**Source:** [RSS paper](https://www.roboticsproceedings.org/rss19/p026.pdf).

**Published method:** Learns conditional action distributions through denoising and executes predicted action sequences with receding-horizon control. The paper studies convolutional and transformer formulations, including feature modulation and observation conditioning through attention.

**Use here:** Shows useful places to inject a pose embedding repeatedly into an action-generation network. Relative pose can condition a diffusion model alongside visual and force-history features.

**Limits:** A diffusion action head is optional for this project. Its iterative action-generation process complicates some conventional RL likelihood calculations and can increase inference work. Preserve the current competent head unless multimodality or other observed failures motivate changing it.

### 11.8 DPPO — Diffusion Policy Policy Optimization

**Source:** [paper](https://arxiv.org/html/2409.00588v3).

**Published method:** Fine-tunes diffusion policies with policy-gradient RL by treating the denoising process as part of the decision process and using its conditional transition probabilities.

**Use here:** A direct RL option if the current action head is a diffusion model and improving the whole action distribution is important.

**Limits:** This is more involved than applying PPO to an ordinary Gaussian actor. The probability of a final generated action is not obtained by pretending it is the mean of a single Gaussian. Compare implementation cost and runtime against a force-aware residual actor.

### 11.9 Making Sense of Vision and Touch — Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks

**Source:** [paper](https://arxiv.org/abs/1810.10191).

**Published method:** Learns multimodal representations with self-supervised objectives and uses them for contact-rich control, including peg insertion.

**Use here:** Motivates shared features trained to capture contact and motion consequences rather than relying only on action imitation. Our proposed action-conditioned next-pose/wrench heads are an adaptation of this broad representation-learning direction.

**Limits:** Auxiliary accuracy alone does not establish useful control. Our exact objectives, actor architecture, and pose gate are not claims about the paper's implementation.

### 11.10 HIL-SERL — Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning

**Source:** [paper](https://arxiv.org/html/2410.21845v1).

**Published method:** Combines demonstrations, online RL, and human corrections/interventions to learn demanding manipulation behavior.

**Use here:** A relevant training workflow if targeted corrective experience can be collected. Interventions can supply recovery states missing from successful demonstrations and make online refinement more productive.

**Limits:** It does not establish that every task requires or uses force sensing, and it is not a geometric pose-conditioning method. The availability of an intervention interface and suitable online environment determines whether this workflow is practical here.

### 11.11 3D Diffuser Actor — Policy Diffusion with 3D Scene Representations

**Source:** [paper](https://arxiv.org/html/2402.10885v2).

**Published method:** Uses 3D scene representations and relative-position attention to connect scene, proprioceptive, and action features. The spatial attention uses 3D rotary positional information.

**Use here:** A stronger spatial bias when reliable depth/3D geometry is available: the target and candidate actions can interact with spatially located scene features.

**Limits:** This requires more representation work than adding a pose MLP. Translation-relative attention does not automatically confer full SE(3) equivariance on the entire policy. Three RGB camera streams do not automatically provide calibrated, accurate 3D scene tokens.

### 11.12 ReKep — Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

**Source:** [paper](https://arxiv.org/html/2409.01652v1).

**Published method:** Describes tasks with relational keypoint constraints, including goal and path constraints, and uses optimization to generate motion.

**Use here:** Helpful for distinguishing the final mating relation from requirements along the approach. A few keypoint error features can inspire compact policy inputs.

**Limits:** Adopting its full constraint/planning framework is not the recommended first change. The reported experiments disable collision checking for deformable-object tasks, so it should not be cited as a ready-made solution to cable collision avoidance.

### 11.13 On the Continuity of Rotation Representations in Neural Networks

**Source:** [paper](https://arxiv.org/abs/1812.07035).

**Published result:** Analyzes continuity issues in common low-dimensional rotation representations and develops continuous higher-dimensional representations, including a widely used 6D construction.

**Use here:** Represent relative orientation using two rotation-matrix columns. When predicting this representation, orthonormalize the decoded vectors with appropriate numerical handling.

**Limits:** This is a representation result, not a robotics controller or obstacle-avoidance method. It does not increase the physical degrees of freedom, and it does not make an arbitrary network rotation-equivariant.

### 11.14 A micro Lie theory for state estimation in robotics

**Source:** [paper](https://arxiv.org/html/1812.01537v9).

**Published contribution:** A practical treatment of Lie groups, local perturbations, exponential/logarithm maps, Jacobians, and uncertainty for robotics estimation.

**Use here:** A reference for computing relative transforms and choosing consistent perturbation/frame conventions. Useful when implementing pose noise or uncertainty propagation.

**Limits:** A mathematically valid pose parameterization does not determine a useful learned policy. In particular, se(3) translation coordinates should not be confused with Cartesian displacement under finite rotation.

### 11.15 SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space

**Source:** [paper](https://arxiv.org/abs/2507.01723).

**Published direction:** Builds SE(3)-equivariant policy structure using a spherical Fourier representation, aiming to exploit geometric symmetry in diffusion-based control.

**Use here:** Relevant if generalization over substantial 3D pose variation remains a dominant bottleneck after simpler relative conditioning.

**Limits:** It is a substantially different architecture, not a property obtained by passing six pose numbers into a transformer. The task's actual symmetry assumptions and the treatment of force, gravity, obstacles, and cable anchoring require careful evaluation.

### 11.16 Equivariant Diffusion Policy

**Source:** [paper](https://arxiv.org/abs/2407.01812).

**Published method:** Exploits SO(2) symmetry for diffusion-policy learning, including full six-DoF control outputs, to improve data efficiency and generalization.

**Use here:** Illustrates that the appropriate symmetry group may be smaller than SE(3), for example rotations around an axis consistent with the task setup.

**Limits:** Six-DoF outputs do not imply SE(3)-equivariance. SO(2) symmetry must still be valid for the chosen observation/action representation and physical task.

### 11.17 EquiBot — SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning

**Source:** [paper](https://arxiv.org/abs/2407.01479).

**Published direction:** Uses similarity-transform symmetry, covering translation, rotation, and scale, to improve policy generalization and data efficiency.

**Use here:** A reference for stronger geometric architectural bias if large variation across object placements and scales matters.

**Limits:** Geometric scaling is not automatically a symmetry of contact dynamics, connector tolerances, friction, or force limits. Applying it to precision insertion requires more than scaling visual coordinates.

### 11.18 Dreamer 4 — Training Agents Inside of Scalable World Models

**Source:** [paper](https://arxiv.org/html/2509.24527v1).

**Published method:** Trains a causal tokenizer and action-conditioned dynamics with shortcut forcing, adapts the model with agent/task tokens and policy/reward heads, then learns behavior through imagined rollouts. Agent tokens attend to world-model modalities, while those modalities cannot attend back to agent tokens. The showcased task is Minecraft.

**Use here:** Provides the architectural context for attaching goal-conditioned policy components to a pretrained predictive model. See section 8 for the proposed robotics adaptation and its sensor-prediction requirements.

**Limits:** The paper is not evidence that frozen video dynamics already model precise insertion forces. Continuous robotic action heads and explicit force/pose adapters need their own validation.

### 11.19 FiLM — Visual Reasoning with a General Conditioning Layer

**Source:** [paper](https://arxiv.org/abs/1709.07871).

**Published method:** Conditioning information generates feature-wise affine transformations, scaling and shifting intermediate activations.

**Use here:** A lightweight mechanism for making the relative pose influence several action-network layers. The pose embedding can generate the modulation parameters.

**Limits:** The paper's original visual-reasoning results do not demonstrate this insertion application. The learned force/context gate and IL-to-RL training setup are our proposed extensions.

### 11.20 Scalable Diffusion Models with Transformers — DiT

**Source:** [paper](https://arxiv.org/abs/2212.09748).

**Published method:** Studies transformer-based diffusion architectures and conditioning mechanisms, including adaptive LayerNorm and identity-friendly initialization.

**Use here:** An implementation reference for injecting pose conditions into transformer blocks with modest extra parameters.

**Limits:** Image-generation performance does not establish robotic control performance. Borrow the conditioning mechanism without assuming a large diffusion transformer is necessary for this task.

## 12. Robotics refresher and primary references

These background points explain where the learned policy sits in the robot stack.

### 12.1 FK, IK, and Jacobians

**Forward kinematics (FK)** maps joint configuration q to the end-effector pose \(T_E(q)\). It supplies the current robot-side geometry needed for the relative goal.

**Inverse kinematics (IK)** finds a joint configuration that achieves a requested end-effector pose. It can have multiple solutions or no solution. IK does not establish a collision-free path.

A body Jacobian relates joint velocity to hand velocity:

\[
V_b=J_b(q)\dot q.
\]

A local numerical IK update can use \(e_b=\log(T(q)^{-1}T^*)^\vee\) and a damped pseudoinverse. Translation and rotation require appropriate scaling. Near singularities, some requested Cartesian motions are difficult to achieve.

References: [body Jacobian](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-1-2-body-jacobian/), [numerical inverse kinematics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-2-of-2/).

### 12.2 Twists and frame changes

With translation-first ordering, a twist is \(V=[v,\omega]\). Its frame conversion uses:

\[
\operatorname{Ad}_T=
\begin{bmatrix}R&[p]_\times R\\0&R\end{bmatrix}.
\]

The reference origin matters because angular motion produces linear velocity at an offset point. A body twist satisfies \(\widehat V_b=T^{-1}\dot T\). Under a constant body twist, \(T(t+\Delta t)=T(t)\exp(\widehat V_b\Delta t)\).

Reference: [Modern Robotics, twists](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-2-twists-part-2-of-2/). Check its ordering conventions against the translation-first notation used here.

### 12.3 Wrenches and the sensor reference point

A wrench is force and torque, here ordered \(w=[f,\tau]\). Transforming both its coordinate frame and reference point gives:

\[
f_A=R_{AB}f_B,\qquad
\tau_A=R_{AB}\tau_B+p_{AB}\times(R_{AB}f_B).
\]

Rotating torque alone does not shift its reference point. A connector contact can produce substantial wrist torque due to the lever arm. With consistent conventions, power is \(w^\top V\), and generalized joint forces are related by \(J^\top w\).

Establish whether the logged signal is raw, bias-compensated, gravity-compensated, filtered, or an external-contact estimate before training.

Reference: [Modern Robotics, wrenches](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/).

### 12.4 Commands, controllers, and achieved motion

The policy issues commands to an existing low-level controller. Compliance, contact, robot dynamics, clipping, and joint limits determine what motion actually occurs.

The insertion policy therefore needs feedback about both commanded and achieved motion. A correct desired pose can coexist with a blocked connector and excessive contact load. Geometric conditioning helps specify the goal; multimodal feedback and training determine the response.

**First experiment to implement:** verified body-relative goal features → small pose MLP → repeated action-feature modulation; compare with relative concatenation; add a context gate as an ablation; retain force/action/motion history; then select full-actor or residual RL based on measured failures.
