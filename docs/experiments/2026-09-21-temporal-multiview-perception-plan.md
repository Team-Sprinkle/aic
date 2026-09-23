# Causal temporal and multi-view perception for cable insertion

**Status:** design and reading guide. No temporal model has been trained. The
current single-frame MobileNetV3-small model remains frozen evidence. Policy
training and RL remain gated.

## What problem this is intended to solve

The current observation-only model predicts the plug point and four port-opening
corners from three synchronized RGB cameras. On six fresh episode-grouped
development configurations it reached 0.278 mm median and 0.670 mm p95 lateral
error. Most predictions are close to the 0.5 mm insertion corridor, but a small
tail of frames is worse. The saved images show likely causes: the cable or
gripper crosses the interaction region, the opening is small or oblique in one
view, and a point estimate discards ambiguity in a heatmap.

The proposed model estimates the **current** pose from a short causal history.
It does not predict future RGB and is not the parked generative world model.
At decision time it may use only images up to time \(t\), robot observations,
camera calibration, and its previous hidden state:

$$
\hat{T}^{\text{port}}_{\text{plug},t},\;\hat{\Sigma}_t,
\;\hat{o}_{v,k,t}
= f_\theta\left(I_{1:3,t-K+1:t},\;s_{t-K+1:t},\;\mathcal{C}_{1:3}\right).
$$

Here \(K\) is initially 4 or 6 decisions, \(\mathcal{C}\) contains camera
intrinsics and extrinsics, \(\hat{o}\) is predicted visibility/occlusion, and
\(\hat{\Sigma}\) is calibrated pose uncertainty.

## Reading list

### Read these first

1. **Li and Schoellig, “Multi-View Keypoints for Reliable 6D Object Pose
   Estimation” (2023).** [Paper](https://arxiv.org/abs/2303.16833). This is the
   closest robotics reference. It combines image keypoint heatmaps from known
   camera poses into a probability distribution in 3D and uses keypoint and
   alignment confidence to reject false positives. Its roughly 0.5 mm reported
   average error shows that multi-view keypoints can reach the relevant scale,
   although its rigid-object bin-picking setting and metric do not establish
   our 0.5 mm tail requirement.

2. **Iskakov et al., “Learnable Triangulation of Human Pose” (2019).**
   [Paper](https://arxiv.org/abs/1905.05754). It gives two useful constructions:
   differentiable algebraic triangulation with learned view confidences, and
   volumetric aggregation of features before a 3D heatmap. We use the first as
   the light baseline and treat the second as a possible accuracy ablation
   because a dense 3D volume may cost too much latency.

3. **He et al., “Epipolar Transformers” (2020).**
   [Paper](https://arxiv.org/abs/2005.04551). Instead of detecting independently
   in each camera and combining only the final points, it transfers features
   between views along epipolar lines. This supplies the geometric inductive
   bias needed when one camera is occluded. Our version must be much smaller and
   operate on native interaction crops.

4. **Doersch et al., “TAPIR: Tracking Any Point with per-frame Initialization
   and Temporal Refinement” (2023).**
   [Paper](https://arxiv.org/abs/2306.08637). TAPIR separates independent
   per-frame matching from temporal refinement and explicitly predicts
   occlusion. The useful idea is recovery after temporary disappearance rather
   than blindly smoothing the previous position.

5. **Karaev et al., “CoTracker3” (2024).**
   [Paper](https://arxiv.org/abs/2410.11831). CoTracker jointly tracks points and
   supports causal online inference, with training that mixes synthetic labels
   and pseudo-labelled real video. It motivates joint temporal reasoning and
   occlusion supervision. We do not plan to deploy its full general-purpose
   tracker initially because the task has only six named landmarks and a strict
   latency budget.

### Robotics and cable context

6. **Wang et al., “End-to-end Reinforcement Learning of Robotic Manipulation
   with Robust Keypoints Representation” (2022).**
   [Paper](https://arxiv.org/abs/2202.06027). It trains compact keypoints with
   simulator depth/segmentation supervision, equivariance constraints, and then
   feeds them to an actor-critic. This supports the staged pose-first design,
   while our controller remains supervised until its insertion gate passes.

7. **Weng et al., “Interactive Perception for Deformable Object Manipulation”
   (2024).** [Paper](https://arxiv.org/abs/2403.05177). It treats occlusion of a
   deformable object as an active perception problem and coordinates camera and
   manipulator motion. Our cameras are fixed and we cannot assume an exploratory
   motion is safe near a port, but uncertainty-triggered retreat or view-clearing
   motion may become a later recovery behavior.

8. **Kernbach et al., “Behavioral Cloning for Robotic Connector Assembly: An
   Empirical Study” (2026).** [Paper](https://arxiv.org/abs/2602.22100). This is
   directly relevant to connectors on deformable cables and reports combining
   fixed-camera vision with force/torque information. Its connector geometry,
   robot, demonstrations, and success criterion differ, so it informs the later
   force-conditioned controller rather than proving that our perception will
   work.

### Useful optional references

- **Sárándi et al., “Synthetic Occlusion Augmentation with Volumetric
  Heatmaps” (2018).** [Paper](https://arxiv.org/abs/1809.04987). It demonstrates
  that explicitly pasted occluders can materially improve pose estimation. We
  will use cable-shaped and gripper-shaped masks as a controlled ablation, while
  retaining physically rendered occlusions as the primary evidence.
- **Ma et al., “TransFusion” (2021).**
  [Paper](https://arxiv.org/abs/2110.09554). It uses transformer cross-view
  fusion with an epipolar field. It is an alternative implementation reference
  if the simpler weighted triangulator plateaus.
- **Bharadhwaj et al., “Track2Act” (2024).**
  [Paper](https://arxiv.org/abs/2405.01527). It converts predicted point tracks
  into rigid transforms and then uses a closed-loop residual policy. This is
  relevant to the later control interface, but it predicts goal-conditioned
  future tracks from broad video pretraining rather than estimating a
  submillimetre current connector pose.
- **Mazza et al., “Active Cross-Modal Visuo-Tactile Perception of Deformable
  Linear Objects” (2026).** [Paper](https://arxiv.org/abs/2601.13979). It
  demonstrates why touch can recover cable shape under severe visual
  occlusion. We do not currently have its tactile hardware or need full cable
  reconstruction, so force/contact is initially used for phase and blocked-state
  estimation rather than as a spatial image replacement.

## Gaps between the papers and this task

| Gap | Why it matters here | How the experiment handles it |
| --- | --- | --- |
| Most multi-view papers estimate human joints or larger rigid objects. | Their errors are commonly measured in centimetres or average millimetres; insertion depends on a 0.5 mm tail. | Train only task landmarks, report median and p95 in the port frame, and retain the fixed 0.25/0.5 mm gate. |
| The closest 6D object paper assumes rigid, textured object geometry and often point clouds. | The port is small and the cable/gripper can hide it; the cable is deformable. | Keep full-view context, use native RGB crops, supervise explicit visibility, and stratify evaluation by the actual occluder. |
| Generic trackers follow a queried surface point. | The plug tip and opening center may be textureless or fully invisible, and their semantic identity matters. | Detect named landmarks every frame and use temporal memory for refinement/recovery. Never propagate a point indefinitely without a new measurement. |
| Human-pose models exploit a strong skeleton prior. | Plug and port are separate rigid bodies; cable shape does not define a fixed skeleton. | Use only valid geometry: four coplanar opening corners, calibrated rays, port rigidity within an episode, and robot motion. Do not impose a fixed cable shape. |
| Many methods are offline or use future frames. | The robot must respond causally within 300 ms. | Use a causal recurrent/attention mask, fixed \(K\), cached history features, and report complete p50/p95/p99 online latency. |
| Published confidence scores are often uncalibrated heatmap peaks. | Our prior heatmap-variance camera rejection failed and retained all views. | Train visibility and covariance against held-out residuals; test calibration coverage and risk-versus-coverage instead of assuming a wide heatmap means a bad view. |
| Papers often split individual frames randomly. | Adjacent frames and repeated reset configurations leak nearly identical geometry. | Split complete episodes, reset configurations, and cable seeds. Keep the four existing final configurations sealed. |
| Simulator occlusion can be cleaner than Gazebo or reality. | A model may learn simulator-specific cable color, aliasing, or segmentation edges. | Vary cable pose, lighting, textures and partial visibility; reserve a later Gazebo development transfer after Isaac autonomous insertion passes. |
| Our Isaac cable may not span the full physical cable-state distribution. | More frames of nearly identical cable curvature will not teach recovery. | Audit whether cable seeds actually change projected occlusion. If not, use controlled rendered occluders for the perception ablation and collect targeted Gazebo sequences before claiming cable generalization. |

## Proposed model

### 1. Observation-only crop selection

Each camera first runs the existing full-view RGB locator. It predicts an
interaction center \(c_{v,t}\), and the system extracts a native 160x160 or
192x192 crop before resizing. Simulator geometry may supervise the locator but
cannot choose a deployment crop.

### 2. Shared spatial encoder

The current truncated ImageNet MobileNetV3-small remains the starting encoder.
For camera \(v\) and time \(t\):

$$
F_{v,t}=E_\theta\!\left(\operatorname{crop}(I_{v,t},c_{v,t})\right).
$$

It produces plug/opening heatmaps, a visibility logit for each landmark, and a
small feature grid. Full-image features remain available for robot and task
context.

### 3. Geometry-guided multi-view fusion

A feature at pixel \(u\) in one calibrated camera can correspond only to the
epipolar line \(\ell_{v\rightarrow w}(u)\) in another camera. Attention is
restricted to samples along that line instead of comparing every pixel:

$$
\widetilde F_{v,t}(u)=F_{v,t}(u)+
\sum_{w\ne v}\sum_{u'\in\ell_{v\rightarrow w}(u)}
\alpha_{v,w,t}(u,u')F_{w,t}(u').
$$

This allows a clear camera to repair an ambiguous or occluded one. Camera
geometry is an observation/calibration input, not privileged object geometry.

The light first version can instead predict a learned nonnegative reliability
\(w_{v,k,t}\) and use differentiable weighted triangulation. For camera center
\(C_v\) and predicted world ray \(d_v\), the 3D point is

$$
\hat X_{k,t}=\arg\min_X
\sum_v w_{v,k,t}\left\|
\left(I-d_{v,k,t}d_{v,k,t}^{\mathsf T}\right)(X-C_v)
\right\|_2^2.
$$

Unlike the failed hand-written view rejection, the weights receive a direct 3D
loss and visibility supervision.

### 4. Causal temporal fusion

For each named landmark, a small GRU or two-layer causal transformer receives
the last \(K\) fused features, predicted visibility, robot motion, and camera
motion. It outputs the current state:

$$
h_t=\operatorname{GRU}(z_t,h_{t-1}),\qquad
(\hat\xi_t,\hat L_t)=g(h_t),\qquad
\hat\Sigma_t=\hat L_t\hat L_t^{\mathsf T}.
$$

The six-vector \(\hat\xi_t=[\hat p_t,\hat\phi_t]\) is plug pose relative to
the port; \(\hat\phi\) is a rotation vector. The model is trained to hold the
stationary port estimate through a temporary obstruction while updating the
moving plug. A recurrent reset occurs at every episode/reset boundary.

### 5. Training losses

The initial objective is supervised perception:

$$
\mathcal L =
\lambda_{2D}\mathcal L_{\text{heatmap}}+
\lambda_{3D}\operatorname{Huber}(\hat p-p)+
\lambda_R d_{SO(3)}(\hat R,R)+
\lambda_o\operatorname{BCE}(\hat o,o)+
\lambda_{rep}\mathcal L_{\text{reprojection}}+
\lambda_{nll}\mathcal L_{\text{Gaussian NLL}}+
\lambda_{temp}\mathcal L_{\text{temporal}}.
$$

The temporal term penalizes implausible changes after accounting for measured
robot motion. Port rigidity can be enforced strongly within an episode; plug
motion cannot be forced to zero. Visibility targets should come from rendered
depth tests, not only semantic masks: a projected landmark is visible when its
expected depth agrees with the camera depth buffer within a declared tolerance.

All simulator pose, depth and visibility information is training/evaluation
supervision only. It is absent from autonomous inference.

## Data needed

The collection should cross these factors rather than simply add neighboring
frames:

- signed lateral offsets around and beyond the 0.5 mm corridor;
- axial distance, small orientation error and approach phase;
- cable curvature and cable crossing position;
- clear, partially occluded and fully occluded port corners per camera;
- gripper occlusion and oblique views;
- stationary, moving and contact/blocked plug states;
- lighting, texture and modest camera-calibration perturbations.

Every frame should record the predicted input fields plus training-only
visibility, occluder identity and exact pose. The manifest must group by full
episode, reset configuration and cable seed. An occlusion-coverage table should
be produced before training so that “more data” cannot mean repeated clear
views.

## Bounded experimental sequence

1. **Dataset audit:** measure landmark visibility by camera, occluder and phase;
   verify that cable seeds produce different image-space occlusions.
2. **Frozen baseline:** preserve the current single-frame MobileNet result.
3. **Temporal-only ablation:** shared encoder plus causal GRU, independent views,
   same training configurations and update budget.
4. **Multi-view-only ablation:** learned weighted triangulation or one epipolar
   fusion block, one frame only.
5. **Combined model:** enable both only if either controlled ablation improves
   calibration p95.
6. **Fresh development:** open new reset/cable configurations once, only after
   calibration indicates a credible path to the gate.
7. **Controller comparison:** only after perception passes, compare the same
   supervised head with and without predicted pose/phase/uncertainty.

Report overall and occlusion-stratified translation, axial, lateral and
orientation errors; direction accuracy; visibility precision/recall; covariance
coverage; and complete live latency. Promotion still requires near-port lateral
median \(\le 0.25\) mm, p95 \(\le 0.5\) mm, direction accuracy \(\ge 90\%\),
and inference p95 below 300 ms.

## What would falsify this direction

Stop or redesign if any of the following occurs:

- temporal/multi-view calibration p95 does not beat the frozen single-frame
  model under the same data and update budget;
- improvement exists only on random frame splits;
- uncertainty cannot identify the error tail better than a constant baseline;
- cable variation does not change the rendered occlusion distribution;
- latency exceeds 300 ms after caching and model-size reduction;
- pose improves but the later autonomous controller does not improve lateral
  alignment.

In those cases, targeted sensing or contact-guided recovery is more defensible
than increasing model size.

## How to read the prediction-only visual audit

The accompanying review images deliberately do not read simulator plug/port
pose labels. They use RGB, camera calibration, frozen observed features and
trained model outputs. Magenta marks the predicted plug, cyan marks the four
predicted opening corners, yellow marks their predicted center, and the lime
arrow is the predicted image-plane correction. White circles are the
reprojection of the 3D points obtained from the three predictions; red, green
and blue arrows are the predicted opening axes.

This can reveal obviously misplaced points, inconsistent views and implausible
orientation. It cannot prove submillimetre accuracy because a consistent model
can be consistently wrong. The numerical 0.278/0.670 mm claim still comes from
the separately preserved held-out label comparison. The visual audit is an
independent check that the reported model is looking at the intended physical
features rather than exploiting a hidden evaluation label.

The completed audit is available as a
[native-crop sheet](../../outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/opening_landmarks_v1/prediction_only_review/prediction_only_native_crops.png),
[full-view sheet](../../outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/opening_landmarks_v1/prediction_only_review/prediction_only_full_views.png),
and [audit report](../../outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/opening_landmarks_v1/prediction_only_review/README.md).
Manual inspection found the predicted cyan opening boundary on the visible SFP
cage across clear and cable-occluded views, with the predicted plug/correction
remaining in the interaction region. Across all 301 observations, cross-view
reprojection RMS was 0.807 pixels median and 1.309 pixels p95. The predicted
stationary opening still jittered by 0.369 mm median / 1.316 mm p95 relative to
its per-episode median, and its normal jittered by 2.98 / 7.97 degrees. These
prediction-only checks show plausible attention and good cross-view consistency
while exposing temporal instability. They do not replace labelled accuracy.

## Executed bounded continuation

The replay audit found 1,355 usable RGB decisions, but instance masks were
intentionally retained only at decision 1 and every 100 decisions. The 24
mask-labelled decisions span 10 actual reset sequences. In that sparse sample,
the cable occupied more than 15% of the projected opening region in 21/24
center-camera frames, while it did not cross the opening in either side view.
This is useful evidence for multiview fusion, but the reset manifest varies TCP
offsets and has no independent cable seed or cable-shape variable. It cannot
support a cable-generalization claim. The [coverage report and montage](../../outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/opening_landmarks_v1/occlusion_audit/occlusion_coverage.json)
record the exact subset and limitations.

The audit also exposed a sequence bookkeeping bug: `global_episode_index` is
null in this replay. Temporal code now derives reset boundaries from episode-id
changes and terminal transitions. This prevents history from leaking between
separate resets that reuse the same configuration name.

The ImageNet MobileNetV3 spatial model was frozen. Two three-member residual
ensembles were trained on 16 complete reset configurations and selected on four
held-out calibration configurations:

| Model | Parameters/member | Calibration near-port lateral median / p95 | Direction |
| --- | ---: | ---: | ---: |
| Frozen spatial baseline | — | 0.288 / 0.528 mm | 90.8% |
| Learned three-view current frame | 21,447 | 0.141 / 0.421 mm | 100% |
| Six-step causal three-view history | 43,547 | 0.133 / 0.300 mm | 100% |

Both learned heads cleared the calibration thresholds, so six new reset
positions were generated once. They exclude all earlier development starts and
the four reserved final configurations. The new 1,200-step collection contains
301 causal decisions and seven reset sequences across six configurations.

| Model | Fresh development near-port lateral median / p95 | Axial median / p95 | Direction |
| --- | ---: | ---: | ---: |
| Frozen spatial baseline | 0.412 / 0.769 mm | 0.564 / 1.456 mm | 96.8% |
| Learned three-view current frame | **0.307 / 0.573 mm** | **0.304 / 0.796 mm** | 99.4% |
| Six-step causal three-view history | 0.289 / 0.667 mm | 0.376 / 1.225 mm | 99.4% |

The current-frame head materially improves the frozen baseline, but still
misses the 0.25/0.5 mm gate. The temporal head's calibration advantage did not
generalize and its fresh p95 is worse than the current-frame head. Treat that
as calibration overfit, not evidence for promotion. Full three-view perception,
triangulation, and the temporal ensemble measured 6.60 ms p95; pairing it with
the existing frozen policy trunk measured 6.82 ms p95. Latency passes easily.
The [calibration-versus-development plot](../../outputs/experiments/2026-09-21_temporal_multiview_perception/calibration_vs_development.png)
shows the failed transfer against both fixed thresholds.

No conditioned controller, final-scene evaluation, or RL was run. Preserve the
current-frame head as the best diagnostic. Before another temporal model,
collect episode groups with independently varied cable shapes and occlusion
locations, retain a depth buffer or per-landmark visibility label, and include
longer temporary occlusions. The next comparison should keep the same reset
grouping and test learned visibility-weighted fusion against this current-frame
head. Machine-readable results, checkpoints, commands, manifests, and hashes
are under
[`outputs/experiments/2026-09-21_temporal_multiview_perception/`](../../outputs/experiments/2026-09-21_temporal_multiview_perception/).

### Concrete cable-reset requirement

The unified Isaac articulation exposes 46 joints: six arm joints and 40 cable
joints named `joint_0_1:1` through `joint_19_20:2`. The current reset event
selects only the six arm joints. Cable states can evolve during a rollout, but
there is no controlled cable seed or independent cable-shape draw at reset.

Do not apply unconstrained noise to all 40 cable joints. That can move the plug
tip, create contact, or invalidate the requested near-port start, mixing reset
errors with perception errors. The next collector should sample a small bank of
seeded, bounded cable-shape templates; settle physics; reject collision/high
force states; measure the plug tip; and then solve the arm reset so plug axial,
lateral, and orientation bins stay matched. Each manifest must retain the seed,
sampled and settled cable joints, final plug pose, occluder identity, and
depth-derived visibility for each landmark. Split entire cable seeds/templates
across fit, calibration, and development. The
[machine-readable feasibility audit](../../outputs/experiments/2026-09-21_temporal_multiview_perception/cable_reset_feasibility.json)
records the joint inventory and acceptance contract.
