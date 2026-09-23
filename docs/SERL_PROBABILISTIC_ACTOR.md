# Probabilistic Trajectory Actor for SERL

**Status:** Implemented and evaluated; final tight trust-region run retained BC at 3/8 but did not improve it
**Last updated:** 2026-09-23

This note describes how to adapt the selected deterministic four-waypoint BC
transformer to SERL/RLPD-style online reinforcement learning. It also explains
why adding Gaussian noise around the BC prediction is insufficient and why a
small mixture distribution is a better match for recovery.

See [SERL recovery strategy](SERL_RECOVERY_STRATEGY.md) for failure rewards,
replay, backtracking, and the two exploration branches.

## Current evidence boundary

The selected actor and frozen perception have been trained and evaluated in
Isaac SFP-to-NIC near-port scenes. They have not established SC-to-SC
perception, transport, multi-card cable routing, or full-depth official Gazebo
insertion. The tight-trust continuation retained the frozen 3/8 metrics-only
result without improving it. A separate video rerun scored 1/8 BC, 1/8 selected
RL, and 3/8 selected RL plus recovery, but run variability and missing
recovery-mode traces prevent a causal or promotion claim. See the
[video analysis](experiments/2026-09-23-serl-video-failure-analysis.md) and
[SC recovery plan](SC_CABLE_SNAG_RECOVERY_PLAN.md).

## 1. Does SERL require a probabilistic actor?

SERL uses an RLPD-style off-policy method based on Soft Actor-Critic (SAC).
SAC represents the actor as a probability distribution over actions. During
training, it samples actions from that distribution and rewards both useful
behavior and useful diversity. See the
[SERL paper](https://arxiv.org/abs/2401.16013),
[RLPD paper](https://arxiv.org/abs/2302.02948), and
[SAC paper](https://proceedings.mlr.press/v80/haarnoja18b.html).

This does not mean that perception or the world must be probabilistic. The
pose estimator can remain frozen and deterministic. The policy distribution
represents several actions that may be reasonable from the same observation.

## 2. Why one Gaussian around BC is usually just noise

Suppose the deterministic BC policy predicts one trajectory $a_{BC}$. The
simplest probabilistic conversion is

$$
a \sim \mathcal N\left(a_{BC},\Sigma\right).
$$

This says: use the BC trajectory as the center and randomly perturb it.

For a one-dimensional adjustment, this can be adequate. For a four-waypoint,
24-dimensional command, independent noise can produce an incoherent path:

```text
waypoint 1: left
waypoint 2: right
waypoint 3: rotate sharply
waypoint 4: left again
```

Even a correlated single Gaussian still has one center. Consider a blocked
state where two distinct recoveries are plausible:

```text
Recovery A: clear the obstacle on the left
Recovery B: clear the obstacle on the right
```

A single broad Gaussian centered between them assigns substantial probability
to the middle. In this example, the middle may point back into the obstacle.
Increasing its variance explores farther, but also produces more unsafe or
meaningless trajectories. The model is being made noisy rather than being
given a useful representation of alternative strategies.

The mean of a single Gaussian can move toward the eventually preferred
strategy as learning progresses. The initial discovery problem remains: it
must spread one cloud of probability across separated behaviors, and its
deterministic mean can lie between them.

## 3. Mixtures represent separate strategies

A mixture policy uses several probability components:

$$
\pi_\theta(u\mid s)
=
\sum_{k=1}^{K}
p_k(s)\,
\mathcal N\!\left(u;\mu_k(s),\Sigma_k(s)\right).
$$

Here:

- $s$ is the current visual, pose, robot, force, and history observation.
- $u$ is one complete four-waypoint trajectory.
- $p_k(s)$ is the probability of choosing strategy $k$.
- $\mu_k(s)$ is a complete trajectory for strategy $k$.
- $\Sigma_k(s)$ describes limited variation within that strategy.

With two components, the policy can represent:

```text
Component 1: move left around the obstruction
Component 2: move right around the obstruction
```

This is the correct intuition. The two components keep left and right separate.
The policy does not need to average them into a command that goes straight
forward. Rewards teach the mixture probability which side works better for the
observed scene.

With four components, the learned strategies could become:

```text
1. continue a safe insertion
2. move left and reapproach
3. move right and reapproach
4. change height or orientation and reapproach
```

These meanings are examples, not fixed semantic labels. Unless an experiment
explicitly fixes them, the network is free to use its components for the
strategies that best explain successful experience.

### Does a mixture guarantee useful exploration?

No. A mixture supplies enough structure to represent separated alternatives.
Learning still requires:

- Coherent trajectory sampling within each component.
- Failure and success rewards that distinguish the outcomes.
- Resets that expose recoverable blocked states often enough.
- Protection against all components collapsing onto the same trajectory.
- Safe execution limits and a reliable reset path.

The single-Gaussian baseline remains valuable because it tests whether the
mixture actually provides a measurable benefit.

## 4. Proposed actor architecture

```text
three camera observations
frozen observation-only pose estimator
robot pose and measured motion
force/contact history
previously executed actions
recovery mode and backtrack summary
             |
             v
existing pose-conditioned transformer
             |
             v
probabilistic full-trajectory head
             |
             v
four complete port-relative connector poses
             |
             v
existing deterministic SE(3) adapter
             |
             v
four TCP-body-frame commands
```

The simulator may use ground-truth geometry to calculate rewards and training
diagnostics. It must not pass this geometry into the actor.

### Policy action

The selected supervised target is the connector pose after each 50 ms teacher
command, relative to the port opening:

$$
{}^P T_{C,t+50:t+200\,\mathrm{ms}}.
$$

For probability calculations, use a minimal local coordinate vector:

$$
u =
[p_1,\omega_1,\ p_2,\omega_2,\ p_3,\omega_3,\ p_4,\omega_4],
$$

where $p_i\in\mathbb R^3$ is port-relative translation and
$\omega_i\in\mathbb R^3$ is a local axis-angle rotation. This gives 24 policy
coordinates. A deterministic decoder produces four physical SE(3) poses, and
the calibrated adapter produces the four executed TCP-body-frame commands.

The axis-angle conversion must be tested over the actual orientation range
before training. The current continuous 6D rotation output can initialize the
same physical mean after conversion.

### Correlated variation within a component

Each component should perturb the four waypoints coherently. One possible
covariance is

$$
\Sigma_k = B_kB_k^\top + \operatorname{diag}(\sigma_k^2).
$$

The low-rank term $B_kB_k^\top$ represents shared choices such as lateral
direction, retreat amount, rotation, and path curvature. The small diagonal
term keeps the density valid and permits limited local adjustment.

This produces samples such as:

```text
clear left -> continue left -> rotate -> cautiously reapproach
```

rather than unrelated noise at each waypoint.

## 5. BC warm start

1. Copy the existing perception, pose-gating, state encoders, and transformer
   weights.
2. Convert the deterministic physical trajectory to the minimal policy
   coordinates.
3. Initialize the primary mixture component to reproduce the BC trajectory
   with small variance and high probability.
4. Initialize the other components with low probability and small, safe,
   coherent diversity.
5. Freeze the actor while the critics undergo a bounded offline warm-up.
6. Enable actor learning only after held-out critic ranking checks pass.

The actor always produces a complete trajectory. It is not a frozen BC action
plus a learned residual correction. The BC checkpoint provides initial network
weights and an initial physical mean.

Early online actor training may retain a temporary supervised anchor:

$$
\mathcal L_{actor}
=
\mathbb E\left[
\alpha\log\pi_\theta(u\mid s)
-Q(s,u)
\right]
+
\lambda_{BC}\mathcal L_{BC}.
$$

Reduce $\lambda_{BC}$ as verified recovery experience accumulates. Keeping it
large indefinitely would preserve the CheatCode policy's forward-motion bias.

## 6. Critics

The critics are deterministic. Each receives an observation representation and
a proposed complete trajectory and predicts expected future reward:

```text
observation features ----+
                         +--> critic --> expected return
24D proposed trajectory -+
```

Replay must store both the policy-coordinate trajectory and the actual
executed TCP commands. The former identifies the actor decision; the latter
reveals clipping, tracking error, contact, or safety-controller intervention.

Critics can be used during training without being deployed. Normal deployment
only requires perception and the actor.

## 7. Temporal inputs

The actor needs a short history because one frame may not distinguish normal
contact from blockage. Candidate history covers three to five 200 ms decisions
and includes:

- Visual or frozen perception features.
- Predicted pose and uncertainty.
- Commanded motion.
- Measured TCP and plug motion.
- Force/contact measurements.
- Executed actions.
- Blocked-state detector output.
- Recovery-controller mode.

This supplies roughly 0.6--1.0 seconds of evidence without requiring a learned
world model.

## 8. Integration with deterministic backtracking

```text
NORMAL_POLICY
      |
blocked detected
      v
hardcoded BACKTRACK
      |
clearance confirmed
      v
mixture actor chooses a near-lateral trajectory
```

The actor is not queried during deterministic backtracking. After clearance,
its existing temporal condition contains measured state change, the previous
executed action, force, and a contact phase flag. The measured-path controller
also projects the short exploration window near the plane perpendicular to the
blocked direction. A later recurrent extension may add the blocked direction
and clearance explicitly; the current 304D BC condition does not contain
separate fields for them.

Record controller-owned backtracking steps for diagnostics and critic-aware
transition accounting, but mask them out of the actor loss. Otherwise, replay
would incorrectly label the controller's action as an actor decision.

## 9. Training and deployment behavior

During online training:

```text
sample a mixture component
sample a coherent trajectory within it
execute and record the exact trajectory
learn from reward and the resulting observation
```

The initially sampled component is held for three decisions, or 600 ms. This
prevents a left mode at one decision from immediately becoming a right mode at
the next. Each component still emits a newly conditioned four-command path.

The preserved BC checkpoint uses about 94.8% probability on the primary mode.
That is suitable for conservative validation but too close to BC plus noise for
an exploration comparison. A separate exploration initialization uses 70% on
the exact BC mode and 10% on each of three coherent lateral alternatives. The
alternatives lie 120 degrees apart in the local port-opening plane. The source
checkpoint remains unchanged.

### Implemented artifacts

- `outputs/experiments/2026-09-22_serl_recovery/actor_bc/single_gaussian_bc.pt`
- `outputs/experiments/2026-09-22_serl_recovery/actor_bc/mixture_bc.pt`
- `outputs/experiments/2026-09-22_serl_recovery/actor_bc/mixture_exploration_bc.pt`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/rpdp_serl_policy.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/rpdp_serl_actor.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/measured_path_recovery.py`

The single-component BC distillation stopped through validation patience at
2,500 updates; its best checkpoint was update 900. On the episode-grouped
development rows, the deterministic adapter command error was 0.022 mm median,
0.339 mm p95, and 0.0021 degrees median orientation error. Expanding it to a
mixture leaves component 0 exactly unchanged.

During deterministic evaluation:

```text
choose the most probable component
execute that component's mean trajectory
```

Do not average the means of left and right components. The critic need not run
during ordinary deployment unless a separate candidate-selection experiment
explicitly enables it.

## 10. Evaluation plan

Use identical development scenes, replay, update budgets, and success criteria:

1. Frozen deterministic BC.
2. Single-Gaussian SAC around a full trajectory.
3. Four-component correlated mixture SAC.
4. Four-component correlated mixture SAC with deterministic backtracking and
   near-lateral exploration.

Measure:

- Recovery success and final insertion success.
- Results by misalignment, high-force contact, and cable snag.
- Peak force and repeated blocked pushes.
- False backtracking during valid insertion.
- Component usage and component collapse.
- Diversity of sampled trajectories.
- BC behavior retention.
- Full actor inference p50, p95, and p99 under the 300 ms limit.

The mixture is promoted only if its separate components correspond to useful
behavioral alternatives and improve autonomous recovery. Extra modes without a
measured outcome improvement are additional complexity rather than evidence of
better control.

The bounded execution evaluated the four-component actor and recovery branch.
It did not run a full live single-Gaussian control arm after the mixture actor
failed to improve, so these results do not establish that a mixture is better
than a single Gaussian. They establish that the mixture representation works,
that naive online updates can collapse it, and that behavior regularization can
retain BC. Demonstrating a benefit from separate modes still requires actual
successful recovery experience.

## 11. Critic warm-up and first online update

Frozen-policy collection produced a prior with 12 complete episode groups:
seven local successes and five failures. Ten complete groups train the critic;
one success and one failure remain held out. The held-out mean Q values were
1.792 for the success and 1.011 for the failure. This is the desired ordering,
but two episodes are only a smoke-level diagnostic and do not establish critic
calibration.

Two matched online arms start from that critic and the same BC mixture. Each
uses 50% immutable prior replay and 50% new online replay, 250 gradient updates,
a small temporary BC anchor, and no simulator geometry as actor input. The
correlated arm finished with finite losses and nonzero actor gradients. Its
mixture entropy decreased from roughly 1.21 at update 100 to 0.89 at update
250, while maximum component probability rose from 0.36 to 0.51. These are
training diagnostics; held-out autonomous insertion determines whether the
change is useful.

The first held-out test exposed a missing part of that anchor. Preserving only
component 0's mean allowed SAC to put essentially all deterministic-deployment
preference on component 3 or component 2. Both trained arms fell from the
frozen policy's 3/8 to 0/8. The actor objective now regularizes the complete
mixture distribution:

$$
\mathcal L_{anchor}
=D_{KL}\!\left(\pi_{BC}(k\mid s)\,\|\,\pi_\theta(k\mid s)\right)
+\sum_k \pi_{BC}(k\mid s)\,\ell(\mu_{\theta,k},\mu_{BC,k}).
$$

This keeps the high-probability safe behavior intact while still allowing
bounded changes to alternatives. Weight 1.0 prevented categorical collapse,
but a small primary-trajectory shift still produced 0/8. The final bounded
rerun used a tenfold smaller actor rate and a tenfold stronger anchor. A bug in
optimizer restore initially overrode that smaller rate; after repair, the
effective rate was $2\times10^{-6}$ and the run recovered the frozen result:
3/8 on the same episodes. It did not improve insertion, so the actor is not
promoted beyond the supervised checkpoint.
