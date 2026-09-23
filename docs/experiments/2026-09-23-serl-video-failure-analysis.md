# SERL video rerun and per-episode failure analysis


> The MP4s and temporal sheets are local ignored artifacts. Start with the
> [video review page](../../outputs/experiments/2026-09-22_serl_recovery/videos/index.html).
> The committed measurements and conclusions are retained here even if those
> local media files are moved.
Date: 2026-09-23

This report reviews the synchronized center, left, and right videos from the
recording-enabled BC, selected trust-region RL, and selected RL plus
backtracking reruns. It uses the simulator geometry printed on the videos only
for post-run diagnosis. That geometry was not passed to any evaluated policy.

## How to read the overlay

- `s` is signed SFP-tip progress along the port axis. Negative values remain in
  front of the entrance. The configured local target is `+8 mm`.
- `r` is distance from the port axis. Approximately `0.5 mm` is the success
  corridor in this experiment.
- `o` is orientation error in radians. `0.03`, `0.04`, and `0.05 rad` are about
  `1.7`, `2.3`, and `2.9 degrees`.

Positive `s` alone is not insertion. A plug can pass the entrance plane beside
the port while `r` is several millimeters; this is called a lateral bypass
below.

The final numbers are rounded readings from the recorded overlay. The large raw
per-step logs were removed after video encoding because the host filesystem had
less than 500 MB free, so this retrospective review cannot recover a reliable
per-frame force trace. A cable snag is therefore reported only when the video
itself supports it; otherwise cable motion is marked as a possible contributor.

## Summary

| Configuration | BC | RL | RL + backtracking | Most likely explanation |
| --- | --- | --- | --- | --- |
| `serl_dev20_09` | Success: `s=+7.4`, `r=0.4` mm | Success: `s=+7.4`, `r=0.5` mm | Success: `s=+7.4`, `r=0.4` mm | No failure |
| `serl_dev20_10` | Fail: `s=-0.2`, `r=7.8` mm | Fail: `s=-0.2`, `r=8.7` mm | Fail: `s=-0.2`, `r=7.8` mm | Severe lateral divergence before insertion |
| `serl_dev20_11` | Fail: `s=-0.1`, `r=1.5` mm | Fail: `s=-0.1`, `r=3.1` mm | Success: `s=+6.2`, `r=0.4` mm | BC/RL reach the opening region but remain laterally outside the corridor |
| `serl_dev20_12` | Fail: `s=-2.8`, `r=0.3`, `o=0.04` | Fail: `s=-1.2`, `r=1.2`, `o=0.04` | Fail: `s=-2.8`, `r=0.3`, `o=0.05` | Port-lip/angle stall; cable contact is possible in the plain-RL video |
| `serl_dev20_13` | Fail: `s=-1.9`, `r=5.1` mm | Fail: `s=-3.2`, `r=1.8` mm | Fail: `s=+6.3`, `r=16.6` mm | Wrong lateral correction; backtracking arm passes beside the port |
| `serl_dev20_14` | Fail: `s=-0.1`, `r=3.5` mm | Fail: `s=+5.9`, `r=0.5` mm | Success: `s=+6.9`, `r=0.5` mm | BC misses laterally; RL aligns but stops short of seating |
| `serl_dev20_15` | Fail: `s=+4.4`, `r=16.0` mm | Fail: `s=-0.1`, `r=5.8` mm | Fail: `s=-0.2`, `r=8.6` mm | Severe lateral divergence; BC depth is a bypass, not insertion |
| `serl_dev20_16` | Fail: `s=+6.4`, `r=17.5` mm | Fail: `s=-0.1`, `r=6.5` mm | Fail: `s=+3.3`, `r=13.5` mm | Severe lateral divergence; BC/backtracking move past the entrance outside the opening |

All values above are millimeters except `o`, which is radians. A recorded
success can terminate between saved frames, so the displayed final `s` can be
short of exactly `8 mm`.

## Episode details

### `serl_dev20_09`

All three arms align and seat successfully. The cable bends but remains clear
of the connector path. This is the only success common to all three recording
runs.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_01_serl_dev20_09.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_01_serl_dev20_09.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_01_serl_dev20_09.jpg)

### `serl_dev20_10`

The reset starts close to the axis, around `r=0.4 mm`, but every arm moves to
roughly `8 mm` lateral error and makes essentially no axial progress. The cable
is visible beside the robot and does not appear trapped between the connector
and port. This is a policy/control alignment failure, not a visible cable snag.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_02_serl_dev20_10.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_02_serl_dev20_10.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_02_serl_dev20_10.jpg)

### `serl_dev20_11`

BC and plain RL stop around the entrance with lateral errors of `1.5` and
`3.1 mm`. The backtracking run reaches a successful centered state. Several
cables are visible, but none is clearly wedged in the moving connector's path.
The supported diagnosis is lateral misalignment. A single successful rerun is
not enough to say that backtracking solved a cable problem.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_03_serl_dev20_11.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_03_serl_dev20_11.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_03_serl_dev20_11.jpg)

### `serl_dev20_12`

This is the strongest candidate for cable or robot interference. A loose cable
moves across the work region in the plain-RL views. Even so, all three arms end
at or in front of the entrance with `0.04--0.05 rad` orientation error; BC and
backtracking are laterally centered but cannot advance. The primary observable
failure is therefore an angular port-lip stall. Cable load may contribute, but
the recordings do not prove a snag.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_04_serl_dev20_12.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_04_serl_dev20_12.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_04_serl_dev20_12.jpg)

### `serl_dev20_13`

All arms start near the axis. BC and plain RL drift laterally before reaching
the port. The backtracking arm advances to positive depth while moving
`16.6 mm` away from the axis, meaning it passes beside the port rather than
entering it. No cable is visibly blocking the opening. This is a bad lateral
correction, and the recovery branch makes that rollout worse.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_05_serl_dev20_13.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_05_serl_dev20_13.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_05_serl_dev20_13.jpg)

### `serl_dev20_14`

BC drifts to `3.5 mm` lateral error and never inserts. Plain RL reaches the
`0.5 mm` corridor and about `5.9 mm` depth but times out short of the local
seating event. Cable loops are active in its side views, so cable load is a
possible contributor, but the plug continues to make axial progress and no
definite wedging event is visible. The backtracking rerun succeeds. Because the
recording runs are not perfectly repeatable, this comparison does not prove
that recovery cleared a snag.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_06_serl_dev20_14.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_06_serl_dev20_14.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_06_serl_dev20_14.jpg)

### `serl_dev20_15`

Every arm develops a large lateral error. BC reports positive depth only
because the connector moves past the entrance plane outside the port. The
cable remains away from the opening in the available views. This is lateral
bypass/alignment failure.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_07_serl_dev20_15.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_07_serl_dev20_15.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_07_serl_dev20_15.jpg)

### `serl_dev20_16`

This configuration starts almost exactly centered but all three controllers
drive away from the port axis. BC reaches `r=17.5 mm`, plain RL reaches
`6.5 mm`, and backtracking reaches `13.5 mm`. Positive depth in BC and the
backtracking arm is lateral bypass. This is especially strong evidence of an
incorrect lateral policy response because the reset itself is well aligned.
There is no visible cable obstruction at the opening.

[BC sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/bc_episode_08_serl_dev20_16.jpg) ·
[RL sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_episode_08_serl_dev20_16.jpg) ·
[RL + backtracking sheet](../../outputs/experiments/2026-09-22_serl_recovery/videos/diagnostic_sheets/rl_backtracking_episode_08_serl_dev20_16.jpg)

## Conclusion

The recordings do not support a general cable-snag explanation. The dominant
failure is that the policy chooses the wrong lateral motion and leaves the
`0.5 mm` corridor. One configuration stalls at the lip with an angular error,
and one plain-RL rollout aligns but stops short. Cable motion may contribute in
configurations 12 and 14, but proving that requires retaining synchronized
force, measured TCP/plug motion, recovery-state, and contact-event traces in the
next recording run.

## Why a small backoff can help

The implemented recovery is not simply “move backward slightly, then repeat
the same command.” Its configured sequence is:

1. Trigger only after two consecutive control steps with at least `8 N` force,
   a nontrivial motion command, and little measured TCP motion.
2. Retrace measured TCP positions in `0.25 mm` command steps until it has moved
   at least `1 mm` and force has fallen to at most `4 N`. The maximum allowed
   retreat is `10 mm`.
3. For the next 12 control steps, retain the policy's sideways motion but keep
   only 10% of motion toward the blocked direction and 50% of motion away from
   it. At 20 Hz, this modified retry lasts about `0.6 s`.

One millimeter looks small at scene scale, but it is twice the experiment's
`0.5 mm` lateral success corridor. For a connector caught on the port lip, that
can fully separate the contacting surfaces, release friction and elastic cable
preload, and let the next lateral command approach from a different point. The
policy also receives a new observation after the retreat, so its subsequent
trajectory need not repeat the original one.

This mechanism plausibly explains the two additional recorded successes:

- In `serl_dev20_11`, plain RL finishes `3.1 mm` off-axis, while the recovery
  run finishes centered and succeeds. The post-retreat near-lateral phase could
  have prevented another immediate forward push and preserved a better lateral
  correction.
- In `serl_dev20_14`, plain RL is already centered and reaches about `5.9 mm`
  depth. The recovery run needs only a little more axial progress to terminate
  successfully, so releasing a shallow lip contact could be sufficient.

However, the videos do not prove that the hard-coded recovery state machine
caused either success. A short backward move may also be an ordinary learned
policy command. Per-step recovery mode and force traces were not retained, and
the recording-enabled simulator runs were visibly nondeterministic: the BC and
plain-RL results changed from their earlier metrics-only evaluations. Recovery
also made `serl_dev20_13` much worse by ending far to the side of the port.

The defensible conclusion is therefore that a millimeter-scale retreat plus a
changed lateral retry *can* release a local port-lip contact. These recordings
do not establish that it reliably solved cable snagging. A future comparison
must retain compact `backtrack_events`, recovery mode, retreat distance, force,
measured TCP/plug motion, and blocked direction for every control step.
