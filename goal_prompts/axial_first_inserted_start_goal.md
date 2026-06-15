/goal Stop the currently running goal and all existing training/eval/monitor jobs for this task first. Leave the runtime in a clean state before starting this new work.

This replaces the current training attempt. The current issue is that the policy can laterally align near the entrance but does not learn sustained axial insertion. Redesign the experiment to first prove full axial insertion from an already aligned, partially inserted state before reintroducing harder pre-entrance and lateral-offset starts.

Implement and audit this axial-first inserted-start setup:

1. Runtime cleanup
   - Kill all stale train/eval/monitor jobs for the insertion task.
   - Confirm no old training or eval processes are still running before launching the new run.

2. Reward changes
   - Keep the high-level stateful reward design: lateral/orientation alignment first, axial insertion only after alignment.
   - Temporarily remove or disable the force penalty reward term for this experiment.
   - For the axial phase, make axial-forward progress and final depth/full insertion the dominant positive signal.
   - Keep lateral and orientation penalties active during insertion so the policy cannot get credit for off-axis insertion.
   - Do not add guides, guards, action overrides, scripted rollout behavior, or any non-actor control. Training and eval must use the actor policy only.

3. Initial condition for the first curriculum level
   - Start partially inserted and aligned, so the first task is only to finish insertion.
   - Use lateral distance `r = 0 mm`.
   - Use near-zero orientation error, target around `theta = 0.02 rad` or lower.
   - Use an axial start that is already inside the entrance/corridor rather than 3 mm before the entrance. Choose a small positive inserted depth that is safe and contact-stable, then document the exact `s` used.
   - The intended first diagnostic is: can the actor learn full insertion from a partially inserted, centered, nearly aligned state?

4. Reward numerical audit before training
   Audit multiple states/actions and print the results before launching training:
   - partially inserted, `r=0`, `theta≈0.02`, axial-forward action: should be strongly rewarded.
   - same state, hold: should be worse than axial-forward.
   - same state, axial reverse/retreat: should be penalized.
   - same state, lateral action: should be worse than clean axial-forward.
   - partially inserted but laterally off-axis, axial-forward: should be penalized or gated down.
   - partially inserted but orientation misaligned, axial-forward: should be penalized or gated down.
   - fully inserted strict-success candidate: should be highest among valid states.
   - random sampled `s/r/theta/action` cases to confirm monotonicity and sanity.

5. Reset/randomization audit
   - Confirm runtime reset actually produces the requested partial-insertion start.
   - Probe a few reset frames and report measured `s`, `r`, and `theta`.
   - Confirm force penalty is disabled in the active train/eval config.

6. Training run
   - Start a fresh actor-only training run after audits pass.
   - No guide, no guard, no action override.
   - Start with the partially inserted aligned level first.
   - Evaluate periodically at the current level.
   - Save checkpoints periodically.
   - Do not advance to harder starts until the current inserted-start level shows strict success.

7. Reintroduction schedule after success
   Once full insertion works from the partially inserted aligned start:
   - Move start back toward the entrance while keeping `r=0` and `theta≈0.02`.
   - Then test `s=0, r=0`.
   - Then test `s=-0.5 mm, r=0`.
   - Then test `s=-3 mm, r=0`.
   - Only after axial-only starts work, reintroduce lateral offsets gradually.

8. Monitoring/reporting
   Report after each eval:
   - best `s`, `r`, `theta`
   - strict-success count/streak
   - whether axial insertion improved from the initial condition
   - whether any positive depth happened off-axis and should be ignored
   - confirmation that guide/guard/action override stayed disabled

If there is no clear progress for more than 2 hours on the inserted-start aligned level, stop, reassess the reward scale/formula and reset geometry, make a minimal targeted change, audit again, and resume. Do not solve it by adding hard-coded rollout behavior.
