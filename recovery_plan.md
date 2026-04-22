# Recovery Data Generation TODO

Goal: generate insertion data where recovery action is required.

## P0: State Machine

- [ ] Extend `CheatCodePIDController.py` with an explicit state machine:
  - `TRANSPORT`
  - `PRE_INSERTION`
  - `INSERT`
  - `SLIP`
  - `STUCK`
  - `RECOVER`
  - `FAILURE`

- [ ] Label every timestep/sequence with the current state.
- [ ] Train or add an explicit action/state classifier neural network for data
  labeling.
- [ ] Use state labels to split long demonstrations into sub-sequences.

## P0: Recovery Data Structure

- [ ] Treat `RECOVER -> PRE_INSERTION` as the main recovery sub-sequence.
- [ ] Collect recovery sub-sequences separately from full insertion episodes.
- [ ] Allow future teleop collection for recovery-only data.
- [ ] Train policy on recovery sub-sequences as a separate behavior module or
  labeled mode.

## P0: Collision / Failure Injection

- [ ] Inject recovery-required episodes during `TRANSPORT`.
- [ ] Apply an external force in random X/Y direction at a random time.
- [ ] Use force magnitude above 20 N lasting over 1 second for failure
  injection experiments.
- [ ] Log injected force direction, magnitude, duration, and timestamp.
- [ ] Define what counts as contact.
- [ ] Define what counts as `recovery_needed`.
- [ ] Reject episodes that create off-limit contact or unrecoverable failures.

## P0: Contact and Recovery-Needed Detection

- [ ] Add low-pass filter for F/T sensing before contact classification.
- [ ] Detect contact from force/torque signal.
- [ ] Detect `SLIP` from unexpected lateral motion or pose deviation.
- [ ] Detect `STUCK` from low insertion progress under contact force.
- [ ] Declare `recovery_needed` when:
  - state becomes `SLIP` or `STUCK`
  - nominal insertion no longer converges
  - the plug is no longer near the planned pre-insertion pose

## P0: Recovery Behaviors

- [ ] Teleop recovery:
  - human estimates a good pre-insertion pose
  - teleop moves plug back toward `PRE_INSERTION`
  - save this as `RECOVER -> PRE_INSERTION`

- [ ] Planning recovery:
  - move closer to target/pre-insertion pose
  - use small wiggling or local search near the port
  - return to `PRE_INSERTION`, then retry `INSERT`

- [ ] Compare teleop recovery and planning recovery data.

## P1: Scoring / Safety Checks

- [ ] Track max force and force duration above 20 N.
- [ ] Track off-limit contacts.
- [ ] Track success, partial insertion, and failure.
- [ ] Keep successful recovery examples for imitation training.
- [ ] Store failed recovery examples separately for diagnostics.
