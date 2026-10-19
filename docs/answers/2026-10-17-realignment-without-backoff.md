# Realignment without backoff: what was learned?

**Realignment does not always require backing away.** If contact is light enough, a small sideways movement or rotation can align the plug while it stays near the opening. Backoff becomes useful when contact prevents those corrections—for example, the plug is wedged or the cable is caught.

## Did our demonstrations contain alignment?

**Yes. Successful teacher episodes include the movements that produce success, not just perfectly aligned straight insertion.** Also, the current corpus is not solely the original naive CheatCode recordings: it includes later scripted, privileged teachers for approach and alignment.

For example, the SFP wide-approach teacher explicitly:

1. Starts with position/orientation offsets.
2. Moves the plug sideways toward the port axis and rotates it toward alignment.
3. Advances toward and into the opening only while alignment satisfies its thresholds.
4. Continually adjusts its commanded position to compensate for tracking error.

Those commands become BC labels. The trained actor can therefore learn local alignment from successful demonstrations, even without demonstrations of a deliberate “hit, retreat, choose another route” sequence. Simulator geometry was available to the teacher during collection; it is not supplied to the autonomous actor.

Our earlier dataset audit also found backward-pointing near-port commands. However, these can represent settling or controller compensation; they do **not** by themselves prove that we have useful backoff/recovery demonstrations.

## What happened in the successful rollout?

The measured plug initially waited near the opening with roughly **4 mm lateral error**. Its lateral alignment then improved, and it advanced into the port. Hardcoded recovery was disabled.

However, **I should not present that as proof that it learned a deliberate recovery strategy.** The motion can result from a combination of:

- Learned alignment commands from the demonstrations.
- Repeated predictions using the current images and robot state.
- The compliant low-level controller and physical contact guiding the plug.

We have not isolated their contributions or identified a matching teacher episode that proves the actor copied this exact blocked-contact correction. The final stationary period is also partly simulator locking after completion, not evidence of learned stabilization.

**So your concern is correct:** successful expert data can teach ordinary alignment, but does not automatically teach recovery from unfamiliar jams or cable snags. This one success supports local alignment capability; it does not establish robust recovery. That is still why hierarchical backtracking and later recovery training matter.
