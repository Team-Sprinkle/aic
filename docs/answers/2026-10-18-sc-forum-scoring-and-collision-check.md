I checked. Other teams reported similar problems:

- **SC looks inserted but receives partial credit**, sometimes also SFP. A reply explains that completion requires one second of continuous contact. [Discussion](https://discourse.openrobotics.org/t/full-insertion-in-sc-port-seems-impossible/54219)
- **Phantom collisions after earlier trials:** organizers confirmed stale collision meshes and cable friction/inertia problems, and released fixes. Another participant reported almost zero on the third trial. [Official notice](https://discourse.openrobotics.org/t/note-to-participants-gazebo-bug-fix-changes-to-friction-properties-of-the-cable/54098)

**Our 1-point result is not yet evidence of that bug.** Our policy reported an unsuccessful attempt, leaving only model-validation credit. SC also failed in fresh standalone runs.

Both announced fixes are already in our repository history. I am checking the frozen simulator binary separately before attributing failures to physics. I documented these findings in the [experiment record](../experiments/2026-10-18-three-trial-insertion-repair-cycle.md#sc-follow-up-completed-and-public-discussion-checked) and STATUS.md.
