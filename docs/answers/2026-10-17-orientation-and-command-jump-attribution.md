# Did using plug orientation reduce sudden changes?

**Probably, but we have not proven that plug orientation alone caused it.**

We did not add a smoothing filter. Instead, we changed and retrained the system so that:

1. Images estimate the plug's orientation relative to the gripper.
2. The actor receives and predicts the plug's full pose relative to the port.
3. Command conversion uses that same estimated plug-to-gripper relationship.

For example, if the plug is held 2° crooked, the actor now explicitly receives that information. Previously, it had to infer the relationship between the gripper and plug. Giving it a consistent description may help successive movement plans agree better.

After these changes, large SFP command jumps decreased from about15.4mm p95 to4.4/3.2mm in two repeats. We did not increase history length or change the four-step execution schedule.

**The demonstrated result is that the combined changes reduced SFP command jumps in these tests.** To establish that orientation was the cause, we would need a controlled comparison that changes orientation information alone. SC did not show consistent improvement, so this is not yet a general jerk fix.
