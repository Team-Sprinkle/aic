# Explaining the motion improvement in easy words

## 1. What are “consecutive predictions”?

The actor does not plan the whole insertion once. It repeatedly looks at the situation and makes a short movement plan.

Our current plan contains **four target poses, 50 milliseconds apart**: about **200 milliseconds of movement**. We execute those four targets, then ask the actor for another plan using updated observations.

```text
Look → predict 4 targets → execute them
Look again → predict the next 4 targets → execute them
Look again → ...
```

“Consecutive predictions” means these neighboring plans: the plan we just executed and the next one we generate.

Ideally, the next plan continues smoothly from the previous one. For example, imagine sideways target positions, in millimeters:

```text
Previous plan:   1 → 2 → 3 → 4
Smooth next:    5 → 6 → 7 → 8
Abrupt next:  -10 → -9 → -8 → -7
```

In the abrupt case, the commanded destination suddenly switches from +4 to −10 mm. The robot cannot teleport, but its controller now tries to move toward a very different destination. That can produce back-and-forth movement.

**This is the kind of command inconsistency that decreased in the new SFP tests.**

## 2. What did the connector-orientation change help with?

Imagine holding a pen slightly crooked in your hand. Knowing your hand's angle does not tell you the pen's exact angle.

Previously, we explicitly represented the plug tip's position, but used the gripper's orientation. The actor had to learn the missing relationship implicitly.

Now, an image model estimates how crooked the plug is in the gripper. We use that estimate both when training the actor and when converting its predicted plug movements into robot commands.

For example, if the gripper is level but the plug is tilted by 2°, correctly aligning the plug may require tilting the gripper in the opposite direction. The new representation makes that relationship explicit.

**Our explanation is that this helped the actor generate more consistent movements.** We have evidence of smaller command jumps, but have not isolated this mechanism experimentally from the other changes made during retraining.

## 3. What is “simulator completion locking”?

This is behavior built into Gazebo's cable plugin, not something our actor learned.

After the configured port-touch event triggers, the plugin:

1. Makes the cable connectors fixed in the simulated world.
2. After a delay, also fixes the robot's end effector—the gripper/tool—in place.
3. Publishes a task-completion message.

It does this to simulate an attached cable and avoid physics/controller problems after attachment.

**Therefore, a motionless robot at the end of a video does not necessarily mean the policy learned to hold perfectly still. The simulator may be preventing movement.**

In our tests, this also happened in SC episodes that did not meet our stricter depth/alignment requirements. The simulator's completion event and our geometric insertion check are separate things.

That is why we compared early motion **before this locking**, rather than counting the frozen end of the video as improved smoothness.

## 4. What improved, and what still needs fixing?

**Improved:** the SFP actor's next movement plans disagree less with its previous plans. Large back-and-forth commands became smaller. The reported 15.4 → 4.4/3.2 mm values describe large command jumps—the 95th percentile—not how far the robot instantly moved.

**Still not fixed:**

- When the approach controller hands control to the insertion actor, their requested destinations can differ suddenly.
- While the plug is blocked, the actor can request a point far inside the port. When alignment improves, it can then advance quickly.

Think of the second issue as repeatedly telling someone “go another 4 cm forward” while a door is closed. Once the door opens, that distant destination can cause a sudden advance. This is an analogy for position-control demand, not proof that physical force was stored like a spring.

So the next repair would be to **make the handover continuous and prevent the requested forward destination from getting too far ahead of actual progress**. Those repairs have not been tested yet.
