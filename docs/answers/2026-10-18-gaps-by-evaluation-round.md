# Remaining gaps in each evaluation round

For the **latest unchanged public three-round sample**:

| Round | Task | Result | Main remaining gap |
| --- | --- | --- | --- |
| 1 | SFP into NIC 0 | **88.03/100; inserted** | Improve speed and path efficiency; verify repeatability. |
| 2 | SFP into NIC 1 | **1/100; failed** | Incorrect plug-orientation estimate before insertion. |
| 3 | SC into SC port 1 | **81.54/100; inserted** | Increase insertion reliability and seating margin, then shorten the route. |

## Round 1: works in this run

It inserts **46.77 mm**, with **0.264 mm lateral error and 0.146° orientation error**. Transport, handoff and insertion all succeed.

The remaining score loss is from time, path efficiency and smoothness: approximately **48 seconds** and **0.48 m** of motion. This round does not currently justify another model repair; preserve it while fixing round 2. One successful run does not establish reliability across randomized settings.

## Round 2: the main blocker

Transport reaches the insertion stage, but insertion stops at **21.18 mm**, short of the **44 mm** physical gate. Lateral error is **0.825 mm** and orientation error **3.64°**.

The controller believes the plug is straight because its initial visual estimate is already **3.53° wrong**. The actual plug-to-gripper roll is about **26.5°**, outside the original training-label range of **17.8–23.1°**. Repeating the same model on close-up images still gives about **3.57° error**.

The contact evidence shows the plug/module pressing against the NIC opening, rather than a cable snag. The next repair should broaden and validate SFP grasp-orientation coverage. Simply increasing stiffness or averaging more predictions does not address this measured bias. This round succeeded in earlier runs, so the issue is reliability, not an inability to insert SFP at all.

## Round 3: succeeds, but with little margin

SC inserts **15.510 mm**, with **0.141 mm lateral error and 0.469° orientation error**. The gated 25 mm re-estimation and tracking repairs work in this run.

However, its depth is only about **0.010 mm above our 15.5 mm gate**. The four matched SC development cases also pass, but one has only **0.003 mm** depth margin. Broader repeated testing is needed before calling SC robust. The SC pose model also still misses its original standalone qualification gate: orientation p95 **1.026° versus 1° required**.

The SC route takes approximately **59 seconds** and **1.40 m**, so it also loses speed and efficiency points. Optimize that after reliability improves.

**Priority:** fix round 2's orientation coverage first, verify round 3's reliability, then optimize speed and smoothness. Latest total: **170.57/300**. These are explicit visual-controller results, not learned BC/RL insertion. Final scenes remain sealed.

Saved response: [2026-10-18-gaps-by-evaluation-round.md](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-18-gaps-by-evaluation-round.md).
