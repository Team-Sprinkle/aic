# Why 22/22 and 7/7 differ from 2/3

Yes: **2/3 meant two of the first three new evaluation episodes passed the handoff gate.** These were different evaluations from the earlier 22/22 and 7/7.

| Count | What was tested |
|---|---|
| SFP 22/22 | Earlier development cases used while choosing the transport rules. |
| SC 7/7 | Earlier development screen of the selected rules. |
| 2/3 | First three fresh, sample-derived SC evaluation episodes after freezing the rules and models. |

The development sets were smaller and did not establish reliability across new randomized starts. The fresh evaluation varies layouts, card counts, requested ports, and starting grasp perturbations. Cable/contact behavior can also vary between runs.

**The current evaluation has progressed to 53/56 passing:** SC 11/12 and SFP 42/44, at the checked snapshot. This remains an incomplete evaluation, not the final success rate. The planned batch contains 124 sample-derived episodes plus six already-exposed official-scene repetitions after freezing. The first three were sample-derived cases, not those official repetitions. Reserved internal final scenes remain sealed.

## What “handoff passed” means

It is more than the coarse model saying yes:

1. Coarse perception finds the requested port region; the high-resolution refiner estimates its pose.
2. Repeated fresh estimates agree, and the robot reaches and holds the acquisition position.
3. Independent checks confirm the requested port, an opening-position estimate within 1 mm, and the plug approximately **130–180 mm before the opening**, with lateral offset at most 25 mm and orientation error at most 15 degrees.
4. There is no scored harmful contact/force violation, and perception latency p95 is below 300 ms.

Ground-truth geometry is used only afterward to check these results, not to guide the autonomous robot. **This is transport/handoff success, not insertion success. The insertion actor was not run.**

The first failed new episode could not obtain acceptable fresh visual acquisition and ended unsuccessfully after recovery. The other two passed.

One further qualification: the SC 7/7 development screen never triggered backtracking. A separate pilot successfully exercised recovery, so we cannot attribute the 7/7 result to backtracking or claim a measured recovery success rate from it.

My earlier presentation should have separated development results from fresh evaluation results more clearly.
