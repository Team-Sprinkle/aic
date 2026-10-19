# What the 5–6 cm waypoint variation means

**It is across different randomized settings, while keeping the requested port and route stage fixed.** It is not variation from repeatedly running one identical setting.

I calculated the two targets separately:

- **SC port 0:** 35 training episodes.
- **SC port 1:** 48 training episodes.

Each group includes different board positions/orientations, card counts from 1 to 5, card/rail placements and recorded robot starts. Port 0 and port 1 were **not mixed together**. The calculation does not isolate which variable caused how much variation.

For example, for **port 0's “travel forward beside the row” waypoint**, I compared its planned base-frame position across those 35 settings. The 95th-percentile distance from the group's median waypoint was **54.3 mm**.

This measures **where the teacher placed the waypoint**, not robot tracking error, port-localization error, or repeatability. It also is not a ±5–6 cm tolerance we have shown to be safe.

**Your distinction matters:** a route conditioned on a narrower scene configuration—or adjusted using a rough board estimate—could have substantially less remaining variation. We have not measured that residual yet. The next offline check should separate variation explained by board placement, selected-rail placement and initial robot pose before deciding how much clearance a rule-based route needs.

So this number argues against one unadjusted coordinate list across all these settings. It does **not** rule out reliable waypoints for a specific setting or an observation-adapted template.
