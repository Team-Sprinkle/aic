# Why the recent MoveIt test used SFP-to-NIC

You're right: **the difficult cable-around-cards problem is mainly
SC-to-SC**, especially with multiple NIC cards. The recent SFP-to-NIC
MoveIt video did **not** test that problem.

I chose SFP because the active experiment at that point was trying to
repair the **SFP transport actor**. That actor had reached the safe area
near the requested port on **0/3 independent starts**. We needed route
examples from states the actor actually visits, and short CheatCode
recovery snippets were not giving reliable training labels. I therefore
tested whether MoveIt could generate a route from an off-course SFP actor
state and whether Gazebo could execute one. That tests a possible *label
generation tool*, not the SC snag hypothesis.

The distinction matters:

- **SFP issue in this experiment:** the learned actor chooses and follows
  the requested-card route poorly. The single privileged MoveIt/Gazebo run
  reached a point 85 mm before the port with no scored contact, but did
  not insert. It says little about cable recovery.
- **SC issue we care about:** the plug, gripper, or cable may need to go
  *around* a multi-card row. A path can be fine for the rigid arm yet pull
  the cable against a card. Earlier SC work already tested lateral bypasses
  and MoveIt routes; some teacher routes inserted, while cable-contact
  attribution and robust autonomous recovery remain open.

It's slightly too strong to say SFP **never** had collisions: an earlier
two-card **autonomous actor** test received a contact penalty. But the
ordinary privileged SFP trajectories did not establish the persistent
multi-card cable snag that motivated the SC work. The SFP cable-clearance
proxy's 3/6 result was especially easy to overread; it was a conservative
calculation, **not three observed collisions or three insertions**.

So this was a useful check that live-state MoveIt planning and Gazebo
joint replay can work, but **I let the experiment drift away from the main
SC snag question**. The right next application of that tool is a bounded
five-card SC-to-SC test: use the existing SC lateral-bypass scenes, cover
both requested SC ports, record cable/card and plug/port contacts plus
force and measured progress, and compare the route with its matched
failure. We should call something a cable snag only when the contact and
motion evidence supports it. The SFP video is not evidence of solving SC
transport or recovery.

Relevant records: [SFP route audit](../experiments/2026-10-08-sfp-requested-port-corridor-audit.md),
[SC MoveIt pilot](../experiments/2026-09-29-moveit-sc-route-pilot.md), and
[SC cable-snag recovery plan](../SC_CABLE_SNAG_RECOVERY_PLAN.md).
