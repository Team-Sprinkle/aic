# Gazebo recovery: backtrack, sideways retry, insert

[Annotated normal-speed MP4](gazebo_backoff_realign_insert.mp4) · [2× GIF](gazebo_backoff_realign_insert.gif)

![SC recovery](gazebo_backoff_realign_insert.gif)

This supplemental SC-to-SC episode has three cards. It comes from the first repaired released-config replay, not the selected repeat-10 final demonstration. The explicit visual controller supplies insertion actions; a rule-based supervisor supplies recovery. No RL actor or true target geometry chooses these movements.

| Recorded event | Simulation time |
| --- | ---: |
| Forceful stall triggers local recovery | 87.30 s |
| Measured retreat completes | 91.90 s |
| Sideways retry completes | 94.75 s |
| Terminal observation after insertion | 101.90 s |

Net measured TCP retreat is **18.41 mm**; measured retry displacement is **3.16 mm**. The trigger records **1.69 N** load change. There is one local backoff and zero hierarchical route resets. The terminal scoring record awards insertion; this qualitative example does not establish a recovery success rate or superiority over recovery disabled. In particular, it does not establish that the preceding blockage was a cable snag.

The MP4 shows simulation time 84.80–101.90 s at 20 fps, H.264/yuv420p. The left panel is a diagnostic side view; the right is a fixed crop of the center wrist camera. Labels come from saved runtime events, not a new rollout. The recording has a maximum 0.30 s source-camera gap in this interval; nearest recorded frames are held through that gap. The GIF is 800 pixels wide and plays at 2× speed.

[Manifest](manifest.json) contains trace/video/image hashes, event records and the archive-relative source identifier. Recreate with `scripts/export_project_recovery_evidence.py --source <recorded-rollout-directory>`; source frames are external artifacts.

The associated [perception montage](../../images/cable_insertion/gazebo_predicted_port_landmarks.png) draws only saved model predictions from SFP and SC runtime acquisition. Five cyan landmarks and a yellow opening center are overlaid on native-resolution crops; arrows show projected reference directions, not measured 3D orientation-error arrows. No true target points are drawn, and there is no oracle-based crop selection.
