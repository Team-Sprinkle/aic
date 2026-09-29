# Step 5 SC teacher and autonomous review clips

These H.264 MP4s were rendered on September 25 from **existing** three-camera
JPEG frames. No new simulation, policy training, supervisor, or RL ran for
the videos. The left, center, and right camera streams are side by side.
The files are under `outputs/experiments/2026-09-25_step5_review_videos/`.
The two teacher clips below use the earlier SC route. They show a scored
teacher-assisted insertion, **not** an endorsed cable route through crowded
cards. The later [lateral-bypass experiment](2026-09-25-sc-lateral-bypass-and-policy-handoff.md)
records a behind-first, outside-left path with full-cable views and geometry.

| Clip | Source, control, and official result | Capture limits |
| --- | --- | --- |
| [teacher trial 1003101](../../outputs/experiments/2026-09-25_step5_review_videos/teacher_interleaved_trial_1003101.mp4) | Training-only SC port 0. Frozen body-target actor controlled 0.2 s pulses every 4 s until 30 s; privileged teacher controlled between pulses and finished. Tier 3 = 75, full insertion, admitted as a correction candidate. | 623 collected three-camera frames at 288×256 each, encoded in frame order at 20 fps. Collector simulation timestamps have gaps, so the 31.15 s playback compresses a 47.2 s simulation interval. |
| [teacher trial 1003102](../../outputs/experiments/2026-09-25_step5_review_videos/teacher_interleaved_trial_1003102.mp4) | Training-only SC port 1. Same actor-pulse/teacher-rescue schedule. Tier 3 = 75, full insertion, admitted. | 769 collected three-camera frames at 288×256, frame-order playback at 20 fps. Playback is not a synchronized real-time replay of simulator timestamps. |
| [trained policy trial 999001](../../outputs/experiments/2026-09-25_step5_review_videos/trained_window0p5s_trial_999001.mp4) | Two NIC cards, SC port 0, already-open development scene. Focused interleaved-correction **treatment** checkpoint controlled the whole run without teacher, true pose, supervisor, or backtracking. No insertion; final plug-port distance 0.11 m; no scored contact. | 75 existing three-camera rollout snapshots, roughly one image per second. The video does not show every command or brief contact event. |
| [trained policy trial 999003](../../outputs/experiments/2026-09-25_step5_review_videos/trained_window0p5s_trial_999003.mp4) | Three NIC cards, SC port 1, already-open development scene. Same autonomous checkpoint and conditions. No insertion; final plug-port distance 0.07 m; no scored contact. | 63 roughly 1 Hz snapshots, so fine motion cannot be resolved visually. |

The teacher videos are **not matched-scene comparisons** with the two
development videos. Their role is to show what an admitted teacher-rescued
SC insertion looks like beside examples of the trained actor's autonomous
failure. The exact scores and intervention admission are in the
[teacher pilot gate](2026-09-25-sc-body-interleaved-rescue-pilot-gate.json),
and the autonomous scores and checkpoint are in the
[matched live gate](2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json).
The scored-bag command/motion diagnosis is [separate](2026-09-25-step5-body-cadence-scored-diagnosis.json)
and used a frozen parent checkpoint, not these treatment clips.

The original JPEGs are under
`/var/tmp/chmin_aic_20260918_act/sc_body_interleaved_rescue_pilot_20260925/episodes/`
and
`/var/tmp/chmin_aic_20260918_act/shared_body_target_bc_window0p5s_treatment_20260925/eval_new_dev2_replan1/best/attempt_0001/rollout/`.
The scored raw bags are hash-verified on NAS through the pilot and matched
live gate archive manifests. All four rendered MP4s were decoded and sampled
after encoding; the three-camera contents and labels were visually checked.
