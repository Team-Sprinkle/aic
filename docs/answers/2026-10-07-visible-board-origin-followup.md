# Visible board-origin follow-up

I continued the board-landmark investigation using existing nonfinal SFP images. The result is that a fixed board point may help the robot orient itself broadly, but **it cannot by itself tell the robot which exact opening to approach** in the current randomized scenes.

First I checked the saved “board corner” labels. They are corners of a collision box, not exact corners of the visible board mesh. In a fixed 30-panel single-reviewer image packet, none was confidently an exact visible physical corner; a simple image-contour method clearly picked the intended corner in only 4/30 panels. I stopped that label/model route.

The actual rendered board asset contains two visible lower-corner screws. Their projected positions differ from the old corner labels by **19.7 pixels median**. In a first 30-panel review, I judged 21 screw projections usable, four uncertain and five hidden. But an early, 40-panel check across all ten development groups showed the screw on one side is often blocked by installed cards. **No group had both screws confidently visible in the two required cameras** under my conservative single-reviewer labels. I did not train a two-screw detector.

I then tested whether one perfectly located screw would be enough. This was a **privileged upper-bound calculation**, using the true screw position only for diagnosis, a layout prior learned from 70 training groups, and ten separate development groups:

| Inputs to the diagnostic | Requested-opening error, median / p95 | Wrong installed opening was closer |
| --- | ---: | ---: |
| True screw + train-average board yaw | 22.62 / 39.59 mm | 3/10 |
| True screw + true scene yaw | 15.02 / 22.18 mm | 2/10 |
| Train-average screw location + train-average yaw | 37.53 / 63.83 mm | 5/10 |

Even the unrealistically helpful **true-yaw** control missed the requested opening by 15 mm median. The card/port position relative to the board changes across resets. A board origin can help with coarse slot order, but the final selected-card and selected-port location still needs local visual evidence.

**Decision:** do not spend GPU time fitting a screw detector as a stand-alone shortcut to handoff. Keep automatic handoff, insertion and RL disabled. The independent review of the full public RGB/history/TCP observation is still pending; that review should determine whether a board cue is useful as *context* for selected-card identity. Separately, SFP transport's 0/3 safe-arrival gap remains.

The image reviews above are mine alone and are not independent visibility ground truth. No final scenes or new Gazebo episodes were used.

[Full experiment, images, commands and per-scene metrics](../experiments/2026-10-07-sfp-visible-board-origin-audit.md) · [machine summary](../experiments/2026-10-07-sfp-visible-board-origin-audit.summary.json).
