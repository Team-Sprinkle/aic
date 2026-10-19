Yes—that is exactly what the official `sample_config.yaml` specifies. I verified the configuration used for these videos.

| Trial | Task | Cards in `sample_config.yaml` | Cards in released `eval_config.yaml` |
| --- | --- | ---: | ---: |
| 1 | SFP to NIC | 1 | 1 |
| 2 | SFP to NIC | 1 | 2 |
| 3 | SC to SC | 0 | 3 |

The videos I just shared are the **sample configuration**, which scored **287.90/300**. The **276.34/300** best score came from the released evaluation configuration, including its three-card SC scene. These are different tests.

The sample videos therefore do not demonstrate the difficult SC detour around cards. The earlier released-evaluation five-camera recording shows that scenario: `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/repeat_1/rollout/five_cameras_10fps_grid.mp4`.
