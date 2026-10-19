# October 3: spatial SC selected-port identity continuation

I continued the approved perception repair. The previous system could locate a visible SC port precisely but accepted a request for the *other, physically absent* SC port. Its pooled and last-layer presence fixes failed on different one-port scenes.

I collected 12 new **one-port SC perception reset groups** in rootless Gazebo: eight for fitting, four held out for development, with both requested ports and varied board, rail and card arrangements. They contain native three-camera RGB and measured robot state. They are **not** expert insertion trajectories. No reserved final scene was used.

I trained two small heads on the same frozen RGB backbone, data, 800 updates and fixed 0.5 threshold. Both receive the requested port code, camera identity and measured TCP. The control pools the whole image into one vector. The new head keeps the image's spatial feature grid so it can use *where* the port appears. Neither receives simulator port geometry at inference.

| Four new held-out SC reset groups | Pooled control | Spatial head |
| --- | ---: | ---: |
| Absent-port requests accepted with **all three cameras** | 32/109 | **0/109** |
| Real port retained when projected into all three cameras | 34/100 | **96/100** |

This is a promising **SC identity** result, not an insertion result. The spatial head still falsely called an absent port present in **9/304 individual camera requests** in the presampled development set. Requiring two of three cameras accepted 2/109 absent triplets; the fixed all-three rule did not. Some real-port “misses” were images where the port's geometric projection was inside the image but the gripper hid it. The training label measured *in-frame projection*, not true visual identifiability.

I recovered measured TCP from the old scored bags and replayed the frozen spatial head on 20 previously reviewed, provisionally hidden SC triplets. The identity head **alone accepted 4/20**. The existing native keypoint and three-camera reprojection check accepted 0/20, so their conjunction also accepted 0/20. This small review is not a certified occlusion test. It shows why identity and visibility need separate checks.

The three-camera coarse locator, identity head and native crop refiner took **11.19 ms p95** in a 100-iteration, preloaded-image GPU 1 component benchmark. ROS image delivery, transport and insertion actors, triangulation and controller commands were excluded, so the required complete live p95 <300 ms is still unmeasured.

**Decision:** keep automatic transport-to-insertion handoff and RL disabled. The next bounded check is independently reviewed SC occlusion and frozen wrong-route replay, followed by the saved SFP wrong-port case. Only then should we integrate the visual verifier and measure complete live latency. No current result establishes autonomous full insertion or performance on final scenes.

The [experiment record](../experiments/2026-10-02-sc-absent-port-identity.md#spatial-selected-port-continuation-on-new-reset-groups) lists commands, scene manifests, checkpoints, per-group metrics, image-level errors, latency, and the next gate. The artifact root is `/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/`, especially `sc_spatial_identity_v1/` and `sc_spatial_identity_probe_v2/`.
