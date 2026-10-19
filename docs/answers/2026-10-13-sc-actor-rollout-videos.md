# SC actor rollout videos

These are autonomous actor rollouts in Gazebo, recorded from five synchronized cameras at 10 fps. They are not teacher demonstrations. All comparisons labeled `llb_0009` use the same five-card development start. None of these rollouts achieved full insertion.

- [Baseline 50 ms actor, five cards, port 1](</home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Long-history 100 ms actor, five cards, port 1](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Denser 70 s history actor, five cards, port 1](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_100ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Intentional-fit actor, five cards, port 1](</home/nas/chmin/aic_archives/2026-10-13_sc_overfit_50ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Long-history 100 ms actor, one card, port 1](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0005/five_cameras_10fps_active.mp4>)

The matched five-card videos make it easier to compare route behavior visually. The one-card video shows a simpler layout. See [the SC context and overfit experiment](../experiments/2026-10-13-sc-pauses-context-and-overfit.md) for metrics and interpretation.

These new viewing copies end at the actor's last command. The original videos continued recording while Gazebo deleted the board and cable during reset, which made their final frames misleading. The scored bags and original recordings remain preserved.
