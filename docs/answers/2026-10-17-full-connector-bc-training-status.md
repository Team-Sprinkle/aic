# Current status

**The full-connector BC training has finished.** It reached a validation plateau after **13,400 updates**; the selected checkpoint is from update7,400. It was not stopped by the time limit. Training used420episodes, with79held out.

## What improved?

On held-out teacher observations, compared with the matched previous H4 model:

| Metric, p95 | SC | SFP |
| --- | --- | --- |
| Near-port orientation error versus expert command |2.28° → **0.77°**|2.28° → **0.28°**|
| First TCP target position error versus expert |7.98 → **10.00mm**|11.33 → **8.55mm**|

**Orientation improved clearly, but position results are mixed:** SFP improved; SC worsened. These numbers measure imitation accuracy, not actual insertion success.

The separately trained image-based orientation estimators remain at angular p95 **0.89°SC /0.49°SFP**.

## What is still pending?

**The new actor has not had an autonomous Gazebo rollout yet.** The automated job finished at the runtime-integration gate. No training or evaluation job is currently running.

Next is to integrate stable image-based grasp-orientation acquisition and the full-pose command conversion, verify them, then run controlled SC/SFP rollouts with recovery disabled to isolate the actor. Only those tests can show whether insertion or oscillation improved.

The previous result remains **0/14strictfullinsertions**. No new RL has run, and reserved final scenes remain sealed.

Saved status: [full-connector BC status](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-17-full-connector-bc-training-status.md).
