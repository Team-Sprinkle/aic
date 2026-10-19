# Geometric handoff repair update

I followed the agreed model-improvement direction. A code audit revealed an earlier SC estimator that reads the board's visible edges to infer port orientation, so I reused and tested it instead of training another corner detector. On **87 near-port image triplets from four separate development resets**, its orientation error fell from **4.434° to 0.056° p95**. It also retained five valid saved transport stops and rejected the one known wrong-posture stop. This depends on the SC board's fixed mounting geometry and visible straight edges.

For SFP, I added a cross-camera check to the existing native-image candidate model. At least two cameras must recognize the requested port, and all three proposed pixel locations must fit one 3D point. On the two held-out five-episode subsets, it accepted **85/87** and **159/159** correct constructed proposals, with **zero wrong accepts**; the prior all-camera vote had one wrong accept in each subset. The 25-pixel geometry threshold was chosen using the 19 fitting episodes.

The SC perception path measured **100.27 ms p95 offline**, including native image decode and board-edge search, but excluding ROS, actors and command handling. These tests used saved development images, and the SFP proposals were constructed offline. They do not establish true visual visibility or successful full insertion. The automatic transport-to-insertion switch, actor retraining and RL remain disabled; final scenes remain sealed.

Next I would run a bounded **shadow handoff** on up to four new SC and four new SFP development resets. Transport would propose a pause, the verifier would accept or reject using only observations, and a rejected proposal would resume safely. I would inspect requested-port visibility, test coherent wrong-port cases, and measure complete live latency and force. The first false accepted handoff stops this test and identifies which perception or transport component to repair before actor training.

Details, commands and artifacts: [experiment record](../experiments/2026-10-03-geometric-handoff-repair.md), [machine summary](../experiments/2026-10-03-geometric-handoff-repair.summary.json).
