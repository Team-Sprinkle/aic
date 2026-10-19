# Selected-port identity check: result and decision

I ran the bounded follow-up on original camera images. The good news from the earlier test still holds: when the correct SC port is requested and clearly visible, the frozen perception pipeline localizes it precisely. The new test found a separate, serious **identity** problem.

## What I tested

In two development scenes, **only SC port 0 exists**. For eight three-camera moments from each scene, I gave the same images to the frozen model twice: once requesting the real port 0, and once requesting nonexistent port 1. I also included eight clear requests for real port 1 from a third, two-port scene. The model never saw simulator geometry during inference. The scene's scored TF was used afterward to verify that port 1 truly was absent and to measure where predictions landed.

| Frozen pipeline | Correct clear requests accepted | Requests for absent port 1 accepted by mistake |
| --- | ---: | ---: |
| Current locator + crop refiner + three-camera check | **24/24** | **16/16** |
| Earlier locator trained with an absent-identity loss | 19/24 | 14/16 |

The model predicted the **visible port 0** when asked for nonexistent port 1. Its three camera estimates agreed closely and had high in-frame scores. This explains why the earlier camera-agreement result was encouraging but insufficient: three cameras can agree on the *same wrong object*. [Marked example images](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_absent_request_false_accept_gallery_v1.jpg) · [machine-readable comparison](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_absent_request_identity_gate_v1.json).

This is a deliberately invalid request used as a stress test. Normal task configs request a port that exists, so **16/16 is not a measured deployment failure rate**. It does show the planned selected-port verifier cannot yet reject a wrong identity.

## Cause and attempted repairs

The original SC perception source had **104 episode groups, all with both SC ports**. It never taught the model what a one-port scene looks like. Its original in-frame loss also ignored absent port slots. The repository's eval-like scenario generator can omit the non-target SC port, so this gap matters. I collected four targeted one-port port-1 Gazebo development resets and combined two of them with two existing one-port port-0 recordings for fitting. Four different one-port reset groups stayed out of fitting.

A 500-update repair that changed **only the SC presence output rows** still marked **334/404 absent** development-camera labels as present and hurt real-port recall on the old two-port validation groups. A more expressive frozen-image probe fit its four training groups but still marked **202/404 absent** development labels as present; adding measured TCP improved this to **149/404**, still far from safe. Neither checkpoint was promoted. [Per-group probe results](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_selected_availability_probe_v2/summary.json).

A simple rule rejecting overlapping port-0 and port-1 predictions would also reject the **real** port in a one-port scene, so I did not add it to deployment.

## Decision

The selected-port identity gate **failed**. I kept the automatic transport-to-insertion switch, insertion-policy training, RL, and final-scene evaluation disabled. The existing SFP wrong-port case and hard SC occlusion remain separate open tests. All four new Gazebo jobs and both probe runs finished; no active job was interrupted. Scripts compiled, `git diff --check` passed, and the machine summary counts were verified.

The next repair should use **spatial, task-conditioned target-identity and visibility supervision** on a broader set of one-port and two-port SC scenes, with fresh complete reset groups held out for evaluation. The stopping rule is zero absent/wrong-port accepted handoffs on those new groups while retaining at least 95% of clear valid arrivals; complete live inference p95 must remain below 300 ms. No further actor or RL training follows until that gate passes. [Full experiment record, commands, scene manifests, and artifacts](../experiments/2026-10-02-sc-absent-port-identity.md).
