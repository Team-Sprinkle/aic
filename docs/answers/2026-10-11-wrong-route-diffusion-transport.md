# Wrong route diagnosis and diffusion transport result

The old transport actor sometimes turned toward the wrong card in a five-card SFP-to-NIC task. Its commands initially pointed roughly along the successful teacher path, but a small displacement grew after about five seconds. The older training cache also heavily favored one card/port request, so the model had weak practice distinguishing the other routes. This is evidence for weak route grounding and closed-loop drift, although it does not prove either is the sole cause.

I trained a larger route-aware diffusion actor on 46,761 training rows spanning SFP-to-NIC and SC-to-SC, with balanced SFP target requests. A frozen three-camera ResNet-18 provides image features. A 14.29-million-parameter trainable Transformer predicts where the robot should move over the next second and generates four complete TCP-relative actions. With the frozen image backbone, the total is about 25.47 million parameters. The future route displacement is a **training label**, not privileged information supplied at deployment.

The first noise-prediction run produced unsafe actions because the last diffusion noise level made its calculation numerically unstable. I kept that failed run, changed the model to velocity prediction, and trained it for 12,000 updates. I then fine tuned for up to 4,000 updates on 108 training-scene pairs with valid future labels that show the same image but different requested ports. No final scenes were used.

On held-out teacher observations, the selected fine-tuned model has first-action translation p95 errors of **2.18 mm** in early SFP motion, **3.26 mm** in other SFP motion, and **2.96 mm** for SC. On 13 separable request pairs across six held-out scenes it changed the route in the teacher's direction every time; the median size of its change was **71%** of the teacher's change. Zeroing its predicted route goal made the action error worse, so the actor is using that prediction.

The autonomous result is weaker. On the same previously used five-card Gazebo scene, the old actors missed the requested port laterally by about **56–69 mm** in the relevant axial band. The diffusion actor reduced this to about **18–20 mm**, but **neither of two full rollouts reached the safe buffer or inserted**. The official scorer said “No contact detected”; failure was inaccurate route execution. Early actions matched the teacher direction, then measured position and orientation drift accumulated. This checkpoint is **not promoted**.

The first live run exceeded the 300 ms inference limit. Later auditing found ambiguous physical GPU mapping in the purported split-GPU profile, and a ground-truth TF flood was the principal slowdown in privileged takeover diagnostics. Delaying the diagnostic TF listener restored representative proposal cadence; subsequent normal autonomous runs on physically separate GPUs measured **253–267 ms p95**. Latency did not repair the route.

One privileged diagnostic handed control to CheatCode after six seconds of this diffusion actor on the same development scene. The teacher recovered and fully inserted, with only a small sampled cable–board point force. This shows recovery is possible from **one** such off-route state. That development episode is not a training source or autonomous success.

That proposed correction experiment was completed and failed its autonomous gate; a further targeted early card1/port1 correction also failed. See the [complete later result](2026-10-11-route-choice-diffusion-result.md). Promotion still requires at least two safe buffered arrivals in three starts, acceptable contact, and complete inference below 300 ms. Handoff, insertion and RL remain gated.

See the [full experiment record](../experiments/2026-10-11-route-aware-diffusion-transport.md), [machine decision](../../outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/decision.json), and [measured route comparison](../../outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/fivecard_route_comparison.png).

## Final follow-up

The first corrective continuation described above did **not** repair autonomous transport. A targeted early card1/port1 continuation also fitted its teacher actions but failed the live safe-buffer gate. The earlier statement that GPU splitting alone fixed latency was too strong; a ground-truth TF flood was the main diagnostic slowdown. See the [complete later result](2026-10-11-route-choice-diffusion-result.md) and the [updated experiment record](../experiments/2026-10-11-route-aware-diffusion-transport.md).
