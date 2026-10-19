# Proposed shared coarse history model

## Short answer

Yes: use **one task-conditioned, three-camera history backbone**, followed by (1) a **requested-port crop proposal head** and (2) a **proposal-conditioned stop/accept head**. The heads should share the visual understanding of which numbered port was requested. They should not be independent classifiers: the second head must inspect what the first head proposed. The current images determine the crop and whether to pause now; older images establish route, card identity and whether an apparent port was previously identified. This is a proposed design, not a measured improvement or a deployed handoff.

The coarse decision is *“pause transport and send these crops to the precise refiner”*. It is not yet *“start insertion.”* The refiner must still estimate a current precise pose, and the final handoff must check pose, orientation, force and stability. There is no separate late camera-check network in this proposal: its requested-port identity and safe-proposal decision move into the coarse model.

## Concrete model

```text
public task code: SC/SFP, requested card slot, requested port
measured TCP poses + wrist motion + current force
current synchronized RGB from three cameras
distance-sampled RGB history: initially 30 mm, up to 12 triplets
                     |
        shared spatial image encoder, shared across cameras and times
                     |
       compact spatial history + relative measured camera/TCP motion
                     |
          task-conditioned history attention
                     |
       requested-port query applied to CURRENT spatial feature maps
                /                                   \
  crop head: per-camera heatmap/center       stop/accept head:
  and current-view evidence                  examines proposed crop features,
                                             identity evidence, robot state,
                                             geometric agreement and force
                \                                   /
           either CONTINUE/ABSTAIN or PAUSE + crops for refiner
```

I would initialize the image encoder from the existing frozen ResNet-18 locator and use a small 2-layer, roughly 256-dimensional attention module. Keep a **small spatial grid** for each saved view, not just a global pooled vector; the previous pooled and 4×5 history variants confused neighboring SFP ports. Give each grid its camera identity, image position, requested card/port code, and measured relative robot pose. Use learned attention to gather evidence about the one requested port. Decode onto the **latest** per-camera feature maps to produce a heatmap or distribution over the crop center in each current full-resolution image. The crop itself comes from the original image, before global resize. Historical frames are context; they are never substituted as the current crop. Cache old image features and encode only the newest triplet when it changes, then profile full p50/p95/p99 latency; the entire path must remain below 300 ms p95.

The stop head receives the task query and history summary **plus features sampled around the crop head's proposed current locations** and current measured robot state. It must be allowed to say “no trustworthy target / keep transporting” when the requested face is hidden, the proposal appears to be a neighboring port, the approach geometry or orientation is unsuitable, or force/contact is unsafe. It can produce one accept probability for the *whole three-camera proposal*, with per-camera evidence as intermediate outputs. There is no need for a separately supervised, vaguely defined “pose confidence” number. We can calibrate the accept decision directly against independently defined outcomes. A camera may provide strong evidence while another is occluded; the rule should test the combination, rather than demand three visible openings in every case. A consistency check can help, but prior replays showed that multiple cameras can agree geometrically on the **wrong** port.

The history is anchored by **distance traveled**, not seconds: the ten-recording audit provisionally favored 30 mm and 12 viewpoints. Recompute the current image feature each decision while preserving old route anchors during a stall. Record all three cameras together, with relative TCP translation and rotation. In-place wrist rotations and major visual/contact changes may need extra anchors; determine those triggers from recordings. The 30 mm/12 choice has not passed autonomous identity or latency gates. If transport never supplied a view identifying the numbered slot, the network must abstain; history cannot recover evidence that never existed.

This design borrows the **task query + spatial decoding** idea from [DETR](https://arxiv.org/abs/2005.12872), and the idea of a target query carrying identity through time while separately predicting visibility from [TAPTR](https://arxiv.org/abs/2403.13042). [RoboTAP](https://arxiv.org/abs/2308.15975) illustrates why tracked visual points can help manipulation. Those papers do not solve our numbered, sometimes empty NIC slots, three moving cameras, or safe 10 mm/3° handoff criterion; those are task-specific labels and tests we must supply. A full dense tracker is not the first choice under 300 ms because we need one requested port and a reject decision, not thousands of tracks.

## What training data exist

There is enough existing data for a **bounded first fit**, but not enough correctly labeled data to claim a reliable accept decision.

| Source | Useful for | Gap |
| --- | --- | --- |
| 50 balanced, valid-command SFP teacher train episodes + 10 disjoint development episodes | Current RGB, task code, TCP route, target projection and geometric approach labels across all ten SFP requests | Teacher views are often privileged and favorable; many early/off-route negatives are absent. The target projection is not true visual visibility. |
| 10 additional existing five-card training episodes and ten fresh nonfinal teacher-view stress recordings (581 synchronized rows in the latter) | Harder card layouts, spatial history, wrong-port/crop audits | Previously inspected stress recordings cannot become a new promotion split; projected target is often in all image bounds, so entry/exit coverage is weak. |
| 268 older verified SFP episodes | Extra image/geometry examples | 258/268 use card 0/port 1; sampling without correction would restore bias. |
| 104 successful SC route episodes and other recorded SC diagnostics | SC port 0/1 identity, close views, wrong-route examples | Need an exact causal camera/state/geometry join and true-visibility/acceptance review; SC and SFP should be reported separately. |
| Recorded autonomous wrong-port, route-miss, side-view and occlusion diagnostics | Critical reject examples: plausible other port, wrong approach, partial card row, cable/tool blocking | Must curate exact frames and independent labels. Some earlier candidate lists filtered by simulator presence, which would leak privilege if reused. |
| 180 scored-successful agent/VLM SFP replays | Scene variety and images | Their action arrays contain invalid zeros; do not use as direct BC action labels. |

The existing projected *in-frame* flag cannot train “I can identify the requested opening.” It only says a point falls inside image bounds. Our prior SFP auxiliary depth attempts failed, so true occlusion or human-identifiability labels need targeted curation. The old requested-history crop fit used 50 episodes, then 60 with extra five-card examples; its best inspected raw stress replay still put 173/572 proposed centers nearer another port, and a strict geometry filter retained only 3/10 early episodes. Repeating that fit with a larger transformer and the same labels would be a weak experiment.

## Label curation I would do before training

1. **Freeze episode-grouped splits first.** Never split individual frames from one reset across train and test. Keep the four final IK configurations sealed, and reserve genuinely new nonfinal starts for promotion. Keep the ten already inspected stress scenes as diagnosis only.
2. **For each current three-camera observation and requested task code**, record the true requested port identity and original-image location from simulator geometry *for supervision/scoring only*. Annotate whether the requested opening is actually identifiable in each image, partly occluded, outside the view, or confused with another port. Validate any renderer/depth-derived mask by inspecting the RGB; use human review for ambiguous cases. If invisible, mask the crop-location loss but retain a reject label. Do not call a mere projection “visible.”
3. **Create hard negatives on the same images.** Change the requested code to a neighboring occupied port, an empty SFP slot, or an offscreen/occluded port; include plausible wrong crops from the existing models. Preserve the correct code and neighboring ports in the image so the model must use identity, not just detect any aperture. For SFP, label absolute rail slot identity rather than rank among visible cards.
4. **Label the proposed decision from independent facts.** A positive coarse accept requires an identifiable requested opening, a crop that covers *that* opening with margin, and a safe observed approach posture/force. The broad transport proximity/orientation can be computed from simulator geometry for offline labels, but true geometry never enters runtime. Since acceptability depends on the particular proposed crop, train the verifier on correct, jittered, neighboring-port and offscreen proposals, scored against those independent labels. This is not generating a confidence label from the current model's confidence. Keep coarse acceptance distinct from the stricter precise-refiner insertion handoff.
5. **Use actual autonomous transport frames**, especially where the actor first drifts at ~4 seconds, stalls, shows only part of the row, or approaches the wrong port. Existing teacher trajectories alone do not supply the deployed input distribution. Sample balanced SFP/SC, requested port, card count, empty-slot pattern, distance, orientation, occlusion and accept/reject outcome. Curate rather than collect a large undirected batch; only collect new nonfinal scenes to fill a measured gap.

Train crop localization with a heatmap/location loss on identifiable positive views; train the accept head with a binary or three-way (continue / pause-and-inspect / invalid) classification loss over the *proposed* crop and safe-state labels. Explicitly upweight false accepts during checkpoint selection. Test three ablations under identical splits and update budgets: current frame only; 30 mm distance history with the crop head; then the same shared backbone plus proposal-conditioned accept head. Also compare against the frozen coarse and separate-verifier baselines. Report wrong-port accepts, missed safe proposals, requested-target crop recall by card/port and occlusion, early acquisition per complete episode, SC and SFP separately, and complete live latency. A low aggregate pixel error is not a pass if it hides even a few confident wrong-port stops.

**Decision gate:** first prove reliable identity and crop coverage on new episode-grouped development starts with zero observed wrong-port accepts under the frozen threshold, then prove the coarse stop does not falsely accept an unsafe orientation/route, and only then connect the precise refiner. No model or data was trained or collected for this answer.
