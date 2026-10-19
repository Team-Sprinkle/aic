I found concrete issues that **change the earlier conclusion**. The actor is learning more of the SC route than our previous evaluation suggested. The strongest next direction is to repair **terminal training data and stopping/handoff**, before enlarging the model.

I inspected code, all 104 SC transport episodes’ label coverage, saved rollout traces, representative video frames, and relevant papers. **No training, simulation, or GPU inference was started. Existing jobs were left alone.**

**1. We were asking transport to reach somewhere beyond its training data.**

The successful SC teacher episodes contain several stages. Our transport dataset includes only the first portion; later alignment, approach and insertion are separated out.

- The transport labels end with the TCP about **168 mm axially from the opening**, and the plug tip about **148 mm away**.
- We evaluated the transport actor against a TCP band **70–100 mm away**.
- **None of the 104 teacher transport prefixes reaches that band either.**

So an actor that perfectly copies these transport labels would still fail that test. We need to define who performs the missing approach: extend transport using the existing recorded approach segment, or validate that the downstream actor can start farther away. We cannot simply relax the threshold and claim successful handoff.

**2. Some actors reach the place we taught them to reach—then continue moving.**

I rescored existing rollouts against each teacher’s actual transport endpoint:

| Actor | Median closest error to taught endpoint | Passed within 20 mm and 5° |
| --- | ---: | ---: |
| Dense history, 27 requested observations | 22.7 mm | 3/6 |
| Finer history, 38 requested observations | **13.2 mm** | **5/6** |
| Intentional overfit, second stage | 15.6 mm | 4/6 |

These are the same six development scenes. **They are retrospective proximity measurements, not successful autonomous stops.** The finer-history actor looks more promising under this more relevant transport metric. Extra training accompanied the history change, so we cannot attribute the improvement entirely to history.

The diffusion actor currently has **no trained stopping head**: its runtime explicitly sets handoff probability to zero. Consequently, it can pass near the teacher endpoint and then drift away. The final overfit and cadence-adjusted models also approached their three tested endpoints reasonably closely before leaving.

**3. The longer action horizon accidentally excludes important late examples.**

In the 100 ms cache used by the history experiments, we require all 12 future action targets before admitting a training row. Combined with pause cleanup and strict timestamp/index checks, this causes:

- **29/83 training episodes** to have no surviving examples carrying the final return-stage label.
- Only **104 of 388** cached final-stage training rows to survive.
- **All 61 cached handoff-positive rows** to disappear from that valid-action subset.

One concrete problem: the builder rejects repeated command indices. But when a teacher command is a few milliseconds late, repeating the preceding target is the correct recorded behavior—not corrupt data.

The repair is to construct future targets from the original timestamped commands, keep observation history intact, and downweight pauses independently. Terminal observations need their own stopping labels and valid-slot masks. More training cannot recover supervision that was excluded.

**4. “Wrong route after 6–7 seconds” was too strong.**

That measurement compared actor and teacher positions at the same simulator time. They have different startup times, speeds and pauses.

Comparing their geometric paths instead shows substantial agreement along the bypass, followed by later deviations. This does not eliminate genuine route errors, but it changes where we should investigate. [The new path plots show the distinction](</home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/path_alignment.png>).

**My recommended order is:**

1. **Fix the transport-to-insertion boundary.** Check saved endpoint images with coarse perception/refinement. Determine where perception and the insertion actor actually become usable. Assign the intervening approach to one component explicitly.
2. **Repair terminal and action-label construction.** Reuse original recordings; retain final return, valid held commands, and stopping examples. Audit both ports and different card counts before training.
3. **Check execution separately from learning.** In a later bounded test, replay teacher targets through the actor’s exact chunk executor. This distinguishes controller/timing problems from prediction problems.
4. **Run one controlled retraining comparison**, keeping the architecture fixed and adding a properly supervised stop/proposal head. Measure stable arrival, wrong-port stops and visual handoff—not just action error or closest distance. Proposed initial budget: one GPU, up to four hours per arm, with an explicit plateau rule; then six new development scenes if the reused diagnostic scenes improve.
5. **Then test better spatial vision if needed.** Compare ResNet spatial features against DINOv2-S patch features under the same repaired pipeline. Our current globally pooled encoder compresses spatial layout. The original Diffusion Policy’s end-to-end visual experiments used spatial softmax, according to its [author’s clarification](https://github.com/real-stanford/diffusion_policy/issues/9). The earlier small linear DINO probe does **not** settle which encoder is better for this policy.
6. **Collect targeted corrective demonstrations only for remaining failures.** Choose actual route/progress errors rather than same-clock differences. Broader SC orientation coverage and SFP regression testing follow before integrated deployment.

Inference gaps remain a secondary concern. [Real-Time Chunking](https://arxiv.org/abs/2506.07339) provides a relevant approach, but adapting it requires handling our changing TCP action frame correctly. I would test executor parity before implementing it.

**My conclusion:** keep diffusion and the finer-history checkpoint as candidates. We have not demonstrated that capacity is the main limitation. We have demonstrated a mismatched endpoint, missing terminal supervision, and absent stopping behavior. Fix those first, then judge the architecture again.

Saved this answer [here](2026-10-13-transport-root-cause-and-next-directions.md). The [detailed audit and bounded experiment plan](../experiments/2026-10-13-transport-independent-root-cause-review.md) includes exact artifacts, code findings, research references and limitations. I also updated the main status, dataset and handoff-plan documents. Final scenes remain sealed.
