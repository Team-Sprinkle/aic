# Did coarse camera training use enough positive and negative examples?

**Partly, but not in the complete way you describe. Your concern is valid.** We generated many *geometric* positives and negatives from each recorded image, and later tried same-image alternate-request training. We did **not** yet build reliable labels for “the requested port is genuinely visible and identifiable now” or “the robot is near enough and safely oriented to pause.” Those missing distinctions are central to a trustworthy camera check.

## What the training actually did

The main SFP original-image row locator used **66 complete training episodes**, providing **1,739 three-camera image timepoints**; ten separate episode groups supplied 136 development timepoints. For each image, simulator scene geometry supplied the position of every physically installed card/port. We projected those openings into each camera. The model's ten SFP port channels were trained to locate every installed opening that projected inside the picture, with extra weight on the episode's requested port. A missing card or a projected-out-of-frame opening got a zero in-frame target for that channel. The model received RGB and camera identity; geometry supplied training labels and later scoring, not inference input.

Across the 1,739 training timepoints, installed-port/camera pairs yielded **42,667 projected-in-frame labels** and **4,697 projected-out-of-frame labels**. Considering all ten possible port channels, physically absent slots added about **4,806 more zero labels**. These counts are *correlated labels from the same 1,739 scenes*, not tens of thousands of independent episodes. They also show that easy projected-in-frame positives were much more common than out-of-frame negatives.

The requested-port history crop model also sometimes changed the requested card/port code on the **same image** during training (40% chance), so it was not limited to the route's original request. A later explicit counterfactual dataset repeated each of the **1,739 training image triplets** with every installed port as the request, making **15,788 request-image rows**. Its observation-only crop generator produced 8,832 crops; the requested opening was inside all three crops in 7,048. The matched crop-model continuation did **not** improve the gate: it accepted two wrong ports on grouped development and acquired only 6/10 early episodes. Thus we did try alternate-request negatives, but the resulting model was still unreliable.

## What was missing

The label **“projects inside the camera image” is not “I can see and identify it.”** A port can be inside the mathematical field of view but hidden by the plug, gripper housing, cable or another card. The current in-frame head can be confident on such cases. We lacked enough independently checked clear, obscured and ambiguous RGB labels to train a reliable visibility/identity or safe-stop head.

We also did not give the coarse crop model your proposed **near-versus-far pause label**. A far port can be perfectly visible: it should be a **positive for localization** but a **negative for pausing now**. A nearby requested port that is covered or the robot approaches at the wrong angle may also be a negative for pausing. The previous learned stop head used separate teacher-arrival labels and reached only 2/10 timely held-out episodes. These were not a well-curated set of same-image/request hard negatives for the camera check.

There is another subtlety: on one image, a visible neighboring port is a **positive location for its own port identity** and a **hard negative for the currently requested identity**. Merely labeling it “negative” without the task code would teach the wrong thing. Physically absent requested slots, ports outside the view, visible neighbors and far-but-visible requested ports should be kept as distinct cases.

## How I would rebuild this data

For each saved three-camera timepoint, enumerate every possible requested card/port code, while splitting train and validation by **complete reset configuration**, not by frames:

| Question | Label and examples |
| --- | --- |
| Where is this specific requested port? | Projected pixel/3D geometry for training; add explicit absent/unknown when the card or port is not physically present. |
| Is it identifiable from these RGB views? | Reviewed clear, partly covered, fully covered, clipped-board and ambiguous examples. Use geometry as a starting point for review, not as the visibility answer. |
| Is this request the visible neighbor instead? | Same image with alternate card/port codes, especially adjacent ports and sparse-card gaps. |
| Is this a good place to pause and inspect? | Requested-port relative distance **plus** robot orientation, route, force/contact and camera quality. Far-but-visible is negative for pause, positive for localization. |

Then sample these categories deliberately so easy positives do not dominate, and report results per **episode and request**, including accepted wrong ports, abstentions, early acquisitions and safe-pause timing. The first bounded comparison should reuse existing recorded images and actions. New Gazebo episodes are justified only for a demonstrated missing category or independent development check. The same simulator geometry can supervise training and score a rollout afterward; it must never choose a crop or port at autonomous inference.

So the short answer is: **we did some of your suggested all-port and same-image counterfactual construction, but we did not sufficiently construct the visibility, proximity and safe-pause negatives needed for the complete coarse camera check.** More frames from the same rollout alone will not repair that label gap or add much reset diversity.

Evidence: [coarse experiment and counterfactual result](../experiments/2026-10-04-coarse-shared-history-execution.md), [row locator training code](../../scripts/train_sfp_native_row_locator.py), [history crop training code](../../scripts/train_distance_coarse_crop.py), and [current spatial diagnosis](../experiments/2026-10-05-spatial-transport-prerequisite.md).
