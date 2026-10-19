# SC crop-identity experiment result

I trained and tested the proposed selected-port model using a 160×160 crop from each original camera image. The existing RGB locator chose each crop center from the image; simulator port coordinates were used only for training labels and later scoring. I froze four new SC reset configurations before fitting and kept the reserved final scenes sealed.

The new model **failed the promotion gate**. On 144 three-camera samples from the four new resets, the existing spatial model recognized the real port in 134 of 135 projected in-frame samples and never accepted the physically absent port. The crop model recognized the real port in only 102 of 135 and accepted the absent port in 25 of 144. It still passed four previously good saved handoffs and rejected 20 provisionally hidden examples, which shows why the independent reset check mattered.

The crop did recognize the previously missed port-0 image. But the complete verifier then measured the robot tool orientation **6.71° away** from the image-derived target orientation; the safe limit is 3°. It correctly refused the handoff. So the crop did not produce a usable insertion start.

One cause of the crop regression is clear: in a new port-1 scene, the existing locator placed the real-port and nonexistent-port crop centers only about 2–10 pixels apart. Both 160-pixel crops show essentially the same opening. The current spatial full-image model keeps enough layout information to distinguish them; the new crop model did not.

I left the experimental checkpoint off by default. No insertion actor or RL ran, and I did not claim a full live latency result after the accuracy gate failed. The next bounded repair should keep the spatial identity model and trace the port-0 transport orientation command against measured TCP orientation and image-derived port orientation, then test one correction on a matched development start. The [full experiment record](../experiments/2026-10-03-sc-original-rgb-crop-identity.md) has commands, image panels and artifacts.
