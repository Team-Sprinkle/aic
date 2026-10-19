# How many ports does coarse perception predict?

The current **joint coarse locator** predicts **12 possible port locations in each camera image**:

| Task family | Possible outputs per camera |
| --- | ---: |
| SC-to-SC | 2 ports |
| SFP-to-NIC | 5 possible card slots × 2 ports = 10 |
| Total | **12** |

These are 12 output channels in one network, not 12 separate networks or 12 independently known 3D poses. For each of the center, left, and right images, it outputs a 2D location heatmap and an in-image score for every possible port. Matching 2D predictions across cameras can then be triangulated into a 3D estimate. A requested task code tells the downstream logic which port matters; the joint network still emits all possible port channels. Some card slots may be empty, so their channels can still produce false predictions. The in-image score is not a verified measure of true visual visibility.

There is also a **selected-port model** that predicts only the *requested* port. It has been more cautious about accepting the wrong port on the tested scenes, but it often abstains, particularly for middle SFP cards. The joint model offers more candidates and better coverage, with a serious wrong-port risk.

The present coarse locator uses the **current three camera images**, without camera history. History is a sensible next input when a close view hides the card-row context that was visible earlier. I would phrase it as a strong, testable hypothesis for those cases, rather than a confirmed universal requirement: one five-card failure already showed the whole row in the current view, and the simple history tracker did not acquire any new early episodes. The tracker maintained an identity *after* a trusted selected-port estimate existed; it did not solve initial port identity.

During this code check I found that an earlier **offline crop audit** enumerated only ports marked physically present in simulator labels, even though the network predicts all ten possible SFP ports. I fixed the candidate enumeration and replayed the same frozen models and ten scenes. With all ten channels, the four-model union made **47 wrong accepted predictions** instead of the earlier reported 46; the selected-anchor history tracker gained **53** later accepted frames instead of 52, with no gain in early episode acquisition. This does not change the decision: autonomous SFP handoff remains disabled. The [corrected experiment record](../experiments/2026-10-04-sfp-fresh-anchor-shadow.md#corrected-all-channel-crop-replay-october-4) preserves the commands and artifacts.
