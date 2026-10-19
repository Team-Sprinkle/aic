# Project completion snapshot

Updated October 19, 2026. The demonstration milestone is complete: both SFP-to-NIC and SC-to-SC insert using **rule transport → requested-port visual handoff → explicit image-based insertion**, with force/progress recovery. The final recordings use RGB, the public task code and measured robot state; simulator target poses are excluded from runtime decisions.

| Evaluation | Result | Scope |
| --- | --- | --- |
| Released `eval_config.yaml`, selected repeat 10 | **276.34/300; 3/3 insertion awards** | Final demonstration; 1, 2 and 3 NIC cards across the trials |
| Same frozen configuration, additional repeat batch | **15/19 complete passes; 53/57 trial insertions** | Four intermittent failures remain; repeat 20 was stopped at the user's request |
| Exact public `sample_config.yaml`, latest recording | **287.90/300; 3/3 insertions** | Simpler scenes: 1, 1 and 0 NIC cards |
| Earlier selected public-sample controller | **290.19/300** | Earlier configuration, not the final released-evaluation recording |

The released configuration was inspected and used for repair after publication. These are local results on exposed scenes; the separate internal reserved final scenes remain sealed. The repeat failures were initial force-reference timeout, SFP visual-acquisition timeout, early SC blockage and missing SC marker acquisition.

- [Final videos, GIFs and provenance](media/final_evaluation/README.md)
- [Short project writeup](robotic_cable_insertion_project.md)
- [Experiments: what worked, what failed, and why](EXPERIMENTS.md)
- [Technical completion notes and historical detail](answers/2026-10-19-final-project-completion-notes.md)
- [One future-work plan](FUTURE_WORK.md) · [Rootless reproduction](LOCAL_WORKFLOW.md)

`STATUS.md` remains a short entry point for existing links. Detailed experiment history lives in the linked records rather than accumulating here.
