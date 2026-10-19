# Available scores

Yes—one new development evaluation is complete, but there is **no new complete 300-point sample score yet**.

| Evaluation | Result |
| --- | --- |
| Previous complete `sample_config.yaml` run, SFP + SC | **165.42/300**,0/3 passed the strict insertion gate |
| New 10-scene SFP image-based controller evaluation | **78.05/100 average**,6/10 passed the combined insertion gate |
| Current learned-BC comparison | Still running:0/4 completed episodes passed so far,8 planned |
| Backoff comparison | Not started; waiting for current comparisons |

The new controller passed **5/5 for SFP port 0 but only 1/5 for port 1**. Importantly, all 10 terminal geometry measurements look deeply inserted/aligned, yet the toolkit reports partial insertion for four port 1episodes. That disagreement needs diagnosis before claiming reliable insertion. No scored excessive force or harmful contact was reported in these 10 runs.

The new SFP result uses an **explicit vision-based controller, not the learned BC actor**, and is not directly comparable to the mixed-task 300-point sample score.
