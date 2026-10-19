# V56 on the released official evaluation

I ran the frozen best configuration once on the upstream `eval_config.yaml` with its three trials in original order. **Score: 98.83/300; 1/3 physical insertions.**

| Round | Result | Main reason |
| --- | --- | --- |
| SFP to NIC card2 | 96.83/100; inserted | Measured46.763mm depth |
| SFP to NIC card4 | 1/100; failed | Cable contacted intervening card3; force guard stopped transport before port acquisition |
| SC, three NIC obstacle cards | 1/100; failed | Transport blockage led to a long measured-path retreat; controller wall-clock guard ended the retry before port acquisition |

The earlier**290.19/300** used the easier default `sample_config.yaml`. It does not generalize to this released qualification configuration. The failed rounds never reached the close insertion controller; transport/recovery is the measured gap. Every used inference stage passed its300ms p95 limit. No training or evaluation-based tuning was done, and the reserved internal final split stayed sealed.

[Full metrics, root causes, commands and videos](../experiments/2026-10-18-frozen-V56-official-evaluation.md).
