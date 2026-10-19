# Live progress terminal

Run this in your Ubuntu terminal to refresh progress every five seconds:

```bash
watch -n 5 python3 /data1/chmin/yj/ws_aic/src/aic/scripts/show_insertion_repair_progress.py
```

At 02:04 UTC:

- **Training:** one SC spatial-pose fit remains running, at approximately12,800updates. Other listed fits reached validation plateaus; the completed SC pose candidates failed their accuracy gates.
- **Fresh SFP controller evaluation:**8/10episodes scored,5full insertions so far. This is an explicit image-based controller, not BC.
- **BC comparison:**1/8episodes scored,0full insertions so far.
- **Backoff comparison:** waiting for the current training and evaluations to finish.

These are partial results. The dashboard is read-only. Press Ctrl+C to close it without stopping experiments.
