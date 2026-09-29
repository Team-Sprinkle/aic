**Yes, `git add -A` is reasonable for the current working tree**, provided you review the staged list before committing. My dry run found 468 paths to stage: 18 modified tracked files and 450 new files totaling about 7 MB. The new files are code, configs, Markdown, JSON experiment records, and one 249 KB plot. A basic credential-pattern scan of those new files found no hits.

The current `.gitignore` excludes `outputs/`, new files under `artifacts/`, MP4s, MCAPs, bags, Parquet, and local environments. Thus **the new rollout videos, datasets, and model checkpoints will not be staged** by ordinary `git add -A`. Avoid `git add -f` on those paths.

One historical exception: **32 videos (about 84 MB) are already tracked** under `artifacts/` and one older `outputs/experiments/` folder. Ignore rules do not untrack existing files. None of these videos is modified now, so this staging operation adds no video changes. Removing the already tracked videos would be a separate cleanup that could break documentation links; I would not mix it into this commit.

After staging, check:

```bash
git add -A
git diff --cached --check
git diff --cached --stat
```

This is a large accumulated documentation and code change, so splitting it into a few topic-based commits would make review easier, although there is no storage reason to exclude the current untracked reports. I have not staged or committed anything.

Saved copy: [this answer](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-09-29-git-staging-audit.md).
