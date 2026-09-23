# Production CheatCode audit artifacts

Compact, reviewable evidence is kept here. Bulk MCAP bags and one-Hz raw camera
frames are stored at:

```text
/var/tmp/chmin_aic_prod_cheatcode_audit_20260923/
```

The bulk directory contains:

- `official_qualification/run_*/`: raw frames and MCAP files from the five
  unchanged three-trial replays;
- `production_family_invalid_diskfull/`: a preserved invalid first stress run;
  scoring became unusable when `/data1` filled during trial 5;
- `production_family_valid2/`: clean production-family rerun on the larger root
  filesystem.

Do not count the `invalid_diskfull` results. The exact replay summaries and the
selected run-3 SC failure analysis remain under `official_qualification/` in
this directory. The stress manifest labels every deliberate scene variation.

All simulator runs used rootless Docker and GPU 1. Existing ACT and Isaac
containers were left running and were not modified.

## Ordinary development and VLM route follow-up

`ordinary_broad_followup/` holds the 19-scene broad development manifest,
selected score summary (including rejected invalid attempts), historical
archive review, measured plug/port analyses, and compact one-second videos.
The clean scores were 13 full, five partial, and one no insertion. The first
long batch developed Gazebo physics-entity errors; isolated replays were used
for its affected scenes.

`ordinary_broad_followup/vlm_review/` holds a scan of 190 retained SC
agent/VLM replay attempts, three selected low-score force-review sheets, and
an exact scene/trajectory replay of one saved three-card score-1 attempt. It
also holds a regenerated five-card seed-51500 stock-CheatCode control. The
three-card route failed again at port handoff. The seed control ended in a
partial insertion. Neither gave a causal cable-snag label. A historical note
records an earlier five-card VLM route catching the cable on cards, but its
failed raw video/trajectory was not retained in the checked archives.

Scripts under this directory regenerate manifests, run rootless Gazebo,
extract bag geometry, render synchronized reviews, and select only valid
scores. Exact commands and interpretation are in the
[ordinary development cable audit](../../docs/experiments/2026-09-23-ordinary-development-cable-audit.md).
Bulk bags, engine logs, and captured frames remain under the `/var/tmp` roots
listed there. `artifact_map.json` and `sha256.txt` identify the compact files
that were preserved here.

## Fixed five-card route probe

`ordinary_broad_followup/cable_route_probe/` contains a second, deliberately
longer-route comparison on one fixed SC five-card Gazebo scene. Three
across-card runs yielded one full, one partial, and one no insertion; three
outside-left runs yielded one full and two partial insertions. The failed
across-card run brought a cable link close to the main PCB and then nearly
stopped that link during its failed approach. A lower route is retained as a
robot-clearance confound. The full [experiment record](../../docs/experiments/2026-09-23-fixed-five-card-route-probe.md)
explains the missing cable/card contact identity, S3 audit, exact commands,
videos, and overhead/side cable views. Bulk MCAPs, camera frames, and logs
remain at `/var/tmp/chmin_aic_cable_route_probe_20260923/`. No model was
trained.
