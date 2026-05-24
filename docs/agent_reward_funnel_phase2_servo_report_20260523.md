# Phase 2 Guarded Privileged Servo Report - 2026-05-23

Run:

```text
outputs/agent_reward_funnel/servo_sweeps/20260523_phase2_privileged_servo_sweep
```

Artifacts:

- `config.json`
- `git_status.txt`
- `git_diff.patch`
- `metrics.json`
- `metrics.csv`
- `case_summary.csv`
- `trajectory_snapshot.png`
- `failure_modes.png`
- `summary.md`

This is a geometry/controller smoke test, not an Isaac/Gazebo success claim. It uses privileged `s/r/theta/module_consistency/force` state to test the guarded servo logic before launching GPU simulation.

## Sweep

- lateral starts: `1, 2, 4, 6, 10 mm`
- axial starts: `3, 6, 10, 20 mm`
- orientation starts: `small=0.035 rad`, `medium=0.070 rad`, `hard=0.120 rad`
- total cases: 60

## Result

| outcome | count |
| --- | ---: |
| strict_success | 2 |
| near_success_not_strict | 2 |
| module_consistency_failure | 20 |
| contact_spike | 36 |

Strict synthetic successes:

| case | final s mm | final r mm | theta rad | module | max force N |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lat1_ax3_small` | 9.04 | 0.00 | 0.000 | 0.927 | 1.12 |
| `lat2_ax3_small` | 9.05 | 0.00 | 0.000 | 0.929 | 3.36 |

Near misses:

| case | mode | final s mm | final r mm | theta rad | module | max force N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lat1_ax3_medium` | near_success_not_strict | 8.81 | 0.02 | 0.000 | 0.891 | 2.35 |
| `lat2_ax3_medium` | near_success_not_strict | 8.53 | 0.00 | 0.000 | 0.837 | 4.59 |

## Interpretation

The guarded servo can solve only the easiest synthetic starts: small lateral error, close axial start, and small orientation error. The dominant failure for wider starts is contact spike, followed by module consistency failure. This matches the prior Isaac diagnosis: maintaining lateral correction and module/body consistency during final insertion is more important than increasing axial reward.

The positive result is limited but useful: the strict success predicate is hard enough to reject tip-depth-only progress. Several cases reached positive `s` but failed module consistency, and they were correctly classified as failures.

## Next Isaac Run

Use the audit smoke command in `docs/agent_reward_funnel_audit_20260523.md`. The key Isaac check is whether the same controller-aware guard preserves realized semantic tip motion and visible module insertion under real contact. If no strict success occurs, classify from `audit_log.jsonl` and videos as lateral bypass, orientation residual, contact spike, controller realization mismatch, module consistency failure, or no axial progress.
