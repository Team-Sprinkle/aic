# Phase 3 Reward Funnel Report - 2026-05-23

Implemented scripts:

- `aic_utils/aic_isaac/scripts/audit_phase_reward_funnel.py`
- `aic_utils/aic_isaac/scripts/privileged_insertion_servo_sweep.py`
- tests in `aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`

Existing reward implementation already supports the required funnel terms:

- `lateral_gate_width(s)` through scheduled lateral sigma.
- `orientation_gate_width(s)` through scheduled orientation tolerance.
- `axial_progress_gate` through axial progress scale and insert gate.
- `action_axis_gate`.
- `module_consistency_penalty` through semantic/trailing-body gate.
- `bypass_penalty`.
- contact/force penalty remains in SERL via force delta terms and in the new servo audit as a recovery classifier.
- smoothness/action penalties remain available in existing SERL losses/action clips.

## Audits

Hand-tuned baseline:

```text
outputs/agent_reward_funnel/reward_audits/20260523_phase3_hand_tuned
```

Auto from servo trajectories:

```text
outputs/agent_reward_funnel/reward_audits/20260523_phase3_auto_from_servo
```

Both runs write `config.json`, `summary.json`, `summary.md`, and reward surface PNGs.

## Parameters

| version | lateral gate m | orientation gate rad | module threshold | bypass penalty | bad-forward surfaces |
| --- | ---: | ---: | ---: | ---: | ---: |
| hand_tuned | 0.0006 | 0.030 | 0.80 | 8.0 | 0 |
| auto_from_servo | 0.0004 | 0.025 | 0.60 | 8.0 | 0 |

The auto tuner uses successful/near-success servo trajectories but is clamped so it cannot relax the final orientation gate above strict success. An earlier unconstrained auto pass loosened orientation too far and the audit caught false-positive forward reward; that was fixed by bounding the final insertion gate.

## False-Positive Check

The reward audit sweeps:

- axial depth `s`,
- lateral error `r`,
- orientation error `theta`,
- action lateral error.

It flags positive axial reward in the near/inside region when `r` or `theta` are outside the tight gate. Both retained funnels have zero flagged surfaces. This means the formula rejects the known failure where forward insertion is rewarded while the tip is laterally or orientationally bad.

This does not prove Isaac insertion. It only proves the reward surface is not obviously paying for the known bad axial moves.
