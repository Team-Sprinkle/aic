# Isaac Online SERL/SAC Design

The primary long-term hybrid path should use the same actor family across
offline and online stages:

```text
Gazebo expert dataset
  -> ACT BC checkpoint
  -> offline vision SERL with frozen ACT + trainable adapter
  -> Isaac online SERL/SAC with the same ACT-adapter actor
  -> Gazebo transfer validation and recovery data
```

The existing PPO/RSL-RL code remains useful as a smoke test, baseline, and
backup trainer, but it is not the primary compatible transfer path because its
actor architecture differs from the ACT-backed SERL actor.

## Actor

Online Isaac SERL should load the offline vision SERL checkpoint:

```text
obs -> ACT -> base action chunk a_ACT
state + a_ACT -> adapter MLP -> delta action
final action = a_ACT + adapter_scale * delta action
```

ACT is frozen by default. Optional partial finetuning can use a low `act_lr`,
while the adapter and critics use higher learning rates.

## Critics

Twin critics are trained from scratch or resumed from the offline vision SERL
checkpoint. They may use the lightweight image/state encoder already used by
offline vision SERL initially. A later Isaac-specific encoder can replace it if
the checkpoint format preserves the actor contract.

## Replay

The online path should be off-policy:

- collect Isaac transitions with dense Isaac reward;
- store observations, action chunks or first executed action, rewards, dones,
  and next observations;
- update critics with SAC/SERL TD targets;
- update the adapter actor with RL loss, BC or ACT-preservation regularization,
  adapter magnitude penalty, and optional entropy tuning.

## Current Code State

`aic_utils/aic_isaac/scripts/train_isaac_online_serl.py` launches the Isaac Lab
online SERL trainer:

- loads the offline ACT-adapter SERL checkpoint;
- loads an ACT TorchScript export so Isaac does not need to import LeRobot;
- keeps Isaac camera sensors enabled;
- disables PPO-specific ResNet observation terms for this trainer only;
- reads raw Isaac camera RGB tensors from the camera sensors and resizes them to
  the LeRobot ACT image shape;
- collects replay transitions from `AIC-Task-v0`;
- runs critic and adapter actor updates with ACT frozen;
- writes `checkpoint_latest.pt`, `metrics.jsonl`, and `train_config.json`.

The first artifact-producing run is intentionally tiny: 3 Isaac steps and 2
online updates. A second sanity run requested 300 steps and 100 updates with a
30-minute guard; it stopped after 107 Isaac steps and 100 updates in about 4.09
minutes. A 1k-step guarded run completed 1000 Isaac steps and 993 updates in
about 5.90 minutes. These runs prove the loop creates real online RL
checkpoints, not that the policy is useful yet. The 1k run showed substantial
adapter growth, so the next training iteration should use stronger delta
control, such as lower adapter LR, action/delta clipping, or higher
ACT-preservation regularization.
