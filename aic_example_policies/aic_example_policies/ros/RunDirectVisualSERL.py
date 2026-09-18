"""Official-runtime entry point for a direct visual policy, with no ACT dependency.

Set AIC_SERL_CHECKPOINT and AIC_SERL_N_ACTION_STEPS=1 for a one-step policy.
Command frame, control frequency, and runtime limits use the AIC_SERL_* options.
"""

from .RunACTAdapterSERL import RunACTAdapterSERL


class RunDirectVisualSERL(RunACTAdapterSERL):
    ALLOW_ACT_EXPORT = False
    DEFAULT_ACTION_STEPS = 1
    DEFAULT_TRANSLATION_DEADBAND = 0.0
    DEFAULT_ROTATION_DEADBAND = 0.0

    def __init__(self, parent_node):
        super().__init__(parent_node)
        if self.policy.actor.actor_mode != "direct_visual":
            raise ValueError("RunDirectVisualSERL requires a direct_visual checkpoint")
