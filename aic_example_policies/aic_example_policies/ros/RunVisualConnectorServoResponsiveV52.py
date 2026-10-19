"""Reduce empty waiting and codec latency without changing model predictions."""
from .RunVisualConnectorServoPrewarmedFreeV51 import RunVisualConnectorServoPrewarmedFreeV51


class RunVisualConnectorServoResponsiveV52(RunVisualConnectorServoPrewarmedFreeV51):
    parallel_fresh_JPEG = True

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        # Moving to a better view is conservative; accepting the target still
        # needs the unchanged three fresh native fits and downstream grasp lock.
        transport.initial_reject_limit = 1
        return transport

    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache.parallel = True

    def _initial_force_reference(self, get_observation, move_robot):
        original = list(self.local.cartesian_stiffness)
        self.local.cartesian_stiffness = [500.,500.,500.,50.,50.,50.]
        self._trace(event='initial_free_hold_stiffness_diagnostic',
                    stiffness_base=self.local.cartesian_stiffness,
                    force_samples_and_spread_gate_unchanged=True,
                    no_contact_retare=True,geometry_input=False)
        try:
            return super()._initial_force_reference(get_observation,move_robot)
        finally:
            self.local.cartesian_stiffness = original
