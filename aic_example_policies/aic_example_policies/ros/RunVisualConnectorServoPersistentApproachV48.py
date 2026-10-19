"""Initialize the persistent approach from its actual preceding public hold."""
from .RunVisualConnectorServoPersistentApproachV47 import RunVisualConnectorServoPersistentApproachV47


class RunVisualConnectorServoPersistentApproachV48(RunVisualConnectorServoPersistentApproachV47):
    def _approach_tcp_rotation(self, port):
        reference = self.fresh_public_reference
        if reference is None:
            raise ValueError('Fresh public API hold must precede persistent approach')
        # The insertion governor seed is normally written by the first approach
        # command. The angular planner needs that reference before its first tick.
        self.local.governor_seed = (reference.copy(),self.local.last_observation_stamp)
        self._trace(event='persistent_approach_seed_from_public_hold',
                    source='public_controller_reference_tcp_pose',
                    previous_episode_seed_reused=False,geometry_input=False)
        return super()._approach_tcp_rotation(port)
