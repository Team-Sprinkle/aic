"""V31 control with family-specific lateral stiffness; axial compliance unchanged."""
from .RunVisualConnectorServoBiasSeedV31 import RunVisualConnectorServoBiasSeedV31


class RunVisualConnectorServoStiffV32(RunVisualConnectorServoBiasSeedV31):
    lateral_stiffness=240.

    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        sc=self.local.task_vector(task)[1]==1
        stiffness=self.lateral_stiffness if sc else 90.
        self.local.cartesian_stiffness=[stiffness,stiffness,90.,50.,50.,50.]
        self._trace(event='SC_lateral_stiffness_diagnostic',family='sc' if sc else 'sfp',
            stiffness_base=self.local.cartesian_stiffness,damping=self.local.cartesian_damping,
            axial_stiffness_unchanged=True,force_guards_unchanged=True,geometry_input=False)
        return super()._execute(task,get_observation,move_robot,send_feedback,**kwargs)
