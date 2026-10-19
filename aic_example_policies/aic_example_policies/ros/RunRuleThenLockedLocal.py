"""SC rule transport -> strict RGB handoff -> predicted-port insertion BC."""
import os
import numpy as np

from .RunRuleTransport import RunRuleTransport
from .RunLockedPortLocal import RunLockedPortLocal


class RunRuleThenLockedLocal(RunRuleTransport):
    local_translation_stiffness=240.

    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        super().insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        if not self.rule_final_valid or self.rule_failure:
            return False
        observation=get_observation()
        if observation is None:return False
        state=self.assembler.assemble_ros(observation)
        sim=self._control_sim_time(observation)
        vector=self.task_vector(task)
        requested=int(np.argmax(vector[2:4]))
        images={c:self.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
        report=self.verifier.inspect(images,state,requested)
        if not report['valid']:
            self._trace(event='locked_local_switch_rejected',sim_time=sim,report=report)
            return False
        checkpoint=os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT']
        local=RunLockedPortLocal(self._parent_node,checkpoint)
        local.bind_visual_handoff(report,sim)
        local.control_hz=20.
        local.control_clock='simulation'
        local.execute_steps=4
        local.execute_complete_chunk=True
        local.max_simulation_sec=70.
        local.cartesian_stiffness=[self.local_translation_stiffness]*3+[50,50,50]
        local.cartesian_damping=[50,50,50,20,20,20]
        local.max_translation=.015
        local.max_rotation=.1
        self._trace(event='locked_local_switch',sim_time=sim,state=state.tolist(),
                    checkpoint=checkpoint,report=report,geometry_input=False,
                    local_stiffness=local.cartesian_stiffness,
                    privileged_actor_guidance=False,insertion_enabled=True)
        return local.insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
