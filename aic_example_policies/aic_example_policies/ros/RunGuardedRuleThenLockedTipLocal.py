"""SC acquisition and one full-target BC actor with a camera-derived tip input."""
import os,time
import numpy as np
from .RunYawMemoryTrackedRuleTransport import RunYawMemoryTrackedRuleTransport
from .RunLockedPortLocal import RunLockedPortLocal
from sc_native_grasp_observation import SCNativeGraspObservation


class RunGuardedRuleThenLockedTipLocal(RunYawMemoryTrackedRuleTransport):
    def prepare_local_start(self,task,get_observation,move_robot,send_feedback):
        return True

    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.local=RunLockedPortLocal(parent_node,os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'])
        self.grasp=SCNativeGraspObservation(os.environ['AIC_LOCKED_GRASP_CHECKPOINT'],str(self.device))
        self.local.control_hz=20.;self.local.control_clock='simulation'
        self.local.execute_steps=4;self.local.execute_complete_chunk=True
        self.local.max_simulation_sec=70.;self.local.cartesian_stiffness=[90,90,90,50,50,50]
        self.local.cartesian_damping=[50,50,50,20,20,20];self.local.max_translation=.015;self.local.max_rotation=.1

    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        super().insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        if not self.rule_final_valid or self.rule_failure:return False
        requested=int(np.argmax(self.task_vector(task)[2:4]));predictions=[];latest=None;last_sim=-1.;start=time.monotonic()
        while len(predictions)<3 and time.monotonic()-start<15:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            state=self.assembler.assemble_ros(observation);sim=self._control_sim_time(observation)
            if sim-last_sim<.15:time.sleep(.005);continue
            images={c:self.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
            self._guidance_sim=sim
            # Preserve current identity and original-image refinement. The
            # remembered board pose is observational and reset per action.
            report=self.inspect_with_guidance(images,state,requested)
            if not report['valid']:
                self._trace(event='tip_local_switch_rejected',sim_time=sim,report=report,geometry_input=False)
                return False
            prediction=self.grasp.predict(images);predictions.append(prediction);latest=(report,sim,state);last_sim=sim
        if len(predictions)!=3:return False
        positions=np.asarray([p['predicted_tip_position_tcp_m'] for p in predictions]);body=np.median(positions,axis=0);spread=float(np.max(np.linalg.norm(positions-body,axis=1)))
        if spread>.0005:
            self._trace(event='tip_local_switch_rejected',reason='inconsistent_visual_grasp_position',spread_mm=spread*1000,geometry_input=False)
            return False
        report,sim,state=latest;self.local.bind_visual_handoff(report,sim);self.local.bind_predicted_body_tip(body)
        self._trace(event='locked_local_switch',sim_time=sim,state=state.tolist(),report=report,
                    tip_input_mode=self.local.tip_input_mode,predicted_body_tip_position_m=body.tolist(),
                    tip_orientation_input=False,grasp_predictions=predictions,grasp_spread_mm=spread*1000,
                    local_stiffness=self.local.cartesian_stiffness,geometry_input=False,
                    privileged_actor_guidance=False,insertion_enabled=True,backoff_enabled=False)
        if not self.prepare_local_start(task,get_observation,move_robot,send_feedback):
            return False
        return self.local.insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
