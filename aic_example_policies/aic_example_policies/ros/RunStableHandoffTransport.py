"""Matched learned transport: neural proposal, fixed hold, strict visual approval.

Vision can approve/reject this comparison's stop, but cannot correct its route.
No scene/target geometry is available to the deployable controller.
"""
import time

import numpy as np

from .RunTemporalTransport import RunTemporalTransport
from aic_model.policy import build_pose_from_vectors
from aic_control_interfaces.msg import TargetMode


class RunStableHandoffTransport(RunTemporalTransport):
    def _inspect_transport_handoff(self, observed, state, task_vector,
                                   get_observation, move_robot, decision_sim):
        if task_vector[0] == 1 or self.sc_shadow_verifier is None:
            raise ValueError('This initial matched comparison requires the SC visual verifier')
        held_pose=build_pose_from_vectors(state[:3],state[3:7])
        if hasattr(self._parent_node,'_target_mode'):
            self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
        self.set_pose_target(move_robot=move_robot,pose=held_pose,frame_id='base_link',
                             stiffness=[240,240,240,50,50,50],damping=[50,50,50,20,20,20])
        requested=int(np.argmax(task_vector[2:4]))
        self._trace(event='matched_handoff_hold',sim_time=decision_sim,state=state.tolist(),
                    stiffness=[240,240,240,50,50,50],route_correction_enabled=False,
                    geometry_input=False)
        deadline=time.monotonic()+20.
        last_sim=decision_sim
        valid_start=None
        valid_count=0
        while time.monotonic()<deadline:
            observation=get_observation()
            if observation is None:
                time.sleep(.005)
                continue
            sim=self._control_sim_time(observation)
            if sim-last_sim<.15:
                time.sleep(.005)
                continue
            last_sim=sim
            fresh=self.assembler.assemble_ros(observation)
            images={c:self._shadow_raw_rgb(getattr(observation,c+'_image'))
                    for c in ('center','left','right')}
            started=time.perf_counter()
            report=self.sc_shadow_verifier.inspect(images,fresh,requested)
            elapsed=(time.perf_counter()-started)*1000
            speed=float(np.linalg.norm(fresh[7:10])*1000)
            angular=float(np.linalg.norm(fresh[10:13])*180/np.pi)
            self._trace(event='matched_handoff_inspection',sim_time=sim,state=fresh.tolist(),
                        requested_port=requested,report=report,verification_ms=elapsed,
                        tcp_speed_mm_s=speed,angular_speed_deg_s=angular,
                        geometry_input=False,route_correction_enabled=False)
            if not report['valid']:
                self._trace(event='matched_handoff_rejected',sim_time=sim,
                            reason=report['reason'],geometry_input=False)
                return False
            if valid_start is None:
                valid_start=sim
            valid_count+=1
            if sim-valid_start>=2. and valid_count>=3 and speed<=5. and angular<=3.:
                self._handoff_terminal_snapshot=(fresh.copy(),sim)
                self._trace(event='matched_handoff_accepted',sim_time=sim,state=fresh.tolist(),
                            requested_port=requested,report=report,
                            held_valid_s=sim-valid_start,tcp_speed_mm_s=speed,
                            angular_speed_deg_s=angular,geometry_input=False,
                            insertion_enabled=False)
                return True
            if sim-decision_sim>=8.:
                break
        self._trace(event='matched_handoff_rejected',sim_time=last_sim,
                    reason='stationary_hold_timeout',geometry_input=False)
        return False
