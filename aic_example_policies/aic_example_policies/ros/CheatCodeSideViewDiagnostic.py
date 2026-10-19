"""Privileged SFP side-view diagnostic; never a deployable perception policy.

Stock CheatCode transports to the requested port. At 80 mm plug-tip height
above it, pause and take two bounded lateral views with force aborts. The
controller then resumes the stock route. Geometry is used for this *camera
experiment* only, never as an input to scored visual identity inference.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
from rclpy.time import Time

from .CheatCode import CheatCode


class UnsafeViewMotion(RuntimeError):
    pass


class CheatCodeSideViewDiagnostic(CheatCode):
    STANDOFF_M = .080
    SIDE_M = .015
    FORCE_ABSOLUTE_STOP_N = 35.0
    FORCE_RISE_STOP_N = 5.0
    DT_S = .05

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._get_observation_for_view = None
        self._view_done = False
        self._view_running = False
        self._events = []
        self._baseline_force_n = None

    def _oracle_pairs(self):
        path=Path('/audit/oracle_contacts.jsonl')
        if not path.exists():
            raise UnsafeViewMotion('Oracle contact stream missing')
        with path.open('rb') as f:
            f.seek(0,2)
            size=f.tell();f.seek(max(0,size-131072))
            lines=f.read().splitlines()
        if not lines:raise UnsafeViewMotion('Oracle contact stream empty')
        latest=None
        for line in reversed(lines[-3:]):
            try:
                latest=json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if latest is None:raise UnsafeViewMotion('Oracle contact stream has no complete recent row')
        pairs=[];contact_force={}
        for contact in latest['contacts']:
            a=contact['collision1']['name'].split('::')[1]
            b=contact['collision2']['name'].split('::')[1]
            pair=tuple(sorted((a,b)))
            pairs.append(pair)
            for wrench in contact.get('wrenches',[]):
                force=wrench['body_1_wrench']['force']
                norm=math.sqrt(sum(float(v)**2 for v in force.values()))
                contact_force[pair]=max(contact_force.get(pair,0.),norm)
        return latest['sim_ns']/1e9,sorted(set(pairs)),contact_force

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        if 'nic' not in task.target_module_name.lower():
            raise ValueError('Side-view diagnostic is SFP-to-NIC only')
        self._get_observation_for_view = get_observation
        try:
            return super().insert_cable(task, get_observation, move_robot, send_feedback)
        except UnsafeViewMotion as ex:
            self._event('unsafe_abort', reason=str(ex))
            return False
        finally:
            Path('/audit/side_view_events.json').write_text(json.dumps({
                'schema':'aic_sfp_privileged_side_view_diagnostic/v1',
                'task_module':task.target_module_name,'port':task.port_name,
                'standoff_m':self.STANDOFF_M,'side_m':self.SIDE_M,
                'force_absolute_stop_n':self.FORCE_ABSOLUTE_STOP_N,
                'force_rise_stop_n':self.FORCE_RISE_STOP_N,'events':self._events,
                'privileged_motion_only':True,'deployable_identity_geometry':False,
            },indent=2)+'\n')

    def _event(self, stage, **extra):
        ns=self.time_now().nanoseconds
        obs=self._get_observation_for_view()
        force=obs.wrist_wrench.wrench.force
        force_n=float(np.linalg.norm([force.x,force.y,force.z]))
        oracle_time,pairs,contact_force=self._oracle_pairs()
        if ns/1e9-oracle_time>.25:
            raise UnsafeViewMotion('Oracle contact stream stale')
        tf=self._parent_node._tf_buffer.lookup_transform('base_link','gripper/tcp',Time())
        position=tf.transform.translation
        row=dict(stage=stage,sim_time_s=ns/1e9,force_n=force_n,
                 oracle_sim_time_s=oracle_time,contact_pairs=[list(x) for x in pairs],
                 contact_force_n={'/'.join(k):v for k,v in contact_force.items()},
                 tcp_base_m=[position.x,position.y,position.z],**extra)
        self._events.append(row)
        self.get_logger().info('SIDE_VIEW '+json.dumps(row))
        return force_n

    def _guard(self, stage, **extra):
        force=self._event(stage,**extra)
        if self._baseline_force_n is None:self._baseline_force_n=force
        if force>self.FORCE_ABSOLUTE_STOP_N or force-self._baseline_force_n>self.FORCE_RISE_STOP_N:
            raise UnsafeViewMotion(f'{stage}: force {force:.2f} N; baseline {self._baseline_force_n:.2f} N')
        event=self._events[-1];pairs=event['contact_pairs']
        for pair in pairs:
            if pair==['cable_0','enclosure']:continue
            if pair==['cable_0','task_board'] and event['contact_force_n'].get('cable_0/task_board',0.)<=1.0:
                continue
            raise UnsafeViewMotion(f'{stage}: unsafe oracle contact pair {pair}')

    def calc_gripper_pose(self, port_transform, slerp_fraction=1.0,
                          position_fraction=1.0, z_offset=.1,
                          reset_xy_integrator=False):
        pose=super().calc_gripper_pose(port_transform,slerp_fraction,
                                       position_fraction,z_offset,reset_xy_integrator)
        if (not self._view_done and not self._view_running and
                position_fraction>=.999 and slerp_fraction>=.999 and
                z_offset<=self.STANDOFF_M):
            self._pending_view_pose=copy.deepcopy(pose)
        return pose

    def _send_delta_pose_target(self,move_robot,target_pose):
        if (not self._view_done and not self._view_running and
                getattr(self,'_pending_view_pose',None) is not None):
            self._view_running=True
            self._view_done=True
            origin=copy.deepcopy(self._pending_view_pose)
            self._pending_view_pose=None
            self._view_sweep(move_robot,origin)
            self._view_running=False
        super()._send_delta_pose_target(move_robot,target_pose)

    def _view_sweep(self,move_robot,origin):
        self._guard('baseline_before_hold')
        for _ in range(20):
            super()._send_delta_pose_target(move_robot,origin)
            self.sleep_for(self.DT_S)
        self._guard('baseline_view')
        baseline_y=self._events[-1]['tcp_base_m'][1]
        previous=0.0
        for label,offset in (('positive',self.SIDE_M),('negative',-self.SIDE_M)):
            steps=max(1,int(np.ceil(abs(offset-previous)/.002)))
            for j in range(1,steps+1):
                fraction=j/steps
                pose=copy.deepcopy(origin)
                pose.position.y+=previous+(offset-previous)*fraction
                super()._send_delta_pose_target(move_robot,pose)
                self.sleep_for(self.DT_S)
                self._guard('moving_'+label,step=j,total=steps,offset_m=pose.position.y-origin.position.y)
            for _ in range(20):
                super()._send_delta_pose_target(move_robot,pose)
                self.sleep_for(self.DT_S)
            self._guard(label+'_view',offset_m=offset)
            actual=self._events[-1]['tcp_base_m'][1]-baseline_y
            if abs(actual-offset)>.005:
                raise UnsafeViewMotion(f'{label}: measured y shift {actual:.4f} m vs {offset:.4f} m target')
            previous=offset
