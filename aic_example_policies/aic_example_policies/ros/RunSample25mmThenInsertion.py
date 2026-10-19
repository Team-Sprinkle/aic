"""Bounded diagnostic: extend visual transport to25mm, then frozen BC.

The visual approach owns the150->25mm segment. This is not a learned actor
improvement. Current diagnostic excludes recovery to isolate initial control;
hierarchical acquisition checkpoints need separate integration at this range.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .RunSampleThenSharedInsertion import RunSampleThenSharedInsertion
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunSample25mmThenInsertion(RunSampleThenSharedInsertion):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        if self.mode != 'none':
            raise ValueError('The25mm handoff diagnostic requires recovery=none; preserve old hierarchical controls')

    def _approach_tcp_rotation(self, port):
        return port[:3, :3] @ Rotation.from_quat(self.rule.params['nominal_tcp_in_port_xyzw']).as_matrix()

    def _approach_orientation_tolerance(self):
        return np.deg2rad(2.)

    def _approach_speed(self, axial):
        return .012 if axial < -.050 else .004

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        if not super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context):
            return False
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        body = self.local.visual_body_tip.copy()
        rotation = self._approach_tcp_rotation(port)
        integral = np.zeros(3)
        tracker = None
        orientation_tracker = None
        reference = None
        last = None
        first = None
        best = float('inf')
        last_progress = None
        stable_since = None
        force_streak = 0
        started = time.monotonic()
        self.pipeline_phase = 'visual_25mm_approach'
        while time.monotonic() - started < 100:
            observation = get_observation()
            if observation is None:
                time.sleep(.002)
                continue
            sim = self.local._control_sim_time(observation)
            if last is not None and sim - last < .049:
                time.sleep(.002)
                continue
            dt = .05 if last is None else min(.1, sim - last)
            last = sim
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = (inverse @ np.r_[measured[:3, 3] + measured[:3, :3] @ body, 1.])[:3]
            angle = (Rotation.from_matrix(measured[:3, :3]).inv() * Rotation.from_matrix(rotation)).magnitude()
            if first is None:
                first = sim
                reference = np.array([0., 0., point[2]])
                if getattr(self,'approach_antiwindup',False):
                    from approach_reference import ApproachReference
                    bias_base=state[13:16]
                    if getattr(self,'approach_bias_from_issued_reference',False):
                        bias_base=self.local.governor_seed[0][:3,3]-measured[:3,3]
                    tracker=ApproachReference(point,inverse[:3,:3]@bias_base)
                    reference=tracker.reference.copy();integral=tracker.integral.copy()
                    self._trace(event='approach_public_bias_seeded',sim_time=sim,tracking_bias_port_m=integral.tolist(),geometry_input=False)
                if getattr(self,'persistent_approach_orientation',False):
                    from approach_orientation_reference import ApproachOrientationReference
                    seed=self.local.governor_seed[0]
                    orientation_tracker=ApproachOrientationReference(seed[:3,:3],rotation)
                last_progress = sim
            distance = np.linalg.norm(point - [0., 0., -.025])
            if distance < best - .0005:
                best = distance
                last_progress = sim
            force = self._force(state)
            force_streak = force_streak + 1 if force >= 8 else 0
            failure = ('raw_force_guard' if np.linalg.norm(state[26:29]) > 40 else
                       'load_change_guard' if force_streak >= 2 else
                       'approach_simulation_budget' if sim - first > 55 else
                       'approach_no_progress' if sim - last_progress > 12 else None)
            if failure:
                self._trace(event='near_handoff_failed', sim_time=sim, reason=failure,
                    state=state.tolist(), estimated_tip_port_m=point.tolist(), load_change_n=force,
                    geometry_input=False, actor_owned=False)
                return False
            # Align at safe stand-off before moving axially. Slow down for
            # the last50mm; never continue a blind axial push when off-axis.
            alignment_limit = .003 if point[2] < -.050 else .0015
            aligned=np.linalg.norm(point[:2]) <= alignment_limit and angle <= self._approach_orientation_tolerance()
            if tracker is not None:
                if point[2]>-.020:
                    self._trace(event='near_handoff_failed',sim_time=sim,reason='approach_crossed20mm_buffer',geometry_input=False)
                    return False
                reference,integral=tracker.step(point,inverse[:3,:3]@state[7:10],dt,self._approach_speed(point[2]) if aligned else 0.)
            else:
                if aligned:
                    speed = self._approach_speed(point[2])
                    reference[2] = min(-.025, reference[2] + speed * dt)
                integral = np.clip(integral + .6 * dt * (reference - point), -.015, .015)
            commanded_point = reference + integral
            goal = measured.copy()
            relative_rotation = Rotation.from_matrix(measured[:3, :3]).inv() * Rotation.from_matrix(rotation)
            rotvec = relative_rotation.as_rotvec()
            rotvec *= min(1., np.deg2rad(1.) / max(np.linalg.norm(rotvec), 1e-12))
            angular_diagnostic = None
            if orientation_tracker is None:
                goal[:3, :3] = measured[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
            else:
                goal[:3,:3],angular_diagnostic=orientation_tracker.step(measured[:3,:3],dt)
            destination = (port @ np.r_[commanded_point, 1.])[:3] - goal[:3, :3] @ body
            step = destination - measured[:3, 3]
            goal[:3, 3] += step * min(1., .020 / max(np.linalg.norm(step), 1e-12))
            self._command(move_robot, goal, owner='visual_transport_150_to25mm')
            self._trace(event='near_visual_approach', sim_time=sim, state=state.tolist(),
                estimated_tip_port_m=point.tolist(), reference_tip_port_m=reference.tolist(),
                commanded_tip_port_m=commanded_point.tolist(), integral_port_m=integral.tolist(),antiwindup=tracker is not None, load_change_n=force,
                desired_rotation_error_deg=float(np.rad2deg(angle)),
                persistent_orientation_reference=angular_diagnostic,
                port_source='fresh_RGB_lock_at150mm_rigid_fixture', geometry_input=False, actor_owned=False)
            ready = (abs(point[2] + .025) <= .002 and np.linalg.norm(point[:2]) <= .001
                     and angle <= self._approach_orientation_tolerance() and np.linalg.norm(state[7:10]) <= .005)
            stable_since = (sim if stable_since is None else stable_since) if ready else None
            if stable_since is not None and sim - stable_since >= .5:
                self.acquisition_state = state.copy()
                self.local.robot_history = []
                self._trace(event='near_handoff_ready', sim_time=sim, state=state.tolist(),
                    estimated_tip_port_m=point.tolist(), desired_gap_mm=25.,
                    estimated_orientation_error_deg=float(np.rad2deg(angle)),
                    orientation_tolerance_deg=float(np.rad2deg(self._approach_orientation_tolerance())),
                    geometry_input=False, learned_actor_not_yet_started=True,
                    observation_source='retained_public_observation_before_return',
                    observation_wall_time_ns=time.time_ns(), post_reset_geometry_used=False)
                return True
        self._trace(event='near_handoff_failed', reason='approach_wall_budget', geometry_input=False)
        return False
