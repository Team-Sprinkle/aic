"""Oracle recorded targets through the actor's exact TCP-anchor chunk executor."""
import bisect
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .RunSharedPortTCP import RunSharedPortTCP
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RecordedTeacherChunkExecutorDiagnostic(RunSharedPortTCP):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.source = json.loads(Path(os.environ['AIC_TEACHER_REPLAY_SOURCE']).read_text())
        if not self.source['transport_only'] or self.source['final_scene']:
            raise ValueError('Diagnostic is restricted to nonfinal transport')
        self.target_frame = 'body'
        self.transport_terminal_reason = 'recorded_executor_diagnostic_complete'
        self.execute_steps = 6
        self.control_hz = 10.
        self.execute_complete_chunk = True
        self.source_times = [r['elapsed_s'] for r in self.source['commands']]
        self.replay_first_sim = None
        self.replay_elapsed = 0.

    def reset_for_task(self, task):
        super().reset_for_task(task)
        self.replay_first_sim = None
        self.replay_elapsed = 0.
        if task.target_module_name != self.source['task_module']:
            raise ValueError('Requested task differs from recorded diagnostic source')

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        tick = time.perf_counter()
        state = self.assembler.assemble_ros(observation)
        sim = self._control_sim_time(observation)
        if self.replay_first_sim is None:
            self.replay_first_sim = sim
        self.replay_elapsed = sim-self.replay_first_sim
        inverse = np.linalg.inv(matrix_from_xyz_xyzw(state[:7]))
        targets = []
        for j in range(12):
            wanted = self.replay_elapsed+j/self.control_hz
            idx = max(0,bisect.bisect_right(self.source_times,wanted)-1)
            absolute = matrix_from_xyz_xyzw(self.source['commands'][idx]['pose'])
            relative = inverse@absolute
            targets.append(np.r_[relative[:3,3],relative[:3,:2].T.reshape(-1)])
        self._trace(event='oracle_executor_proposal', sim_time=sim, elapsed_s=self.replay_elapsed,
                    action_horizon=12, execute_steps=6, source=str(os.environ['AIC_TEACHER_REPLAY_SOURCE']),
                    privileged_recorded_route=True, autonomous_actor=False)
        self._inference_latency_ms.append((time.perf_counter()-tick)*1000)
        return np.asarray(targets,np.float32),np.zeros(9,np.float32),state

    def _should_stop_transport(self, task_vector):
        return self.replay_elapsed >= self.source_times[-1]+8.
