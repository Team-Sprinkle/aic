"""Observation-only transport contact gate and short sideways retry geometry.

The gate uses load change from a frozen free-hold reference and measured
progress toward the current route waypoint. It does not read scene geometry.
"""
from collections import deque

import numpy as np


class TransportLoadStallMonitor:
    def __init__(self, *, force_n=8., window_s=.45, maximum_progress_m=.002,
                 minimum_goal_error_m=.015):
        self.force_n=float(force_n)
        self.window_s=float(window_s)
        self.maximum_progress_m=float(maximum_progress_m)
        self.minimum_goal_error_m=float(minimum_goal_error_m)
        self.reset()

    def reset(self):
        self.stage=None
        self.samples=deque()

    def update(self, *, sim_time, stage, goal_error_m, load_change_n):
        if stage!=self.stage:
            self.reset()
            self.stage=stage
        sample=(float(sim_time),float(goal_error_m),float(load_change_n))
        if self.samples and sample[0]<=self.samples[-1][0]:
            return False
        self.samples.append(sample)
        while len(self.samples)>1 and sample[0]-self.samples[0][0]>self.window_s:
            self.samples.popleft()
        if len(self.samples)<3 or sample[0]-self.samples[0][0]<self.window_s*.7:
            return False
        if sample[1]<self.minimum_goal_error_m:
            return False
        if sum(item[2]>=self.force_n for item in self.samples)<3 or sample[2]<self.force_n:
            return False
        progress=self.samples[0][1]-sample[1]
        return progress<=self.maximum_progress_m


def measured_approach_axis(positions, *, lookback_m=.012):
    """Recent measured travel direction, independent of route/port geometry."""
    points=np.asarray(positions,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('At least two finite measured TCP positions are required')
    current=points[-1]
    for previous in points[-2::-1]:
        delta=current-previous
        distance=float(np.linalg.norm(delta))
        if distance>=lookback_m:
            return delta/distance
    delta=current-points[0]
    distance=float(np.linalg.norm(delta))
    if distance<.003:
        raise ValueError('No reliable recent measured travel direction')
    return delta/distance


def horizontal_lateral_axis(approach_axis, sign=1):
    """Sideways in the base horizontal plane and orthogonal to recent motion."""
    axis=np.asarray(approach_axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Invalid approach axis')
    axis=axis/np.linalg.norm(axis)
    horizontal=np.array([-axis[1],axis[0],0.])
    norm=float(np.linalg.norm(horizontal))
    if norm<.1:
        # Any horizontal direction is orthogonal to nearly vertical travel.
        horizontal=np.array([1.,0.,0.]);norm=1.
    return (1 if sign>=0 else -1)*horizontal/norm
