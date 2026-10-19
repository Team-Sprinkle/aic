#!/usr/bin/env python3
"""CPU checks for the final runtime's frame and observation-consensus contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'aic_example_policies'))
import numpy as np
from scipy.spatial.transform import Rotation
from aic_example_policies.final.runtime.shared_port_tcp_geometry import body_delta_from_target, inferred_base_port
from aic_example_policies.final.runtime.stable_pose_window import StablePoseWindow


def pose9(matrix):
    return np.r_[matrix[:3,3], matrix[:3,:2].T.reshape(-1)]


def transform(xyz, degrees):
    m=np.eye(4);m[:3,3]=xyz;m[:3,:3]=Rotation.from_euler('xyz',degrees,degrees=True).as_matrix()
    return m


class RuntimeContracts(unittest.TestCase):
    def test_rotated_tcp_and_port_frames(self):
        tcp=transform([-.4,.2,.3],[165,-20,90])
        port=transform([-.3,.1,.1],[90,35,-60])
        desired=transform([.001,-.002,-.025],[1,-.5,0])
        state=np.r_[tcp[:3,3],Rotation.from_matrix(tcp[:3,:3]).as_quat()]
        inferred=inferred_base_port(state,pose9(np.linalg.inv(port)@tcp))
        np.testing.assert_allclose(inferred,port,atol=1e-12)
        delta=body_delta_from_target(state,port,pose9(desired))
        body=np.eye(4);body[:3,3]=delta[:3];body[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()
        np.testing.assert_allclose(tcp@body,port@desired,atol=1e-12)

    def test_pose_window_does_not_accept_a_single_outlier(self):
        w=StablePoseWindow();m=np.eye(4)
        self.assertIsNone(w.update(m,0.))
        self.assertIsNone(w.update(m,.1))
        self.assertIsNotNone(w.update(m,.2))
        outlier=m.copy();outlier[0,3]=.005
        self.assertIsNone(w.update(outlier,.3))
        self.assertIsNone(w.update(m,.4))
        self.assertIsNone(w.update(m,.5))
        accepted=w.update(m,.6)
        self.assertIsNotNone(accepted)
        np.testing.assert_allclose(accepted['matrix'],m)

    def test_duplicate_observation_time_is_rejected(self):
        w=StablePoseWindow();w.update(np.eye(4),1.)
        with self.assertRaises(ValueError):w.update(np.eye(4),1.)


if __name__=='__main__':
    unittest.main()
