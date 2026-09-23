import importlib.util
import sys
from pathlib import Path

import torch


PATH = (Path(__file__).parents[1] / "aic_isaaclab/scripts/serl/rpdp_geometry.py")
spec = importlib.util.spec_from_file_location("rpdp_geometry", PATH)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


def assert_quat_equivalent(a, b, atol=1e-5):
    assert torch.allclose(a.abs(), b.abs(), atol=atol)


def test_pose_inverse_round_trip():
    p = torch.tensor([.2, -.3, .1])
    q = g.rotvec_to_quat(torch.tensor([.2, -.1, .3]))
    pi, qi = g.inverse(p, q)
    z, identity = g.compose(p, q, pi, qi)
    assert torch.allclose(z, torch.zeros(3), atol=1e-6)
    assert_quat_equivalent(identity, torch.tensor([1., 0, 0, 0]))


def test_rot6d_round_trip():
    q = g.rotvec_to_quat(torch.tensor([.4, -.2, .1]))
    recovered = g.rot6d_to_quat(g.quat_to_rot6d(q))
    assert_quat_equivalent(q, recovered)


def test_zero_connector_motion_produces_zero_tcp_chunk():
    adapter = g.ConnectorTCPAdapter(torch.tensor([-.0043, -.0174, .0568]),
                                    g.rotvec_to_quat(torch.tensor([0., 0., -.3])))
    p = torch.tensor([.01, -.02, .03])
    q = g.rotvec_to_quat(torch.tensor([.01, -.02, .03]))
    action = adapter.action_chunk(p, q, p, q)
    assert action.shape == (4, 6)
    assert torch.allclose(action, torch.zeros_like(action), atol=1e-6)


def test_adapter_chunk_composes_to_requested_tcp_endpoint():
    adapter = g.ConnectorTCPAdapter(torch.tensor([-.0043, -.0174, .0568]),
                                    g.rotvec_to_quat(torch.tensor([0., 0., -.3])))
    cp0 = torch.tensor([.001, -.002, -.01])
    cq0 = g.rotvec_to_quat(torch.tensor([.01, -.02, .03]))
    cp1 = torch.tensor([.002, -.001, -.007])
    cq1 = g.rotvec_to_quat(torch.tensor([.03, -.01, .04]))
    ep, eq = adapter.connector_to_tcp(cp0, cq0)
    for action in adapter.action_chunk(cp0, cq0, cp1, cq1):
        ep, eq = g.compose(ep, eq, action[:3], g.rotvec_to_quat(action[3:]))
    target_p, target_q = adapter.connector_to_tcp(cp1, cq1)
    assert torch.allclose(ep, target_p, atol=1e-6)
    assert_quat_equivalent(eq, target_q)


def test_adapter_broadcasts_fixed_transform_over_runtime_batch():
    adapter = g.ConnectorTCPAdapter(torch.tensor([-.0043, -.0174, .0568]),
                                    g.rotvec_to_quat(torch.tensor([.3, 0., 0.])))
    p0=torch.tensor([[.001,-.002,-.01]]);q0=g.rotvec_to_quat(torch.tensor([[.01,-.02,.03]]))
    p1=torch.tensor([[.002,-.001,-.007]]);q1=g.rotvec_to_quat(torch.tensor([[.03,-.01,.04]]))
    action=adapter.action_chunk(p0,q0,p1,q1)
    assert action.shape==(1,4,6)
    assert torch.isfinite(action).all()


def test_waypoint_chunk_reproduces_distinct_recorded_body_deltas():
    adapter=g.ConnectorTCPAdapter(torch.tensor([-.0043,-.0174,.0568]),
                                  g.rotvec_to_quat(torch.tensor([.3,0.,0.])))
    cp=torch.tensor([.001,-.002,-.01]);cq=g.rotvec_to_quat(torch.tensor([.01,-.02,.03]))
    tcp_p,tcp_q=adapter.connector_to_tcp(cp,cq)
    target=torch.tensor([[.0002,0.,0.,0.,0.,0.],
                         [0.,.0001,0.,0.,0.,.001],
                         [0.,0.,.0005,0.,0.,0.],
                         [-.0001,0.,0.,0.,-.001,0.]])
    waypoints=[]
    for delta in target:
        tcp_p,tcp_q=g.compose(tcp_p,tcp_q,delta[:3],g.rotvec_to_quat(delta[3:]))
        p,q=g.compose(tcp_p,tcp_q,adapter.tcp_to_connector_p,adapter.tcp_to_connector_q)
        waypoints.append(g.pose9(p,q))
    wp=torch.stack(waypoints);wp_p,wp_q=g.unpack_pose9(wp)
    recovered=adapter.waypoint_chunk(cp,cq,wp_p,wp_q)
    assert torch.allclose(recovered,target,atol=2e-6)
