"""Full connector/port SE(3), using explicitly supplied estimated grasp poses.

No scene lookup or privileged geometry. These helpers are a separate candidate
interface; the frozen hybrid tip-position/TCP-orientation actor is unchanged.
"""
import numpy as np


def rigid(value):
    value=np.asarray(value,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all() or
        not np.allclose(value[3],[0,0,0,1],atol=1e-7) or
        not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) or
        not np.isclose(np.linalg.det(value[:3,:3]),1,atol=1e-5)):
        raise ValueError('Full estimated SE(3) required')
    return value


def connector_in_port(base_port,base_tcp,tcp_connector):
    return np.linalg.inv(rigid(base_port))@rigid(base_tcp)@rigid(tcp_connector)


def recorded_command_to_connector_target(port_tcp_command,tcp_connector):
    """Encode recorded TCP commands, not future measured motion."""
    return rigid(port_tcp_command)@rigid(tcp_connector)


def connector_target_to_tcp(port_connector_target,tcp_connector):
    """Decode a full desired connector pose through the SAME estimated grasp."""
    return rigid(port_connector_target)@np.linalg.inv(rigid(tcp_connector))


def connector_target_to_body_command(base_tcp,base_port,port_connector_target,tcp_connector):
    return np.linalg.inv(rigid(base_tcp))@rigid(base_port)@connector_target_to_tcp(port_connector_target,tcp_connector)
