import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from prepare_verified_insertion_command_rows import base_command, measured_state, episode_identity


def pose(xyz, quaternion=(0,0,0,1)):
    return dict(position=dict(zip('xyz',xyz)),orientation=dict(zip('xyzw',quaternion)))


def test_body_delta_uses_past_measured_tcp_rotation():
    q=(0,0,np.sqrt(.5),np.sqrt(.5))
    controllers=[[90,dict(tcp_pose=pose([1,2,3],q))],
                 [110,dict(tcp_pose=pose([9,9,9]))]]
    command=dict(sim_ns=100,message=dict(header=dict(frame_id='gripper/tcp'),pose=pose([.01,0,0])))
    result=base_command(command,controllers,[90,110])
    assert np.allclose(result[:3,3],[1,2.01,3])
    # Label is the commanded offset, never future measured pose [9,9,9].
    assert not np.allclose(result[:3,3],[9,9,9])


def test_body_command_rejects_future_only_and_stale_tcp():
    command=dict(sim_ns=100_000_000,message=dict(header=dict(frame_id='gripper/tcp'),pose=pose([.01,0,0])))
    for times in ([110_000_000],[1]):
        with pytest.raises(ValueError):
            base_command(command,[[times[0],dict(tcp_pose=pose([1,2,3]))]],times)


def test_base_target_does_not_apply_current_tcp_twice():
    command=dict(sim_ns=100,message=dict(header=dict(frame_id='base_link'),pose=pose([.01,.02,.03])))
    assert np.allclose(base_command(command,[],[])[:3,3],[.01,.02,.03])


def test_state_rejects_future_only_observations():
    stream=dict(streams={'/aic_controller/controller_state':[[110,{}]]})
    with pytest.raises(ValueError):
        measured_state(stream,100)


def test_same_name_in_different_collection_banks_does_not_merge_history():
    a=episode_identity('/bank_a/episode_00/scored_stream.json','episode_00')
    b=episode_identity('/bank_b/episode_00/scored_stream.json','episode_00')
    assert a!=b
    assert a==episode_identity('/bank_a/episode_00/scored_stream.json','episode_00')
