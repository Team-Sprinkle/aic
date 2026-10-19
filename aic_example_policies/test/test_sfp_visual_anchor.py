"""Partial views cannot invent or replace a precise requested-port identity."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sfp_anchored_port_observation import AnchoredSfpPort


class Observer:
    optics={}
    def predict_batch(self,items):return [copy.deepcopy(self.result) for _ in items]


def prediction(views,x=0.):
    return dict(accepted=True,predicted_base_m=[x,0.,0.],
        predicted_orientation_base_xyzw=[0.,0.,0.,1.],
        pose_fit=dict(cameras=['center','left','right'][:views]))


def test_unanchored_two_views_cannot_authorize_handoff():
    model=Observer();model.result=prediction(2)
    gate=AnchoredSfpPort(model)
    assert not gate.predict_batch([(None,None,0,0)])[0]['accepted']


def test_consistent_three_view_anchor_allows_matching_partial_pose_only():
    model=Observer();gate=AnchoredSfpPort(model);item=(None,None,0,0)
    model.result=prediction(3)
    gate.predict_batch([item]);gate.predict_batch([item])
    model.result=prediction(2,.0003)
    assert gate.predict_batch([item])[0]['accepted']
    model.result=prediction(2,.04)
    assert not gate.predict_batch([item])[0]['accepted']


def test_new_request_and_reset_need_their_own_three_view_anchor():
    model=Observer();gate=AnchoredSfpPort(model);item=(None,None,0,0)
    model.result=prediction(3)
    gate.predict_batch([item]);gate.predict_batch([item])
    model.result=prediction(2)
    assert not gate.predict_batch([(None,None,0,1)])[0]['accepted']
    gate.reset_context()
    assert not gate.predict_batch([item])[0]['accepted']


def test_conflicting_three_view_pose_cannot_overwrite_existing_anchor():
    model=Observer();gate=AnchoredSfpPort(model);item=(None,None,0,0)
    model.result=prediction(3)
    gate.predict_batch([item]);gate.predict_batch([item])
    model.result=prediction(3,.04)
    assert not gate.predict_batch([item])[0]['accepted']
    model.result=prediction(2,.0001)
    assert gate.predict_batch([item])[0]['accepted']
