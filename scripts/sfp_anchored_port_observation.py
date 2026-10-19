"""Two-view updates require an earlier consistent THREE-view visual lock.

Unanchored two-view estimates may guide transport but cannot approve handoff.
The anchor is cleared at every reset/reacquisition, never across episodes.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def agrees(a,b,translation_m=.001,angle_deg=1.):
    return (np.linalg.norm(np.asarray(a['predicted_base_m'])-b['predicted_base_m'])<=translation_m and
        (Rotation.from_quat(a['predicted_orientation_base_xyzw']).inv()*
         Rotation.from_quat(b['predicted_orientation_base_xyzw'])).magnitude()<=np.deg2rad(angle_deg))


class AnchoredSfpPort:
    def __init__(self,observer):
        self.observer=observer;self.observer.minimum_views=2;self.optics=observer.optics
        self.reset_context()

    def reset_context(self):
        self.pending={};self.anchors={}
        if hasattr(self.observer,'reset_context'):self.observer.reset_context()

    def predict_batch(self,items):
        results=self.observer.predict_batch(items)
        for item,result in zip(items,results):
            key=tuple(item[2:4]);anchor=self.anchors.get(key)
            used=(result.get('pose_fit') or {}).get('cameras',[])
            result['three_view_anchor_available']=anchor is not None
            result['unanchored_pose_accepted']=result['accepted']
            if not result['accepted']:continue
            if len(used)==3:
                if anchor is not None and not agrees(result,anchor):
                    result['accepted']=False;result['anchor_reason']='three_view_conflicts_static_anchor'
                    continue
                previous=self.pending.get(key)
                if previous is not None and agrees(result,previous):
                    self.anchors[key]={k:result[k] for k in ('predicted_base_m','predicted_orientation_base_xyzw')}
                self.pending[key]={k:result[k] for k in ('predicted_base_m','predicted_orientation_base_xyzw')}
                result['anchor_reason']='current_three_view_pose'
            else:
                result['accepted']=anchor is not None and agrees(result,anchor)
                result['anchor_reason']=('two_view_agrees_prior_three_view_anchor' if result['accepted']
                    else 'two_view_has_no_matching_three_view_anchor')
            result['three_view_anchor_available']=key in self.anchors
        return results
