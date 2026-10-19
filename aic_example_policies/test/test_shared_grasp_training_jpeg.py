"""Exact codec equivalence, no double encoding, and no stale mutable-frame reuse."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from shared_grasp_training_jpeg import SharedGraspTrainingJPEG,SharedJPEGEstimator


def test_exact_codec_and_scope():
    rng=np.random.default_rng(12)
    images={c:rng.integers(0,256,size=(1024,1152,3),dtype=np.uint8) for c in ('center','left','right')}
    cache=SharedGraspTrainingJPEG();first=cache.convert(images,begin=True)
    for c in images:
        ok,encoded=cv2.imencode('.jpg',images[c][...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86]);assert ok
        reference=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        assert np.array_equal(first[c],reference)
    assert cache.convert(images) is first
    assert cache.encodes==3 and cache.cache_hits==1
    images['center'][:]=0
    fresh=cache.convert(images,begin=True)
    assert fresh is not first and np.count_nonzero(fresh['center'])==0
    assert cache.encodes==6


def test_estimators_receive_same_preprocessed_frame_once():
    cache=SharedGraspTrainingJPEG();images={c:np.full((1024,1152,3),123,np.uint8) for c in ('center','left','right')}
    class Estimator:
        def predict(self,images,*,already_training_jpeg,tag):
            assert already_training_jpeg
            return images,tag
    pose=SharedJPEGEstimator(Estimator(),cache,begin=True);position=SharedJPEGEstimator(Estimator(),cache,begin=False)
    a,tag=pose.predict(images,tag='pose');b,_=position.predict(images,tag='position')
    assert a is b and tag=='pose' and cache.encodes==3
