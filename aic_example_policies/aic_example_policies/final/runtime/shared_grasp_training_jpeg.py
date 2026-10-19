"""Inference definitions extracted from shared_grasp_training_jpeg.py; no training entry point."""
import cv2

class SharedGraspTrainingJPEG:

    def __init__(self):
        self.original = None
        self.converted = None
        self.encodes = 0
        self.cache_hits = 0
        self.parallel = False

    def convert(self, images, *, begin=False):
        cameras = ('center', 'left', 'right')
        same = self.original is not None and all((self.original[c] is images[c] for c in cameras))
        if not begin and same:
            self.cache_hits += 1
            return self.converted
        if self.parallel:
            from .parallel_training_jpeg import normalise_triplet
            converted = normalise_triplet(images)
            self.encodes += 3
        else:
            converted = {}
            for c in cameras:
                rgb = images[c]
                if rgb.shape != (1024, 1152, 3) or str(rgb.dtype) != 'uint8':
                    raise ValueError('Original uint8 RGB triplet required')
                ok, encoded = cv2.imencode('.jpg', rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise ValueError('Native training JPEG86 conversion failed')
                converted[c] = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.encodes += 1
        self.original = {c: images[c] for c in cameras}
        self.converted = converted
        return converted

class SharedJPEGEstimator:

    def __init__(self, estimator, cache, *, begin):
        self.estimator = estimator
        self.cache = cache
        self.begin = begin

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def predict(self, images, *, already_training_jpeg=False, **kwargs):
        rgb = images if already_training_jpeg else self.cache.convert(images, begin=self.begin)
        return self.estimator.predict(rgb, already_training_jpeg=True, **kwargs)
