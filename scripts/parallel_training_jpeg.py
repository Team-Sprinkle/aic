"""Independent camera codec work in parallel, retaining exact JPEG86 pixels."""
from concurrent.futures import ThreadPoolExecutor
import cv2

_pool = ThreadPoolExecutor(max_workers=3,thread_name_prefix='aic-camera-codec')


def normalise(image):
    if image.shape!=(1024,1152,3) or str(image.dtype)!='uint8':
        raise ValueError('Original uint8 RGB required')
    ok,encoded = cv2.imencode('.jpg',image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
    if not ok:
        raise RuntimeError('Native RGB codec failed')
    return cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)


def normalise_triplet(images):
    cameras = ('center','left','right')
    futures = [_pool.submit(normalise,images[c]) for c in cameras]
    return {camera:future.result() for camera,future in zip(cameras,futures)}
