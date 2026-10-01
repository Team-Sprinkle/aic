"""One explicit camera resize/codec contract for new transport BC and runtime."""
import cv2
import numpy as np
from PIL import Image
import torch


def jpeg_roundtrip_rgb(rgb, quality):
    ok, encoded=cv2.imencode('.jpg',np.ascontiguousarray(rgb[...,::-1]),
                             [cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:raise RuntimeError('BC image encoding failed')
    decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if decoded is None:raise RuntimeError('BC image decoding failed')
    return decoded[...,::-1]


def transport_rgb_tensor(rgb, *, native_jpeg_quality=None, native_already_encoded=False):
    if native_jpeg_quality is not None and not native_already_encoded:
        rgb=jpeg_roundtrip_rgb(rgb,native_jpeg_quality)
    intermediate=cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA)
    decoded=jpeg_roundtrip_rgb(intermediate,95)
    resized=np.asarray(Image.fromarray(decoded).resize((224,224),Image.Resampling.BILINEAR))
    return torch.from_numpy(np.array(resized,copy=True,order='C')).permute(2,0,1)
