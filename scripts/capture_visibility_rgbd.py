#!/usr/bin/env python3
"""Capture timestamp-matched native RGB/depth pairs for visibility supervision."""
import json
import os
from pathlib import Path
import time
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo


class Capture(Node):
    def __init__(self):
        super().__init__('label_only_visibility_capture')
        self.bridge = CvBridge()
        self.directory = Path('/audit/visibility_frames')
        self.directory.mkdir(exist_ok=True)
        self.log = Path('/audit/visibility_pairs.jsonl').open('w', buffering=1)
        self.pending = {camera:{} for camera in ('center','left','right')}
        self.intrinsics = {}
        self.subs = []
        for camera in self.pending:
            root = (f'/{camera}_camera' if os.environ.get('AIC_VISIBILITY_SOURCE_ROOT')=='primary'
                    else f'/audit/visibility/{camera}_camera')
            for kind in ('image','depth_image'):
                self.subs.append(self.create_subscription(Image,root+'/'+kind,
                    lambda msg,c=camera,k=kind:self.save(c,k,msg),qos_profile_sensor_data))
            self.subs.append(self.create_subscription(CameraInfo,root+'/camera_info',
                lambda msg,c=camera:self.info(c,msg),qos_profile_sensor_data))

    def info(self,camera,msg):
        self.intrinsics[camera] = dict(width=msg.width,height=msg.height,k=list(msg.k),
                                     p=list(msg.p),d=list(msg.d),frame_id=msg.header.frame_id)

    def save(self,camera,kind,msg):
        stamp = msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
        items = self.pending[camera]
        pair = items.setdefault(stamp,dict(sim_ns=stamp,wall_ns=time.time_ns(),camera=camera))
        filename = f'{stamp}_{camera}'
        if kind == 'image':
            filename += '.png'
            assert cv2.imwrite(str(self.directory/filename),self.bridge.imgmsg_to_cv2(msg,'bgr8'))
            pair['rgb'] = filename
        else:
            # Physical depth in metres. Never quantize to 8-bit or normalize per frame.
            assert msg.encoding == '32FC1', msg.encoding
            depth = self.bridge.imgmsg_to_cv2(msg,'passthrough')
            filename += '.npz'
            np.savez_compressed(self.directory/filename,depth_m=depth.astype(np.float32))
            pair['depth'] = filename
        if 'rgb' in pair and 'depth' in pair:
            pair['intrinsics'] = self.intrinsics.get(camera)
            self.log.write(json.dumps(pair)+'\n')
            del items[stamp]
        # Bound cache if a stream is absent; save diagnostics rather than joining wrong frames.
        for stale in [s for s in items if stamp-s>2000000000]:
            del items[stale]


def main():
    rclpy.init()
    node=Capture()
    try:rclpy.spin(node)
    finally:
        node.log.close();node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':main()
