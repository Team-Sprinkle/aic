#!/usr/bin/env python3
"""Add read-only contact reporting and close audit cameras to a copied world."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


def main(audit):
    world_path = audit / 'world_audit.sdf'
    tree = ET.parse(world_path)
    world = tree.getroot().find('world')
    plugin = ET.SubElement(world, 'plugin', {
        'filename': '/repo/outputs/experiments/2026-09-30_force_oracle/libOracleContactsPlugin.so',
        'name': 'aic_gazebo::OracleContactsPlugin'})
    # Explicitly tied to frozen official image's gz-physics 8.3.0.
    ET.SubElement(plugin, 'legacy_bullet_impulse_contacts').text = 'true'
    # Fixed world poses chosen for this frozen SC-port-1 development scene.
    target = [0.258, -0.313, 1.168]
    cameras = {}
    for name, eye in [('port_close', [0.56, -0.49, 1.29]),
                      ('port_reverse', [0.12, -0.53, 1.27])]:
        delta = [target[i] - eye[i] for i in range(3)]
        yaw = math.atan2(delta[1], delta[0])
        pitch = math.atan2(-delta[2], math.hypot(delta[0], delta[1]))
        pose = eye + [0, pitch, yaw]
        model = ET.SubElement(world, 'model', {'name': 'oracle_' + name})
        ET.SubElement(model, 'static').text = 'true'
        ET.SubElement(model, 'pose').text = ' '.join(map(str, pose))
        link = ET.SubElement(model, 'link', {'name': 'camera_link'})
        sensor = ET.SubElement(link, 'sensor', {'name': name, 'type': 'camera'})
        ET.SubElement(sensor, 'always_on').text = 'true'
        ET.SubElement(sensor, 'update_rate').text = '20'
        ET.SubElement(sensor, 'topic').text = '/audit/' + name + '/image'
        camera = ET.SubElement(sensor, 'camera')
        ET.SubElement(camera, 'horizontal_fov').text = '0.70'
        image = ET.SubElement(camera, 'image')
        ET.SubElement(image, 'width').text = '1280'
        ET.SubElement(image, 'height').text = '720'
        clip = ET.SubElement(camera, 'clip')
        ET.SubElement(clip, 'near').text = '0.01'
        ET.SubElement(clip, 'far').text = '10'
        cameras[name] = {'pose_world_xyz_rpy': pose, 'horizontal_fov': .70,
                         'width': 1280, 'height': 720, 'target_world_xyz': target}
    tree.write(world_path, encoding='unicode', xml_declaration=True)
    bridge_path = audit / 'bridge_audit.yaml'
    bridge = yaml.safe_load(bridge_path.read_text())
    for name in cameras:
        bridge.append({'ros_topic_name': '/audit/' + name + '/image',
                       'gz_topic_name': '/audit/' + name + '/image',
                       'ros_type_name': 'sensor_msgs/msg/Image',
                       'gz_type_name': 'gz.msgs.Image', 'direction': 'GZ_TO_ROS', 'lazy': True})
    bridge.append({'ros_topic_name': '/audit/oracle/contacts',
                   'gz_topic_name': '/audit/oracle/contacts',
                   'ros_type_name': 'ros_gz_interfaces/msg/Contacts',
                   'gz_type_name': 'gz.msgs.Contacts', 'direction': 'GZ_TO_ROS'})
    bridge_path.write_text(yaml.safe_dump(bridge, sort_keys=False))
    (audit / 'oracle_cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
