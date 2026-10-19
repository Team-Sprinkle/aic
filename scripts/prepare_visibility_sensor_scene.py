#!/usr/bin/env python3
"""Add label-only, colocated RGBD sensors to a copied frozen robot description.

Original RGB cameras, link geometry, dynamics and controller remain unchanged.
Depth is supervision only; deployable policies may not subscribe to these topics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--aux-near-m', type=float, default=.07)
    p.add_argument('--exclude-own-camera-visual', action='store_true',
                   help='Label-only self-view mask; ordinary RGB sees all original visuals')
    p.add_argument('--world-follower', action='store_true',
                   help='Keep original robot; use independent visual-only camera models')
    p.add_argument('--primary-rgbd', action='store_true',
                   help='Diagnostic: turn copied primary camera into RGBD at its own link')
    a = p.parse_args()
    if a.primary_rgbd and a.world_follower:
        p.error('Primary RGBD and world follower are separate ablations')
    macro_source = a.sources / 'basler_camera_macro.xacro'
    root_source = a.sources / 'ur_gz.urdf.xacro'
    ET.register_namespace('xacro', 'http://wiki.ros.org/xacro')
    tree = ET.parse(macro_source)
    rgb = tree.getroot().find('.//sensor')
    assert rgb.attrib['type'] == 'camera'
    parent = next(e for e in tree.iter() if rgb in list(e))
    aux = ET.fromstring(ET.tostring(rgb))
    aux.set('name', '${name}_visibility_label_rgbd')
    aux.set('type', 'rgbd_camera')
    aux.find('update_rate').text = '2.0'
    aux.find('topic').text = '/audit/visibility/${name}'
    aux.find('camera').set('name', '${name}_visibility_label_rgbd')
    aux.find('camera/camera_info_topic').text = '/audit/visibility/${name}/camera_info'
    aux.find('camera/clip/near').text = str(a.aux_near_m)
    if a.exclude_own_camera_visual:
        # sdformat15 drops visual extensions on named URDF visuals. Unname only
        # this visual in the diagnostic description; mesh, pose and physics stay.
        for visual in tree.getroot().findall('.//link/visual'):
            visual.attrib.pop('name',None)
        bit="(2 if name == 'center_camera' else 4 if name == 'left_camera' else 8)"
        ET.SubElement(aux.find('camera'), 'visibility_mask').text = '${'+bit+'}'
        # sdformat URDF gazebo visual extension merges into the existing visual.
        macro=tree.getroot().find('{http://wiki.ros.org/xacro}macro')
        extension=ET.SubElement(macro,'gazebo',{'reference':'${tf_prefix}camera_link'})
        visual=ET.SubElement(extension,'visual')
        ET.SubElement(visual,'visibility_flags').text='${4294967295 - '+bit+'}'
    if a.primary_rgbd:
        rgb.set('type','rgbd_camera')
        rgb.find('topic').text='/${name}'
    elif not a.world_follower:
        parent.append(aux)
    tree.write(a.audit / 'basler_visibility_macro.xacro', encoding='unicode', xml_declaration=True)
    root = root_source.read_text()
    old = '$(find aic_assets)/models/Basler Camera/basler_camera_macro.xacro'
    assert root.count(old) == 1
    if not a.world_follower:
        (a.audit / 'ur_visibility_labels.xacro').write_text(root.replace(old, '/audit/basler_visibility_macro.xacro'))
    else:
        world_path=a.audit/'world_audit.sdf'
        world_tree=ET.parse(world_path);world=world_tree.getroot().find('world')
        ET.SubElement(world,'plugin',dict(filename='/repo/outputs/experiments/2026-09-30_transport_continuation/libLabelCameraFollowerPlugin.so',name='aic_gazebo::LabelCameraFollowerPlugin'))
        for camera in ('center','left','right'):
            model=ET.SubElement(world,'model',dict(name='label_only_'+camera))
            ET.SubElement(model,'static').text='true'
            link=ET.SubElement(model,'link',dict(name='camera_link'))
            sensor=ET.SubElement(link,'sensor',dict(name='label_only_'+camera,type='rgbd_camera'))
            ET.SubElement(sensor,'update_rate').text='2'
            ET.SubElement(sensor,'always_on').text='true'
            ET.SubElement(sensor,'topic').text=f'/audit/visibility/{camera}_camera'
            ET.SubElement(sensor,'frame_id').text=f'{camera}_camera/sensor_link'
            cam=ET.SubElement(sensor,'camera')
            # World-follower mode leaves the deployed robot description intact.
            # Its ordinary scene visuals keep Gazebo's default bit 1; bit 2
            # produced blank grey RGB/depth in the SFP pilot.
            ET.SubElement(cam,'visibility_mask').text='1'
            ET.SubElement(cam,'optical_frame_id').text=f'{camera}_camera/optical'
            ET.SubElement(cam,'horizontal_fov').text='.8718'
            im=ET.SubElement(cam,'image')
            for k,v in [('width','1152'),('height','1024'),('format','R8G8B8')]:ET.SubElement(im,k).text=v
            clip=ET.SubElement(cam,'clip')
            ET.SubElement(clip,'near').text=str(a.aux_near_m);ET.SubElement(clip,'far').text='20'
        world_tree.write(world_path,encoding='unicode',xml_declaration=True)
    bridge = yaml.safe_load((a.audit / 'bridge_audit.yaml').read_text())
    for camera in ('center', 'left', 'right'):
        suffixes=('depth_image',) if a.primary_rgbd else ('image', 'depth_image', 'camera_info')
        for suffix in suffixes:
            topic = (f'/{camera}_camera/{suffix}' if a.primary_rgbd else
                     f'/audit/visibility/{camera}_camera/{suffix}')
            info = suffix == 'camera_info'
            bridge.append(dict(ros_topic_name=topic, gz_topic_name=topic,
                               ros_type_name='sensor_msgs/msg/CameraInfo' if info else 'sensor_msgs/msg/Image',
                               gz_type_name='gz.msgs.CameraInfo' if info else 'gz.msgs.Image',
                               direction='GZ_TO_ROS', lazy=False))
    (a.audit / 'bridge_audit.yaml').write_text(yaml.safe_dump(bridge, sort_keys=False))
    metadata = dict(schema='aic_label_only_rgbd/v1', deployable_input=False,
                    rate_hz=2, width=1152, height=1024, horizontal_fov=0.8718,
                    original_rgb_sensors_unchanged=True,
                    auxiliary_near_clip_m=a.aux_near_m,
                    auxiliary_excludes_own_camera_visual=a.exclude_own_camera_visual,
                    auxiliary_world_follower=a.world_follower,
                    primary_rgbd_diagnostic=a.primary_rgbd,
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (macro_source,root_source)},
                    reference='https://github.com/gazebosim/gz-sensors/blob/gz-sensors9/src/RgbdCameraSensor.cc')
    (a.audit / 'visibility_sensor_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
