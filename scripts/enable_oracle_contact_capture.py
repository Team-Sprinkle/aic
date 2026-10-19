#!/usr/bin/env python3
"""Add read-only Gazebo contact reporting to a copied development scene."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--plugin',type=Path,required=True)
    a=p.parse_args()
    if not a.plugin.is_file():raise FileNotFoundError(a.plugin)
    world_file=a.audit/'world_audit.sdf';tree=ET.parse(world_file)
    world=tree.getroot().find('world')
    if world is None:raise ValueError('Missing world element')
    if any(x.get('name')=='aic_gazebo::OracleContactsPlugin' for x in world.findall('plugin')):
        raise ValueError('Oracle contact plugin already present')
    plugin=ET.SubElement(world,'plugin',{'filename':str(a.plugin.resolve()),
                                       'name':'aic_gazebo::OracleContactsPlugin'})
    ET.SubElement(plugin,'legacy_bullet_impulse_contacts').text='true'
    tree.write(world_file,encoding='unicode',xml_declaration=True)
    bridge_file=a.audit/'bridge_audit.yaml';bridge=yaml.safe_load(bridge_file.read_text())
    bridge.append({'ros_topic_name':'/audit/oracle/contacts',
                   'gz_topic_name':'/audit/oracle/contacts',
                   'ros_type_name':'ros_gz_interfaces/msg/Contacts',
                   'gz_type_name':'gz.msgs.Contacts','direction':'GZ_TO_ROS'})
    bridge_file.write_text(yaml.safe_dump(bridge,sort_keys=False))


if __name__=='__main__':main()
