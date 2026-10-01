#!/usr/bin/env python3
"""Save all oracle collision pairs and independently stamped sensor snapshots."""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from ros_gz_interfaces.msg import Contacts
from geometry_msgs.msg import WrenchStamped
from aic_control_interfaces.msg import ControllerState


class Capture(Node):
    def __init__(self):
        super().__init__('oracle_contact_audit')
        self.output = open('/audit/oracle_contacts.jsonl', 'w', buffering=1)
        self.latest = {}
        self.subs = [self.create_subscription(Contacts, '/audit/oracle/contacts',
                      self.save, qos_profile_sensor_data)]
        for cls, topic, name in [(WrenchStamped, '/fts_broadcaster/wrench', 'raw_wrench'),
                                (ControllerState, '/aic_controller/controller_state', 'controller')]:
            self.subs.append(self.create_subscription(cls, topic,
                lambda msg, key=name: self.latest.update({key: message_to_ordereddict(msg)}),
                qos_profile_sensor_data))

    def save(self, msg):
        stamp = msg.header.stamp
        self.output.write(json.dumps({'sim_ns': stamp.sec * 10**9 + stamp.nanosec,
            'wall_ns': time.time_ns(), 'contact_count': len(msg.contacts),
            'contacts': message_to_ordereddict(msg)['contacts'],
            **self.latest}) + '\n')


def main():
    rclpy.init()
    node = Capture()
    try:
        rclpy.spin(node)
    finally:
        node.output.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
