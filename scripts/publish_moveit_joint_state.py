#!/usr/bin/env python3
"""Publish the recorded Gazebo start as a current ROS joint state for offline MoveIt init."""

import argparse
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("joint_state", type=Path)
    args = parser.parse_args()
    source = json.loads(args.joint_state.read_text())
    rclpy.init()
    node = Node("aic_recorded_joint_state_source")
    publisher = node.create_publisher(JointState, "/joint_states", 10)

    def publish():
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = list(source["name"])
        message.position = list(source["position"])
        publisher.publish(message)

    node.create_timer(0.1, publish)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
