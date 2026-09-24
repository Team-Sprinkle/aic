#!/usr/bin/env python3
"""Add visual-only fixed cameras to the pinned AIC world for diagnosis."""

from __future__ import annotations

from pathlib import Path
import sys


CAMERAS = (
    # Camera +X is optical forward in Gazebo. The overhead camera looks down.
    ("overhead", "0.24 0.02 1.90 0 1.570796 0", "1.65"),
    # The side camera looks from the near enclosure wall toward the board.
    ("side", "-0.45 -0.55 1.62 0 0.47 0.70", "1.55"),
)


def main(world_source: Path, bridge_source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    original = world_source.read_text()
    assert original.count("</world>") == 1
    models = []
    bridges = []
    for name, pose, fov in CAMERAS:
        models.append(f"""
    <!-- Visual-only audit camera. Static and collision-free. -->
    <model name="audit_{name}_camera">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="camera_link">
        <sensor name="audit_{name}" type="camera">
          <always_on>true</always_on>
          <update_rate>20</update_rate>
          <topic>/audit/{name}/image</topic>
          <camera>
            <horizontal_fov>{fov}</horizontal_fov>
            <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
            <clip><near>0.02</near><far>5.0</far></clip>
          </camera>
        </sensor>
      </link>
    </model>""")
        bridges.append(f"""- ros_topic_name: "/audit/{name}/image"
  gz_topic_name: "/audit/{name}/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS
  lazy: true
""")
    (output / "world_audit.sdf").write_text(original.replace("</world>", "\n".join(models) + "\n  </world>"))
    (output / "bridge_audit.yaml").write_text(bridge_source.read_text() + "\n".join(bridges))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
