from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from pathlib import Path
import yaml


def _load_yaml(package_name, relative_path):
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path

    path = Path(get_package_share_directory(package_name)) / relative_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_text(package_name, relative_path):
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path

    path = Path(get_package_share_directory(package_name)) / relative_path
    return path.read_text(encoding="utf-8")


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    description_file = LaunchConfiguration("description_file")
    controllers_file = LaunchConfiguration("controllers_file")
    launch_move_group = LaunchConfiguration("launch_move_group")

    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    "xacro ",
                    description_file,
                    " name:=ur",
                    " ur_type:=",
                    ur_type,
                    " simulation_controllers:=",
                    controllers_file,
                ]
            ),
            value_type=str,
        )
    }

    robot_description_semantic = {
        "robot_description_semantic": _load_text("aic_moveit_config", "config/aic.srdf")
    }

    kinematics_yaml = {
        "robot_description_kinematics": _load_yaml("aic_moveit_config", "config/kinematics.yaml")
    }
    ompl_yaml = {"ompl": _load_yaml("aic_moveit_config", "config/ompl_planning.yaml")}
    joint_limits_yaml = {
        "robot_description_planning": _load_yaml("aic_moveit_config", "config/joint_limits.yaml")
    }
    controllers_yaml = _load_yaml("aic_moveit_config", "config/moveit_controllers.yaml")
    moveit_cpp_yaml = _load_yaml("aic_moveit_config", "config/moveit_cpp.yaml")

    common_parameters = [
        robot_description,
        robot_description_semantic,
        kinematics_yaml,
        ompl_yaml,
        joint_limits_yaml,
        controllers_yaml,
        moveit_cpp_yaml,
        {"use_sim_time": True},
    ]

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=common_parameters,
        condition=IfCondition(launch_move_group),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument(
                "description_file",
                default_value=str(Path.cwd() / "aic_description" / "urdf" / "ur_gz.urdf.xacro"),
            ),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=str(Path.cwd() / "aic_bringup" / "config" / "aic_ros2_controllers.yaml"),
            ),
            DeclareLaunchArgument("launch_move_group", default_value="true"),
            move_group,
        ]
    )
