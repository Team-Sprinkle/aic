from setuptools import find_packages, setup

package_name = "gazebo_rl"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Team Sprinkle",
    maintainer_email="dev@team-sprinkle.invalid",
    description="Low-throughput Gazebo RL bridge for the AIC ROS/Gazebo stack",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gazebo_rl_rollout = gazebo_rl.rollout:main",
            "gazebo_rl_train_short = gazebo_rl.train:main",
        ],
    },
)
