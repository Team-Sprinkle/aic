from types import SimpleNamespace
import base64

from gazebo_rl.observation import observation_to_dict


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_minimal_observation_conversion_tolerates_missing_fields():
    obs = ns()
    converted = observation_to_dict(obs, step_count=3)
    assert converted["step_count"] == 3
    assert converted["joints"]["position"] == []
    assert converted["wrist_wrench"]["force"] == [0.0, 0.0, 0.0]


def test_observation_conversion_extracts_state_fields():
    obs = ns(
        joint_states=ns(
            header=ns(stamp=ns(sec=2, nanosec=500_000_000)),
            name=["shoulder", "gripper_joint"],
            position=[1.0, 0.2],
            velocity=[0.1, 0.02],
            effort=[],
        ),
        wrist_wrench=ns(wrench=ns(force=ns(x=1, y=2, z=3), torque=ns(x=4, y=5, z=6))),
        controller_state=ns(
            tcp_pose=ns(position=ns(x=1, y=2, z=3), orientation=ns(x=0, y=0, z=0, w=1)),
            reference_tcp_pose=ns(position=ns(x=2, y=3, z=4), orientation=ns(x=0, y=0, z=0, w=1)),
            tcp_velocity=ns(linear=ns(x=0, y=0, z=0), angular=ns(x=0, y=0, z=0)),
            tcp_error=[0, 1, 2, 3, 4, 5],
            target_mode=ns(mode=7),
        ),
    )
    converted = observation_to_dict(obs)
    assert converted["sim_time"] == 2.5
    assert converted["gripper"]["position"] == 0.2
    assert converted["wrist_wrench"]["torque"] == [4.0, 5.0, 6.0]
    assert converted["controller"]["target_mode"] == 7


def test_observation_conversion_can_include_camera_images():
    raw = bytes([1, 2, 3, 4, 5, 6])
    image = ns(
        header=ns(stamp=ns(sec=1, nanosec=250_000_000)),
        height=1,
        width=2,
        encoding="rgb8",
        is_bigendian=0,
        step=6,
        data=raw,
    )
    obs = ns(left_image=image, center_image=image, right_image=image)

    converted = observation_to_dict(obs, include_images=True)

    center = converted["images"]["observation.images.center_camera"]
    assert center["height"] == 256
    assert center["width"] == 288
    assert center["encoding"] == "jpeg_rgb8"
    assert center["stamp"] == 1.25
    assert len(base64.b64decode(center["data_b64"])) < len(raw) * 1000
