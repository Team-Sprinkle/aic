from typing import TypedDict

MotionUpdateActionDict = TypedDict(
    "MotionUpdateActionDict",
    {
        "delta_position.x": float,
        "delta_position.y": float,
        "delta_position.z": float,
        "delta_rotation.x": float,
        "delta_rotation.y": float,
        "delta_rotation.z": float,
    },
)

JointMotionUpdateActionDict = TypedDict(
    "JointMotionUpdateActionDict",
    {
        "shoulder_pan_joint": float,
        "shoulder_lift_joint": float,
        "elbow_joint": float,
        "wrist_1_joint": float,
        "wrist_2_joint": float,
        "wrist_3_joint": float,
    },
)
