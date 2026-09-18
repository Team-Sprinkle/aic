import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "aic_model"
spec = importlib.util.spec_from_file_location("rollout_recording", ROOT / "rollout_recording.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_snapshots_use_sim_time_and_decode_padded_rgb_rows(tmp_path):
    pixels = np.zeros((16, 16, 3), np.uint8)
    pixels[:, :, 0] = 255
    padded = np.zeros((16, 52), np.uint8)
    padded[:, :48] = pixels.reshape(16, 48)
    image = NS(height=16, width=16, step=52, encoding="rgb8", data=padded.tobytes(),
               header=NS(stamp=NS(sec=1, nanosec=0)))
    obs = NS(center_image=image, left_image=image, right_image=image,
             joint_states=NS(name=['wrist_1_joint', 'gripper/finger_joint'], position=[-.2, .0035]),
             controller_state=NS(tcp_pose=NS(position=NS(x=1, y=2, z=3))))
    recording = module.RolloutSnapshots(tmp_path)
    assert recording.capture(obs) is obs
    recording.capture(obs)
    image.header.stamp.sec = 2
    recording.capture(obs)
    recording.close()
    rows = [json.loads(line) for line in recording.directory.joinpath("frames.jsonl").read_text().splitlines()]
    assert [row["sim_time"] for row in rows] == [1, 2]
    decoded = cv2.imread(str(recording.directory / rows[0]["images"]["center"]))
    assert decoded[:, :, 2].mean() > 250
    assert decoded[:, :, 0].mean() < 5
    assert rows[-1]["tcp_position"] == [1, 2, 3]
    assert rows[0]["joint_names"] == ['wrist_1_joint', 'gripper/finger_joint']
    assert rows[0]["joint_positions"] == [-.2, .0035]


def test_policy_thread_exception_becomes_failed_result(monkeypatch):
    monkeypatch.delenv("AIC_POLICY_RECORD_DIR", raising=False)
    tree = ast.parse((ROOT / "aic_model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AicModel")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "action_thread_func")
    fn.args.args[1].annotation = None
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "policy_thread", "exec"), namespace)
    errors = []
    def fail(**kwargs):
        raise RuntimeError("broken checkpoint")
    node = NS(_policy=NS(insert_cable=fail), get_logger=lambda: NS(error=errors.append), _action_thread_result=None)
    namespace["action_thread_func"](node, NS(request=NS(task=None)))
    assert node._action_thread_result is False
    assert "broken checkpoint" in errors[0]
