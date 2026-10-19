#!/usr/bin/env python3
"""Paired 150/25mm RGB grasp estimates; raw image-time labels are offline only."""
import cv2
cv2.setNumThreads(2)
import argparse
import bisect
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from prepare_sfp_native_grasp_labels import label, matrix, stored_matrix
from native_connector_pose_observation import NativeConnectorPoseObservation
from sc_native_grasp_observation import SCNativeGraspObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    report = json.loads((args.runs/'summary.json').read_text())
    checkpoint = Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sc_orientation_jitter_v2/best.pt')
    rotation_model = NativeConnectorPoseObservation(checkpoint, 'cpu')
    position_model = SCNativeGraspObservation('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_translation_observation_v1.pt', 'cpu')
    optics = {c: np.asarray(x) for c, x in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    typ = get_message('tf2_msgs/msg/TFMessage')
    records = []
    rejected = []
    for result in report['results']:
        audit = Path(result['audit'])
        data = json.loads((audit/'scored_stream.json').read_text())
        events = [json.loads(line) for line in (audit/'runtime_trace.jsonl').open()]
        far = next(e for e in events if e['event'] == 'shared_visual_lock')['sim_time']
        near = next(e for e in events if e['event'] == 'near_handoff_ready')['sim_time']
        world_base = stored_matrix(data['edges']['world|tabletop'][0][1]) @ stored_matrix(data['edges']['tabletop|base_link'][0][1])
        with next((audit/'results').glob('bag_*/*.mcap')).open('rb') as raw:
            reader = make_reader(raw)
            world_aic = None
            for _, _, message in reader.iter_messages(topics=['/tf_static']):
                for transform in deserialize_message(message.data, typ).transforms:
                    if (transform.header.frame_id, transform.child_frame_id) == ('world', 'aic_world'):
                        world_aic = matrix(transform.transform)
            if world_aic is None:
                raise ValueError('Missing world/aic calibration')
            for phase, end in [('acquisition_150mm', far), ('hold_25mm', near)]:
                samples = [x for x in data['cameras']['center'] if end-.30 <= x[0]/1e9 <= end]
                samples = [samples[i] for i in np.linspace(0, len(samples)-1, min(5, len(samples)), dtype=int)]
                for stamp, _ in samples:
                    images = {}; stamps = {}
                    for camera in ['center', 'left', 'right']:
                        index = bisect.bisect_right(data['cameras'][camera], stamp, key=lambda x:x[0])-1
                        if index < 0:
                            break
                        stamps[camera], images[camera] = data['cameras'][camera][index]
                    if len(images) != 3:
                        rejected.append(dict(episode=result['case'], phase=phase,
                                             stamp_ns=stamp, reason='missing_causal_camera'))
                        continue
                    row = dict(family='sc', episode=result['case'], source_episode_name=result['case'], scene=data['scene_group_id'],
                               split='development', time_s=stamp/1e9, images=images,
                               image_stamps_ns=stamps, source_stream=str(audit/'scored_stream.json'))
                    try:
                        truth = label(reader, typ, data, row, world_aic, world_base, optics)
                    except ValueError as error:
                        rejected.append(dict(episode=result['case'], phase=phase,
                                             stamp_ns=stamp, reason=str(error)))
                        continue
                    pixels = {c: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for c, path in images.items()}
                    # Capture frames are already JPEG86, matching the live helper
                    # conversion from raw RGB. Do not encode them again.
                    orientation = np.asarray(rotation_model.predict(pixels, already_training_jpeg=True,
                        visual_tip_position_tcp_m=np.zeros(3))['orientation_for_diagnostic_only'])
                    translation = np.asarray(position_model.predict(pixels, already_training_jpeg=True)['predicted_tip_position_tcp_m'])
                    body = np.asarray(truth['body_tip_matrix'])
                    records.append(dict(row, phase=phase, image_to_label_age_ms=truth['image_to_label_age_ms'],
                        predicted_rotation=orientation.tolist(), predicted_translation_m=translation.tolist(),
                        truth_pose=body.tolist(), translation_error_mm=float(np.linalg.norm(translation-body[:3,3])*1000),
                        orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(orientation.T@body[:3,:3]).magnitude())),
                        geometry_label_only=True, deployment_updated=False))
    phases = {}
    for phase in ['acquisition_150mm', 'hold_25mm']:
        rows = [r for r in records if r['phase'] == phase]
        phases[phase] = dict(samples=len(rows), episodes=len({r['episode'] for r in rows}),
            rejected_samples=sum(r['phase'] == phase for r in rejected),
            translation_p50_p95_mm=np.percentile([r['translation_error_mm'] for r in rows],[50,95]).tolist() if rows else None,
            orientation_p50_p95_deg=np.percentile([r['orientation_error_deg'] for r in rows],[50,95]).tolist() if rows else None)
    (args.output/'records.json').write_text(json.dumps(records)+'\n')
    (args.output/'rejected_frames.json').write_text(json.dumps(rejected, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(phases=phases, geometry_labels_only=True,
        no_model_training=True, no_deployment_pose_updates=True, checkpoint=str(checkpoint),
        preprocessing='Recorded native JPEG86 -> no second conversion; matching live raw RGB -> JPEG86',
        interpretation='Paired exposed development observations; consistency alone does not establish absolute pose accuracy.'),indent=2)+'\n')
    print(json.dumps(phases), flush=True)


if __name__ == '__main__':
    main()
