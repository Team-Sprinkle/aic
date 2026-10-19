#!/usr/bin/env python3
"""Bundle already-qualified RGB translation and orientation components.

Requires actual visual review of both components. Certificates contain only
scope, frame contracts and checksums, never simulator labels or desired poses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rotation-fit', type=Path, required=True)
    p.add_argument('--rotation-review', type=Path, required=True)
    p.add_argument('--translation-fit', type=Path, required=True)
    p.add_argument('--translation-review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--experimental-case')
    p.add_argument('--independent-frozen-replay', type=Path)
    a = p.parse_args()
    rotation = json.loads((a.rotation_review/'approval.json').read_text())
    position = json.loads((a.translation_review/'approval.json').read_text())
    experimental = a.experimental_case is not None
    if experimental:
        assert rotation['pose_gate_passed'] is False
        evidence = json.loads(a.independent_frozen_replay.read_text())
        near = evidence['phases']['hold_25mm']
        assert evidence['train_episode_and_scene_disjoint'] and near['episodes'] >= 4
        assert evidence['rotation_checkpoint_sha256'] == rotation['checkpoint_sha256']
        assert near['metrics']['orientation_error_deg']['episode_balanced_p50_p95'][1] <= 1
        assert near['metrics']['translation_error_mm']['episode_balanced_p50_p95'][1] <= .4
    else:
        assert rotation['pose_gate_passed']
    assert rotation['visual_inspection_completed']
    assert position['translation_gate_passed'] and position['visual_inspection_completed']
    assert position['orientation_approved'] is False
    for report, fit in [(rotation, a.rotation_fit), (position, a.translation_fit)]:
        assert report['checkpoint_sha256'] == digest(fit/'best.pt')
        assert report['pose_gate_scope'] == 'SC_held_near25mm_only'
        assert report['pose_frame_contract'] == 'SC_mating_reference_in_gripper_TCP'
    a.output.mkdir(parents=True, exist_ok=False)
    for name, fit in [('rotation', a.rotation_fit), ('translation', a.translation_fit)]:
        shutil.copyfile(fit/'best.pt', a.output/(name+'.pt'))
    common = dict(pose_gate_scope='SC_held_near25mm_only',
                  pose_frame_contract='SC_mating_reference_in_gripper_TCP',
                  visual_inspection_completed=True)
    rc = dict(common, pose_gate_passed=not experimental, prediction_kind='orientation_only',
              checkpoint_sha256=digest(a.output/'rotation.pt'))
    tc = dict(common, translation_gate_passed=True, orientation_approved=False,
              prediction_kind='translation_only_component',
              checkpoint_sha256=digest(a.output/'translation.pt'))
    if experimental:
        rc['experimental_trial_only'] = True
    for name, cert in [('rotation_approval', rc), ('translation_approval', tc)]:
        (a.output/(name+'.json')).write_text(json.dumps(cert, indent=2)+'\n')
    combined = dict(common, pose_gate_passed=not experimental,
                    prediction_kind='experimental_composite_pose' if experimental else 'composite_pose',
                    checkpoint_sha256=rc['checkpoint_sha256'],
                    rotation_approval='rotation_approval.json',
                    translation_approval='translation_approval.json',
                    translation_checkpoint='translation.pt',
                    rotation_approval_sha256=digest(a.output/'rotation_approval.json'),
                    translation_approval_sha256=digest(a.output/'translation_approval.json'))
    if experimental:
        combined.update(experimental_trial_only=True, allowed_development_case=a.experimental_case,
                        independent_frozen_replay_sha256=digest(a.independent_frozen_replay),
                        original_validation_gate_failed=True)
    (a.output/'approval.json').write_text(json.dumps(combined, indent=2)+'\n')
    (a.output/'selection.json').write_text(json.dumps(dict(checkpoint=str(a.output/'rotation.pt'),
                                                          approval=str(a.output/'approval.json')), indent=2)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(dict(rotation_review=str(a.rotation_review),
        translation_review=str(a.translation_review), labels_in_runtime_certificate=False,
        no_autonomous_insertion_asserted=True), indent=2)+'\n')
    print(a.output/'selection.json')


if __name__ == '__main__':
    main()
