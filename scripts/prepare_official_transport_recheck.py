#!/usr/bin/env python3
"""Freeze repeated public qualification scenes and clearly separate range stress."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    repo = Path.cwd()
    url = 'https://raw.githubusercontent.com/intrinsic-dev/aic/main/aic_engine/config/eval_config.yaml'
    raw = urllib.request.urlopen(url, timeout=30).read()
    (a.output/'official_eval_config.yaml').write_bytes(raw)
    official = yaml.safe_load(raw)
    # Keep an exact download and a pinned, externally verifiable provenance.
    commit = json.load(urllib.request.urlopen(
        'https://api.github.com/repos/intrinsic-dev/aic/commits?path=aic_engine/config/eval_config.yaml&per_page=1', timeout=30))[0]['sha']
    bundle = json.loads((a.prior_root/'shared_perception_bundle_approved_v1.json').read_text())
    old_sc = bundle['sc_native_refiner']
    bundle['sc_native_refiner'] = str(a.prior_root/'native_pose_SC_rail_repair_v1/best.pt')
    frozen = a.output/'checkpoints'
    frozen.mkdir()
    for key in ('backbone', 'sc_coarse', 'sfp_coarse', 'sc_native_refiner', 'sfp_native_refiner', 'mount_priors'):
        src = Path(bundle[key]); dest = frozen/(key+src.suffix)
        shutil.copyfile(src, dest); bundle[key] = str(dest)
    shutil.copyfile(old_sc, frozen/'baseline_sc_native_refiner.pt')
    template_case = json.loads((a.prior_root/'SC_fresh60_acquisition_plan_v1.json').read_text())['cases'][0]
    template = Path(template_case['scene'])
    source = (repo/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
    initial = official['robot']['home_joint_positions']
    for key, value in initial.items():
        source, n = re.subn(rf'({re.escape(key)}=)(-?[0-9]+\.[0-9]+)', rf'\g<1>{value:.7f}', source, count=1)
        if n != 1:
            raise ValueError('Cannot set official first-trial robot start: '+key)
    rng = np.random.default_rng(2026101601)
    cases = []

    def add(trial_key, trial, name, domain, rep, arm='candidate'):
        config = copy.deepcopy(official)
        config['trials'] = {trial_key: copy.deepcopy(trial)}
        scene = a.output/'scenes'/name
        scene.mkdir(parents=True)
        (scene/'eval_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (scene/'ur_llb_start.xacro').write_text(source)
        for filename in ('world_audit.sdf', 'bridge_audit.yaml'):
            shutil.copyfile(template/filename, scene/filename)
        task = next(iter(trial['tasks'].values()))
        board = trial['scene']['task_board']
        family = 'sc' if task['plug_type']=='sc' else 'sfp'
        present = [i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        port = int((task['target_module_name'] if family=='sc' else task['port_name']).rsplit('_', 1)[1])
        case = dict(case=name, scene=str(scene), family=family, domain=domain,
                    card_count=len(present), present_cards=present, port=port,
                    requested_card=None if family=='sc' else int(task['target_module_name'].rsplit('_', 1)[1]),
                    replicate=rep, arm=arm, official_trial=trial_key,
                    configured_start_joints=initial, board_pose=board['pose'],
                    exact_public_scene=domain=='exact_public', final_scene=False)
        case['scene_hash'] = sha(scene/'eval_config.yaml')
        if domain=='exact_public':
            assert config['trials'][trial_key] == official['trials'][trial_key]
            assert {k:v for k,v in config.items() if k!='trials'} == {k:v for k,v in official.items() if k!='trials'}
        cases.append(case)

    # Interleave trial types: five fresh simulator launches of each released scene.
    for rep in range(5):
        for trial_key, trial in official['trials'].items():
            add(trial_key, trial, f'exact_{trial_key}_r{rep}', 'exact_public', rep)
        add('trial_3', official['trials']['trial_3'], f'exact_trial_3_baseline_r{rep}',
            'exact_public', rep, arm='baseline_SC_refiner')
    # Published qualification has only three layouts. Other counts/ports are
    # released-range stress cases, never called exact official evaluation.
    for rep in range(5):
        for count in range(1, 6):
            for port in range(2):
                for family in ('sc', 'sfp'):
                    key = 'trial_3' if family=='sc' else f'trial_{port+1}'
                    trial = copy.deepcopy(official['trials'][key])
                    board = trial['scene']['task_board']
                    present = {(rep+i)%5 for i in range(count)}
                    for i in range(5):
                        board[f'nic_rail_{i}'] = dict(entity_present=i in present)
                        if i in present:
                            board[f'nic_rail_{i}'].update(entity_name=f'nic_card_{i}', entity_pose=dict(
                                translation=float(rng.uniform(-.0215, .0234)), roll=0., pitch=0., yaw=0.))
                    for i in range(2):
                        board[f'sc_rail_{i}'] = dict(entity_present=True, entity_name=f'sc_mount_{i}',
                            entity_pose=dict(translation=float(rng.uniform(-.06, .055)), roll=0., pitch=0., yaw=0.))
                    task = next(iter(trial['tasks'].values()))
                    if family=='sc':
                        task['target_module_name'] = f'sc_port_{port}'
                        board[f'sc_rail_{port}']['entity_pose']['translation'] = float(np.linspace(-.06, .055, 5)[rep])
                    else:
                        task['target_module_name'] = f'nic_card_mount_{rep}'
                        task['port_name'] = f'sfp_port_{port}'
                        board[f'nic_rail_{rep}']['entity_pose']['translation'] = float(np.linspace(-.0215, .0234, 5)[rep])
                    add(key, trial, f'range_{family}_n{count}_p{port}_r{rep}', 'published_range_stress', rep)
    # Freeze the actual imported source tree against inter-episode edits.
    watched = list((repo/'aic_example_policies/aic_example_policies').rglob('*.py'))
    watched += list((repo/'scripts').glob('*.py')) + list((repo/'scripts').glob('*.sh'))
    hashes = {str(f):sha(f) for f in watched}
    plan = dict(schema='official_transport_recheck/v1', official_url=url, official_file_commit=commit,
        official_sha256=sha(a.output/'official_eval_config.yaml'), bundle=bundle,
        baseline_sc_checkpoint=str(frozen/'baseline_sc_native_refiner.pt'),
        frozen_checkpoint_hashes={str(f):sha(f) for f in frozen.iterdir()}, runtime_hashes=hashes,
        cases=cases, planned=len(cases),
        exact_scope='Five independent launches per unchanged public trial; trial isolated from three-trial YAML. Same official home/grasp/board/rails/task/time limit. Added passive audit cameras/recorders, no collision edits. Not an unchanged sequential cloud submission.',
        stress_scope='100 released-range derivatives: SC/SFP x1..5cards x2ports x5replicates. Published board/home/grasp fixed by source trial; rail position/card subset/request vary. Not exact official scenes or exhaustive later evaluation.',
        gate='No tuning; keep functional failures. Count correct visual acquisition, strict pose/standoff/noharm/300ms gate, and failure reason separately. No insertion or RL.',
        budget='120 sequential launches; <=300 policy-wall seconds/run plus startup/scoring, ~6-12 hours. GPU3 render/GPU4 inference, legacy GPU0 untouched; no training.',
        reserved_final_scenes_opened=False, insertion_actor_enabled=False)
    (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),planned=len(cases),official_sha256=plan['official_sha256'])))


if __name__=='__main__':
    main()
