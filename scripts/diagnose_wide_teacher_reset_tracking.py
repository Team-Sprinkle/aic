#!/usr/bin/env python3
"""Posthoc reset-controller error and force traces; geometry never actor input."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_sc_port_targets import pose_matrix
from wide_approach_geometry import tcp_target_for_tip_pose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads((a.audit/'scored_stream.json').read_text())
    events=[json.loads(l) for l in (a.audit/'runtime_trace.jsonl').open()]
    start=next(e for e in events if e['event']=='supervisor_approach_reset_started')
    end=next(e for e in events if e['event']=='supervisor_approach_reset_settled')
    released=next(e for e in events if e['event']=='supervisor_approach_reset_released')
    r=start['requested'];goal=tcp_target_for_tip_pose(np.asarray(start['base_port_transform']),np.asarray(start['tcp_tip_transform']),r['axial_m'],r['lateral_xy_m'],np.deg2rad(r['rotation_xyz_deg']))
    samples=[]
    loop_start=end['time_sec']-end['iterations']*.05
    for stamp,m in d['streams']['/aic_controller/controller_state']:
        t=stamp/1e9
        if not start['time_sec']<=t<=end['time_sec']:continue
        def pose(value):return pose_matrix([*[value['position'][k] for k in 'xyz'],*[value['orientation'][k] for k in 'xyzw']])
        actual=pose(m['tcp_pose']);reference=pose(m['reference_tcp_pose'])
        samples.append(dict(time_s=t,error_mm=((goal[:3,3]-actual[:3,3])*1000).tolist(),reference_offset_mm=((reference[:3,3]-goal[:3,3])*1000).tolist(),velocity_mm_s=[m['tcp_velocity']['linear'][k]*1000 for k in 'xyz']))
    local=[s for s in samples if s['time_s']>=loop_start]
    error=np.asarray([s['error_mm'] for s in local]);velocity=np.asarray([s['velocity_mm_s'] for s in local])
    force=[]
    for stamp,m in d['streams']['/fts_broadcaster/wrench']:
        if start['time_sec']<=stamp/1e9<=end['time_sec']:
            force.append(np.linalg.norm([m['wrench']['force'][k] for k in 'xyz']))
    report=dict(audit=str(a.audit),reset=released,settling=end,loop_start_s=loop_start,
        error_norm_mm_p5_p50_p95=np.percentile(np.linalg.norm(error,axis=-1),[5,50,95]).tolist(),
        axis_error_range_mm=np.ptp(error,axis=0).tolist(),
        axis_error_zero_crossings=((error[1:]*error[:-1])<0).sum(0).tolist(),
        measured_speed_mm_s_p50_p95=np.percentile(np.linalg.norm(velocity,axis=-1),[50,95]).tolist(),
        raw_force_norm_n_p50_p95_max=np.percentile(force,[50,95,100]).tolist(),
        geometry_posthoc_only=True,final_scenes_opened=False)
    (a.output/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples));(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(10,6))
    t=[s['time_s'] for s in samples]
    for j,axis in enumerate('xyz'):
        axes[0].plot(t,[s['error_mm'][j] for s in samples],label=axis)
        axes[1].plot(t,[s['velocity_mm_s'][j] for s in samples],label=axis)
    axes[0].set_ylabel('Goal minus measured TCP (mm)');axes[1].set_ylabel('Measured velocity (mm/s)');axes[1].set_xlabel('Simulation time (s)')
    for ax in axes:ax.axvline(loop_start,color='black',linestyle='--');ax.grid();ax.legend()
    fig.tight_layout();fig.savefig(a.output/'tracking.png',dpi=140);plt.close(fig)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
