#!/usr/bin/env python3
"""Small-rollout target discontinuity and measured-motion review."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import re
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


def pct(x):return np.percentile(x,[50,95,99,100]).tolist() if len(x) else None


def completion_time(folder):
    """Map simulator completion wall log to nearby recorded camera sim time."""
    log=(folder/'engine.log').read_text(errors='replace')
    matches=re.findall(r'\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\).*Cable transitioning to COMPLETED state',log)
    if not matches:return None
    wall=int(datetime.fromisoformat(matches[-1]).replace(tzinfo=timezone.utc).timestamp()*1e9)
    frames=[r for r in csv.DictReader((folder/'frame_manifest.csv').open()) if r['camera']=='center']
    nearest=min(frames,key=lambda r:abs(int(r['wall_ns'])-wall))
    if abs(int(nearest['wall_ns'])-wall)>200_000_000:raise ValueError('Completion-to-camera clock join too distant')
    return dict(sim_time=int(nearest['sim_ns'])/1e9,wall_time_ns=wall,
        camera_wall_difference_ms=(int(nearest['wall_ns'])-wall)/1e6,
        note='Gazebo CablePlugin locks the connector/end-effector after successful contact; later zero motion is not evidence of active policy stabilization.')


def review(row,output,tag):
    import matplotlib.pyplot as plt
    folder=Path(row['audit']);trace=list(map(json.loads,(folder/'runtime_trace.jsonl').open()))
    plans=[t for t in trace if t['event']=='locked_local_perception']
    if not plans:return dict(tag=tag,actor_ran=False)
    commands=[t for t in trace if t['event']=='command' and t['sim_time']>=plans[0]['sim_time']]
    ports=np.asarray(row['terminal']['visual_port_matrix']);inverse=np.linalg.inv(ports)
    body=np.asarray(row['terminal']['body_tip_position_tcp_m'])
    values={k:[] for k in ['within_mm','boundary_mm','within_deg','boundary_deg','boundary_extrapolation_residual_mm']}
    times=[];position=[];tip=[];target=[];speed=[];tracking=[];measured_rotations=[]
    completed=completion_time(folder)
    boundary_windows={'first2s':[],'first5s':[],'first15s':[],'first20s':[],'before_simulator_completion':[]}
    for c in commands:
        pose=matrix_from_xyz_xyzw(c['state'][:7]);goal=np.asarray(c['absolute_tcp_target'])
        current=inverse@pose;desired=inverse@goal
        times.append(c['sim_time']);position.append(pose[:3,3]);measured_rotations.append(pose[:3,:3])
        tip.append(current[:3,3]+current[:3,:3]@body);target.append(desired[:3,3]+desired[:3,:3]@body)
        speed.append(np.linalg.norm(c['state'][7:10])*1000)
        tracking.append(np.linalg.norm(goal[:3,3]-pose[:3,3])*1000)
    for i,(a,b) in enumerate(zip(commands,commands[1:])):
        kind='boundary' if b['chunk_index']==0 else 'within'
        aa=np.asarray(a['absolute_tcp_target']);bb=np.asarray(b['absolute_tcp_target'])
        values[kind+'_mm'].append(np.linalg.norm(aa[:3,3]-bb[:3,3])*1000)
        if kind=='boundary':
            for name,seconds in [('first2s',2),('first5s',5),('first15s',15),('first20s',20)]:
                if b['sim_time']-plans[0]['sim_time']<=seconds:
                    boundary_windows[name].append(values[kind+'_mm'][-1])
            if completed is None or b['sim_time']<=completed['sim_time']:
                boundary_windows['before_simulator_completion'].append(values[kind+'_mm'][-1])
        values[kind+'_deg'].append((Rotation.from_matrix(aa[:3,:3]).inv()*Rotation.from_matrix(bb[:3,:3])).magnitude()*180/np.pi)
        if kind=='boundary' and i>0 and commands[i-1]['chunk_index']==a['chunk_index']-1:
            before=commands[i-1];dt=a['sim_time']-before['sim_time'];gap=b['sim_time']-a['sim_time']
            if .02<=dt<=.20 and .02<=gap<=.30:
                prior=np.asarray(before['absolute_tcp_target'])
                expected=aa[:3,3]+(aa[:3,3]-prior[:3,3])*gap/dt
                values['boundary_extrapolation_residual_mm'].append(np.linalg.norm(bb[:3,3]-expected)*1000)
    times=np.asarray(times);tip=np.asarray(tip)*1000;target=np.asarray(target)*1000
    # Resample public TCP position onto a 50ms grid. Long sensor gaps are
    # excluded. This is a numerical motion proxy, not a force-sensor metric.
    times_unique,indices=np.unique(times,return_index=True);position=np.asarray(position)[indices]
    grid=np.arange(times_unique[0],times_unique[-1],.05)
    interpolated=np.stack([np.interp(grid,times_unique,position[:,k]) for k in range(3)],axis=1)
    right=np.searchsorted(times_unique,grid).clip(1,len(times_unique)-1)
    valid=(times_unique[right]-times_unique[right-1])<=.20
    acceleration=np.diff(interpolated,n=2,axis=0)/.05**2
    jerk=np.diff(interpolated,n=3,axis=0)/.05**3
    valid_a=valid[:-2]&valid[1:-1]&valid[2:];valid_j=valid[:-3]&valid[1:-2]&valid[2:-1]&valid[3:]
    # Rotation-direction-independent velocity from position differences.
    changes=np.diff(tip[:,2]);direction=np.where(changes>.1,1,np.where(changes<-.1,-1,0))
    direction=direction[direction!=0]
    reversals=int(np.sum(direction[1:]!=direction[:-1]))
    report=dict(tag=tag,case=row['case'],family=row['family'],actor_ran=True,
        physical_full_insertion=row['physical_full_insertion'],
        terminal={k:row['raw_terminal'][k] for k in ['depth_mm','lateral_mm','orientation_error_deg','official_tier3','no_scored_harm']},
        command_jump_p50_p95_p99_max={k:pct(v) for k,v in values.items()},
        boundary_jump_window_p50_p95_p99_max_mm={k:pct(v) for k,v in boundary_windows.items()},
        simulator_completion=completed,
        measured_TCP_speed_before_completion_p50_p95_p99_max_mm_s=pct(np.asarray(speed)[times<=completed['sim_time']]) if completed else pct(speed),
        measured_TCP_speed_p50_p95_p99_max_mm_s=pct(speed),
        measured_TCP_position_second_difference_p50_p95_p99_max_m_s2=pct(np.linalg.norm(acceleration[valid_a],axis=1)),
        measured_TCP_position_third_difference_p50_p95_p99_max_m_s3=pct(np.linalg.norm(jerk[valid_j],axis=1)),
        numerical_derivative_note='50ms linear-resampled public TCP position, gaps>200ms excluded. Raw finite differences include sampling/noise artifacts; compare only with identically processed baseline.',
        command_tracking_error_p50_p95_p99_max_mm=pct(tracking),
        estimated_tip_axial_reversals_over_point1mm=reversals,
        pose_locks=row.get('pose_lock_count'),backoff_observations=row['backoff_observations'],
        requested_port_locks_passed=row['requested_port_locks_passed'],
        complete_chunk_command_indices=dict(Counter(c['chunk_index'] for c in commands)),
        actor_latency=row['phase_latency']['actor'],
        video=str(folder/'five_cameras_20fps_grid_active.mp4'))
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].plot(times,tip[:,2],label='Measured tip in fixed estimated port');axes[0].plot(times,target[:,2],alpha=.6,label='Commanded tip')
    axes[0].axhline(0,color='k',ls='--');axes[0].set_ylabel('Axial (mm)');axes[0].legend()
    axes[1].plot(times,np.linalg.norm(tip[:,:2],axis=1),label='Estimated measured lateral')
    axes[1].plot(times,np.linalg.norm(target[:,:2],axis=1),label='Commanded lateral',alpha=.6);axes[1].set_ylabel('Lateral (mm)');axes[1].legend()
    axes[2].plot(times,speed,label='Public TCP speed');axes[2].set_ylabel('Speed (mm/s)');axes[2].set_xlabel('Simulation seconds');axes[2].legend()
    fig.suptitle(tag);fig.tight_layout();fig.savefig(output/(tag+'.png'),dpi=120);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    source=a.root/'runs/summary.json'
    if not source.exists():source=a.root/'runs/progress.json'
    new=json.loads(source.read_text())['results']
    if len(new)!=4:raise ValueError('Wait for all four scored rollouts')
    baseline=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1/live_comparison/summary.json').read_text())['results']
    rows=[]
    for i,r in enumerate(new):rows.append(review(r,a.output,f'new_{i+1}_{r["family"]}'))
    for r in baseline:
        if r['arm']=='h4_e4':rows.append(review(r,a.output,'baseline_'+r['family']))
    result=dict(new_rollouts=len(new),new_full_insertions=sum(x['physical_full_insertion'] for x in new),
        results=rows,scope='Four repeated development rollouts, two configurations. Baseline is a saved single repeat per configuration; no paired-seed statistical claim.',
        recovery='none',new_RL=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
