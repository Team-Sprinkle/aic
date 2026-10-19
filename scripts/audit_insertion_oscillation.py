#!/usr/bin/env python3
"""Bounded saved-trace SFP diagnosis; no models, scenes or labels are changed."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


def percentiles(x):
    return np.percentile(x,[50,90,95,100]).tolist() if len(x) else None


def excursions(times, depths, threshold=1.):
    """Direction changes only after >=threshold mm, so jitter is not a cycle."""
    if not len(depths):return []
    anchor=0;extreme=0;direction=0;out=[]
    for i in range(1,len(depths)):
        if not direction:
            if abs(depths[i]-depths[anchor])>=threshold:
                direction=int(np.sign(depths[i]-depths[anchor]));extreme=i
        elif direction*(depths[i]-depths[extreme])>=0:
            extreme=i
        elif direction*(depths[extreme]-depths[i])>=threshold:
            out.append(dict(start_s=float(times[anchor]),end_s=float(times[extreme]),
                change_mm=float(depths[extreme]-depths[anchor]),direction='toward' if direction>0 else 'away'))
            anchor=extreme;extreme=i;direction=-direction
    if direction:
        out.append(dict(start_s=float(times[anchor]),end_s=float(times[extreme]),
            change_mm=float(depths[extreme]-depths[anchor]),direction='toward' if direction>0 else 'away'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((a.comparison/'summary.json').read_text());live=[]
    for result in summary['results']:
        if result['family']!='sfp':continue
        folder=Path(result['audit'])
        trace=list(map(json.loads,(folder/'runtime_trace.jsonl').open()))
        plans=[t for t in trace if t['event']=='locked_local_perception']
        if not plans:continue
        lock=next(t for t in trace if t['event']=='shared_visual_lock')
        # Terminal retains the exact static estimated frame used for this attempt.
        port=np.asarray(result['terminal']['visual_port_matrix']);inverse=np.linalg.inv(port)
        tip=np.asarray(result['terminal']['body_tip_position_tcp_m'])
        commands=[t for t in trace if t['event']=='command' and t['sim_time']>=plans[0]['sim_time']]
        times=[];measured=[];commanded=[]
        for t in commands:
            current=inverse@matrix_from_xyz_xyzw(t['state'][:7])
            target=inverse@np.asarray(t['absolute_tcp_target'])
            times.append(t['sim_time']);measured.append((current[:3,3]+current[:3,:3]@tip)[2]*1000)
            commanded.append((target[:3,3]+target[:3,:3]@tip)[2]*1000)
        times=np.asarray(times);measured=np.asarray(measured);commanded=np.asarray(commanded)
        first=np.asarray([t['actor_control_point_targets_port'][0][:3] for t in plans])*1000
        current=np.asarray([t['predicted_tip_port_position_m'] for t in plans])*1000
        plan_times=np.asarray([t['sim_time'] for t in plans])
        delta=first[:,2]-current[:,2]
        prior=[]
        for t in plans:
            j=max(0,np.searchsorted(times,t['sim_time']-1.2,side='right')-1)
            now=(inverse@matrix_from_xyz_xyzw(t['state'][:7]))
            z=(now[:3,3]+now[:3,:3]@tip)[2]*1000
            prior.append(z-measured[j])
        cycles=excursions(times,measured)
        row=dict(arm=result['arm'],family='sfp',actor_observations=len(plans),
            full_insertion=result['physical_full_insertion'],backoff_observations=result['backoff_observations'],
            hierarchical_events=sum('backtrack' in t['event'] or 'lateral_retry' in t['event'] for t in trace),
            first_target_axial_delta_p50_p90_p95_max_mm=percentiles(delta),
            predicted_retreat_fraction_over1mm=float(np.mean(delta < -1)),
            predicted_retreat_after_forward_history_count=int(np.sum((delta < -1)&(np.asarray(prior)>1))),
            predicted_retreat_after_retreat_history_count=int(np.sum((delta < -1)&(np.asarray(prior)<-1))),
            command_minus_measured_axial_p50_p90_p95_max_mm=percentiles(commanded-measured),
            measured_depth_range_mm=[float(min(measured)),float(max(measured))],
            measured_excursions_over1mm=cycles,static_pose_lock_count=result['pose_lock_count'],
            trace=str(folder/'runtime_trace.jsonl'),video=str(folder/'five_cameras_20fps_grid_active.mp4'))
        live.append(row)
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        axes[0].plot(times,measured,label='Measured tip, estimated static port frame')
        axes[0].plot(times,commanded,alpha=.6,label='Commanded tip')
        axes[0].axhline(0,color='k',ls='--');axes[0].set_ylabel('Axial position (mm)');axes[0].legend()
        axes[1].plot(plan_times,delta,label='First proposed tip minus current tip')
        axes[1].axhline(0,color='k',ls='--');axes[1].set_ylabel('Proposed axial change (mm)')
        axes[1].set_xlabel('Simulation time (s)');axes[1].legend();fig.suptitle(result['arm']+' — no hardcoded recovery')
        fig.tight_layout();fig.savefig(a.output/(result['arm']+'.png'),dpi=130);plt.close(fig)
    # Recorded teacher command distributions, separated by physical reset split.
    buckets=defaultdict(list);episodes=defaultdict(set);history_sign=Counter();sfp_rows=0
    for r in map(json.loads,a.rows.open()):
        if r['family']!='sfp':continue
        sfp_rows+=1
        current=np.asarray(r['predicted_tip_port_pose9'][:3])*1000
        z=current[2]
        phase='inside' if z>0 else 'near_0_30' if z>=-30 else 'middle_30_90' if z>=-90 else 'far'
        target=np.asarray(r['teacher_tcp_targets_predicted_port_pose9'])
        # Declared schema: these are hybrid tip-position / TCP-orientation targets.
        delta=(target[0,2]*1000-z)
        key=r['split']+'/'+phase;buckets[key].append(delta);episodes[key].add(r['episode'])
    teacher={k:dict(rows=len(v),episodes=len(episodes[k]),
        first_target_axial_delta_p50_p90_p95_max_mm=percentiles(v),
        retreat_fraction_over1mm=float(np.mean(np.asarray(v)<-1)),
        forward_fraction_over1mm=float(np.mean(np.asarray(v)>1))) for k,v in buckets.items()}
    report=dict(live=live,teacher=teacher,teacher_sfp_rows=sfp_rows,
        measured_pose_source='Public TCP plus fixed visually estimated tip offset and port; not simulator geometry.',
        limitation='Seven variants on one reused SFP development configuration; descriptive, not generalization or proof that history is sufficient.',
        next='Compare opposite-direction live states to teacher support before any further policy fit.',
        source_modified=False,final_scenes_opened=False,new_RL=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(live_runs=len(live),teacher=teacher)),flush=True)


if __name__=='__main__':main()
