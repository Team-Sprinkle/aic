#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]); rows=[json.loads(l) for l in (p/'corrected_contact_timeline.jsonl').open()];tr=[json.loads(l) for l in (p/'runtime_trace.jsonl').open()];samples={}
for r in tr:
 for c in r.get('recent_recovery_context',[])+r.get('recovery_context_samples',[]):samples[c['time_sec']]=c
s=sorted(samples.values(),key=lambda r:r['time_sec']);fig,ax=plt.subplots(3,1,figsize=(12,10),sharex=True)
t=[r['sim_ns']/1e9 for r in rows];forces=[max([c['max_point_normal_force_n'] for c in r['contacts'] if 'sc_plug_link' in c['pair'] and 'sc_port_1' in c['pair']],default=0) for r in rows]
ax[0].plot(t,forces,label='Largest plug/selected-port normal contact point (corrected N)',alpha=.7)
ax[0].plot([r['time_sec'] for r in s],[r.get('force_delta_n',np.nan) for r in s],label='Wrist vector change from baseline (N)');
for event in tr:
 if event['event']=='contact_detected':ax[0].scatter([event['time_sec']],[event['force_delta_n']],color='red',label='Recorded trigger')
ax[0].legend(fontsize=8);ax[0].set_ylabel('N (different quantities)')
for i,k in enumerate(['x','y','z']):
 ax[1].plot(t,[r.get('raw_wrench',{}).get('wrench',{}).get('force',{}).get(k,np.nan) for r in rows],label=k)
ax[1].legend();ax[1].set_ylabel('Raw wrist force N')
for key,label in [('tcp_pose','measured'),('reference_tcp_pose','controller reference')]:
 ax[2].plot(t,[1000*r.get('controller',{}).get(key,{}).get('position',{}).get('z',np.nan) if r.get('controller') else np.nan for r in rows],label=label)
ax[2].legend();ax[2].set_ylabel('TCP base Z mm');ax[2].set_xlabel('Simulation time s');ax[2].set_xlim(39,53)
for a in ax:a.grid(alpha=.3)
fig.suptitle('Contact reporting corrected with upstream #890; frozen physics and policy');fig.tight_layout();fig.savefig(p/'force_and_motion.png',dpi=150)
