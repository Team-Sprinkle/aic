#!/usr/bin/env python3
"""Review exact recorded camera frames at selected public trace timestamps."""
import cv2
cv2.setNumThreads(2)
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();roll=a.root/'rollout';rows=list(csv.DictReader((roll/'frame_manifest.csv').open()));events=[json.loads(s) for s in (roll/'runtime_trace.jsonl').open()];output=a.root/'review';output.mkdir(exist_ok=True);ends=[x for x in events if x['event']=='terminal_before_return' and x.get('observation_source')=='retained_public_observation_before_return'];start=-1;manifest=[]
for j,end in enumerate(ends,1):
 stop=end['sim_time'];part=[x for x in events if start<x.get('sim_time',-1)<=stop];ready_events=[x['sim_time'] for x in part if x['event']=='near_handoff_ready'];ready=ready_events[0] if ready_events else min(x['sim_time'] for x in part if 'sim_time' in x);times=[ready,min(stop,ready+5),max(ready,stop-.05)];entry=dict(trial=j,near_handoff_occurred=bool(ready_events),frames=[])
 for k,stamp in enumerate(times):
  panels=[]
  for camera,width in [('overhead',960),('side',960),('center',640),('left',640),('right',640)]:
   options=[r for r in rows if r['camera']==camera and start<int(r['sim_ns'])/1e9<=stop]
   row=min(options,key=lambda x:abs(int(x['sim_ns'])/1e9-stamp));image=cv2.imread(str(roll/'frames'/row['file']));scale=min(width/image.shape[1],540/image.shape[0]);image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)));panel=np.zeros((540,width,3),np.uint8);y=(540-image.shape[0])//2;x=(width-image.shape[1])//2;panel[y:y+image.shape[0],x:x+image.shape[1]]=image;cv2.putText(panel,f'{camera} recorded {int(row["sim_ns"])/1e9:.3f}s',(8,25),0,.6,(0,255,0),1);panels.append(panel)
  path=output/f'trial_{j}_stage_{k}.jpg';cv2.imwrite(str(path),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]),[cv2.IMWRITE_JPEG_QUALITY,94]);entry['frames'].append(dict(requested_sim_s=stamp,path=str(path)))
 manifest.append(entry);start=stop
(output/'manifest.json').write_text(json.dumps(dict(recorded_images_only=True,frames=manifest),indent=2)+'\n');print(output)
