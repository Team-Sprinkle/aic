#!/usr/bin/env python3
"""20Hz close views with projected oracle contact markers; diagnostic only."""
import bisect,csv,json,sys
from pathlib import Path
import cv2,numpy as np,yaml
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'artifacts/prod_cheatcode_audit'))
from render_smooth_wide import open_encoder
p=Path(sys.argv[1]); out=p/'oracle_video';out.mkdir(exist_ok=True)
meta=json.loads((p/'oracle_cameras.json').read_text());rows={k:[] for k in meta}
bag=next((p/'results').glob('bag_trial*'))
bag_meta=yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
wall_start=bag_meta['starting_time']['nanoseconds_since_epoch'];wall_end=wall_start+bag_meta['duration']['nanoseconds']
for r in csv.DictReader((p/'frame_manifest.csv').open()):
 if r['camera'] in rows and wall_start<=int(r['wall_ns'])<=wall_end:rows[r['camera']].append((int(r['sim_ns']),p/'frames'/r['file']))
for v in rows.values():v.sort()
events=[json.loads(l) for l in (p/'corrected_contact_timeline.jsonl').open()];stamps=[r['sim_ns'] for r in events]
start=max(v[0][0] for v in rows.values());end=min(v[-1][0] for v in rows.values());cs={k:[r[0] for r in v] for k,v in rows.items()};cache={}
proc=open_encoder(out/'port_contacts_20fps.mp4',(1280,440));snap=[]
for t in range(start,end,50000000):
 idx=min(bisect.bisect_left(stamps,t),len(stamps)-1);e=events[idx];panels=[]
 for k,v in rows.items():
  j=min(bisect.bisect_left(cs[k],t),len(v)-1);candidates=range(max(0,j-1),j+1);j=min(candidates,key=lambda x:abs(v[x][0]-t));path=v[j][1]
  if cache.get(k,(None,None))[0]!=path:cache[k]=(path,cv2.imread(str(path)))
  im=cache[k][1].copy();m=meta[k];eye=np.array(m['pose_world_xyz_rpy'][:3]);rot=Rotation.from_euler('xyz',m['pose_world_xyz_rpy'][3:]).as_matrix();f=m['width']/(2*np.tan(m['horizontal_fov']/2))
  for c in e['contacts']:
   if 'sc_plug_link' not in c['pair']:continue
   for pos in c['positions']:
    q=rot.T@(np.array([pos[x] for x in ['x','y','z']])-eye)
    if q[0]<=0:continue
    x=int(m['width']/2-f*q[1]/q[0]);y=int(m['height']/2-f*q[2]/q[0]);cv2.circle(im,(x,y),8,(0,0,255),2);cv2.putText(im,f"{c['max_point_normal_force_n']:.2f}N",(x+8,y),0,.6,(0,0,255),2)
  im=cv2.resize(im,(640,360));panel=np.zeros((440,640,3),np.uint8);panel[:360]=im
  cv2.putText(panel,f'{k} sim {t/1e9:.2f}s', (8,385),0,.6,(0,255,0),1)
  cv2.putText(panel,'Red: SC plug contacts; normal force only',(8,410),0,.5,(255,255,255),1)
  panels.append(panel)
 frame=cv2.hconcat(panels);proc.stdin.write(frame.tobytes())
 for s in [42,44,45.6,48,50]:
  if abs(t/1e9-s)<.025:cv2.imwrite(str(out/f'sim_{s:.1f}s.jpg'),frame)
cv2.imwrite(str(out/'terminal_before_reset.jpg'),frame)
proc.stdin.close();assert proc.wait()==0
print(out/'port_contacts_20fps.mp4')
