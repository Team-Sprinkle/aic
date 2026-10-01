#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());out={}
for key,rows in r['edges'].items():
 parent,child=key.split('|');needed=(child in {'tabletop','base_link','task_board','cable_1','cable_1/sc_tip_link'} or 'sc_port' in key or any(k in key for k in ['camera/','cam_mount','ati/','gripper/','tool0']))
 if not needed:continue
 selected=[]
 for ts,d in rows:
  if selected and (ts-selected[-1][0]<50000000 or d==selected[-1][1]):continue
  selected.append([ts,d])
 out[key]=selected
r['edges']=out;Path(sys.argv[2]).write_text(json.dumps(r)+'\n');print('TF edges retained',len(out))
