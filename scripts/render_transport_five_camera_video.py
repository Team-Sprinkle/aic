#!/usr/bin/env python3
"""Synchronized observation-only five-camera rollout video for diagnosis."""
import argparse
import bisect
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


CAMERAS=('overhead','side','center','left','right')


def main():
    # Avoid oversubscribing large servers for repeated camera resize calls.
    cv2.setNumThreads(2)
    p=argparse.ArgumentParser()
    p.add_argument('rollout',type=Path)
    p.add_argument('--fps',type=int,default=10)
    p.add_argument('--active-only',action='store_true',
                   help='Stop at the final actor observation, before engine scene teardown')
    p.add_argument('--grid',action='store_true',
                   help='1080p two overview / three wrist layout with preserved image aspect')
    a=p.parse_args()
    rows={name:[] for name in CAMERAS}
    for row in csv.DictReader((a.rollout/'frame_manifest.csv').open()):
        if row['camera'] in rows:
            rows[row['camera']].append((int(row['sim_ns']),a.rollout/'frames'/row['file']))
    for data in rows.values():data.sort()
    start=max(data[0][0] for data in rows.values())
    stop=min(data[-1][0] for data in rows.values())
    if a.active_only:
        trace=a.rollout/'runtime_trace.jsonl'
        events=[json.loads(line) for line in trace.open()]
        terminal=[row.get('sim_time',row.get('time_sec')) for row in events
                  if row.get('event') in ('terminal_before_return','wide_teacher_terminal_before_return')
                  and ('sim_time' in row or 'time_sec' in row)]
        if not terminal:raise ValueError(f'No terminal actor event in {trace}')
        # Earlier hold-comparison logs retained the pre-inspection proposal
        # timestamp. The accepted observation is causal and occurs before
        # reset; include the complete hold rather than clipping it away.
        accepted=[row['sim_time'] for row in events
                  if row.get('event')=='matched_handoff_accepted']
        end=max([*terminal,*accepted])
        if any(row.get('event')=='locked_local_switch' for row in events):
            # The first terminal marks the transport component, not the end
            # of insertion. The scored bag's camera window is reset-scoped.
            stream=a.rollout/'scored_stream.json'
            if not stream.exists():
                raise ValueError('Extract the scored stream before rendering a multi-stage rollout')
            cameras=json.loads(stream.read_text())['cameras']
            end=min(max(row[0] for row in cameras[c])/1e9 for c in ('center','left','right'))
        coverage=a.rollout/'coverage_metrics.json'
        if coverage.exists():
            # The coverage scorer resolves nested stage terminals and the
            # force-stop pose timestamp without joining scene teardown.
            end=float(json.loads(coverage.read_text())['terminal']['time_s'])
        stop=min(stop,int(end*1e9))
    out=a.rollout/f'five_cameras_{a.fps}fps{"_grid" if a.grid else ""}{"_active" if a.active_only else ""}.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
             '-f','rawvideo','-pixel_format','bgr24','-video_size',('1920x1080' if a.grid else '1920x216'),
             '-framerate',str(a.fps),'-i','pipe:0','-an','-c:v','libx264',
             '-preset','veryfast','-crf','23','-pix_fmt','yuv420p',
             '-threads','2','-movflags','+faststart',str(out)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    stamps={camera:[entry[0] for entry in data] for camera,data in rows.items()}
    try:
        for time_ns in range(start,stop,int(1e9/a.fps)):
            panels=[]
            for camera in CAMERAS:
                data=rows[camera]
                j=bisect.bisect_left(stamps[camera],time_ns)
                j=min(range(max(0,j-1),min(len(data),j+1)),key=lambda k:abs(data[k][0]-time_ns))
                image=cv2.imread(str(data[j][1]))
                if a.grid:
                    width=960 if camera in ('overhead','side') else 640
                    scale=min(width/image.shape[1],540/image.shape[0])
                    resized=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
                    image=np.zeros((540,width,3),np.uint8)
                    left=(width-resized.shape[1])//2;top=(540-resized.shape[0])//2
                    image[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
                else:
                    image=cv2.resize(image,(384,216))
                cv2.putText(image,f'{camera} {time_ns/1e9:.1f}s',(6,20),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                panels.append(image)
            canvas=(np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
                    if a.grid else np.hstack(panels))
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    if process.wait():raise RuntimeError(f'ffmpeg failed: {out}')
    print(out)


if __name__=='__main__':main()
