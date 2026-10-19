#!/usr/bin/env python3
"""Render recorded Gazebo recovery and prediction-only perception evidence."""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('docs/media/recovery_demo'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    rows = [json.loads(line) for line in (args.source/'runtime_trace.jsonl').open()]
    trigger_i = next(i for i, r in enumerate(rows) if r.get('event') == 'loaded_servo_stall_recovery')
    retreat_i = next(i for i, r in enumerate(rows) if r.get('event') == 'measured_retreat_complete')
    trigger, retreat = rows[trigger_i], rows[retreat_i]
    lateral = next(r for r in rows if r.get('event') == 'lateral_retry_complete')
    terminal = next(r for r in rows if r.get('event') == 'terminal_before_return' and r.get('episode_index') == 3)
    assert retreat['total_backoffs'] == 1 and retreat['route_resets'] == 0
    initial_state = next(r['measured_state'] for r in rows[trigger_i:]
                         if r.get('event') == 'recovery_waypoint_reference_compensated')
    final_state = next(r['measured_state'] for r in reversed(rows[trigger_i:retreat_i])
                       if 'measured_state' in r)
    retreat_mm = float(np.linalg.norm(np.array(final_state[:3])-initial_state[:3])*1000)
    lateral_mm = float(np.linalg.norm(lateral['measured_displacement_m'])*1000)
    start, stop = trigger['sim_time']-2.5, terminal['sim_time']
    frames = {c: [] for c in ('center', 'left', 'right', 'side')}
    for row in csv.DictReader((args.source/'frame_manifest.csv').open()):
        if row['camera'] in frames:
            frames[row['camera']].append((int(row['sim_ns'])/1e9, args.source/'frames'/row['file']))
    for values in frames.values():
        values.sort()
    stamps = {c: [r[0] for r in values] for c, values in frames.items()}

    def read(camera, stamp):
        values = frames[camera]
        j = bisect.bisect_left(stamps[camera], stamp)
        j = min(range(max(0, j-1), min(len(values), j+1)), key=lambda k: abs(values[k][0]-stamp))
        assert abs(values[j][0]-stamp) <= .20, 'Camera gap exceeds bounded nearest-frame hold'
        image = cv2.imread(str(values[j][1]))
        assert image is not None
        return image

    video = args.output/'gazebo_backoff_realign_insert.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1280x720',
               '-framerate', '20', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-threads', '2', '-preset', 'veryfast', '-crf', '21',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for stamp in np.arange(start, stop, .05):
            canvas = np.full((720,1280,3), (31,27,22), np.uint8)
            phase = ('INSERTION STALL' if stamp < trigger['sim_time'] else
                     'BACKTRACK ALONG MEASURED PATH' if stamp < retreat['sim_time'] else
                     'SIDEWAYS RETRY' if stamp < lateral['sim_time'] else 'REALIGN AND INSERT')
            cv2.putText(canvas, phase, (22,36), cv2.FONT_HERSHEY_SIMPLEX, .9, (100,230,255), 2)
            cv2.putText(canvas, f'Gazebo SC / 3 cards | sim {stamp:.2f}s | 1x playback',
                        (22,67), cv2.FONT_HERSHEY_SIMPLEX, .57, (235,235,235), 1)
            canvas[80:440,:640] = cv2.resize(read('side',stamp),(640,360))
            image = read('center', stamp)
            canvas[80:,640:] = image[384:1024,256:896]
            captions = [f'Measured retreat: {retreat_mm:.1f} mm',
                        f'Measured sideways retry: {lateral_mm:.2f} mm',
                        f'Stall load change: {trigger["load_change_n"]:.2f} N',
                        'Right: wrist close-up (fixed crop)',
                        'Explicit visual controller + local recovery',
                        'Recorded outcome: full insertion award']
            for i, text in enumerate(captions):
                cv2.putText(canvas,text,(18,478+i*38),cv2.FONT_HERSHEY_SIMPLEX,.58,(235,235,235),1)
            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
    assert proc.wait() == 0
    duration = float(json.loads(subprocess.check_output([
        'ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)]))['format']['duration'])
    gif = args.output/'gazebo_backoff_realign_insert.gif'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(video),
        '-t',str(duration/2),'-filter_complex',
        '[0:v]setpts=(PTS-STARTPTS)/2,fps=10,scale=800:-1:flags=lanczos,split[a][b];'
        '[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
        '-filter_complex_threads','2','-threads','2','-loop','0',str(gif)],check=True)

    # Draw ONLY saved network outputs; neither true target TF nor labels enter.
    samples = [next(r for r in reversed(rows) if r.get('event') == event
                    and r.get('report',{}).get('valid'))
               for event in ('SFP_rule_visual_inspection', 'rule_visual_inspection')]
    panel = np.full((1056,1344,3), (31,27,22),np.uint8)
    image_sources = []
    for row_i, sample in enumerate(samples):
        report = sample['report']; native = report['native_pose_component']['native']
        label = ('SFP requested card / port' if row_i == 0 else 'SC requested port')
        cv2.putText(panel, f'{label} | saved runtime predictions, sim {sample["sim_time"]:.2f}s',
                    (15,row_i*500+30),cv2.FONT_HERSHEY_SIMPLEX,.75,(245,245,245),1)
        for ci,camera in enumerate(('center','left','right')):
            source = read(camera, sample['sim_time'])
            points = np.asarray(native['keypoints'][camera],float)
            x,y = np.rint(points[0]-112).astype(int)
            patch = np.zeros((224,224,3),np.uint8)
            x0,y0 = max(x,0),max(y,0);x1,y1=min(x+224,source.shape[1]),min(y+224,source.shape[0])
            patch[y0-y:y1-y,x0-x:x1-x] = source[y0:y1,x0:x1]
            patch = cv2.resize(patch,(448,448),interpolation=cv2.INTER_NEAREST)
            uv = np.rint((points-[x,y])*2).astype(int)
            for point in uv:
                cv2.circle(patch,tuple(point),5,(255,255,0),2)
            cv2.circle(patch,tuple(uv[0]),6,(0,255,255),2)
            for i,j,color in ((1,2,(70,90,255)),(3,4,(100,255,100))):
                v=uv[j]-uv[i];norm=np.linalg.norm(v)
                if norm>0:
                    end=np.rint(uv[0]+v/norm*38).astype(int)
                    cv2.arrowedLine(patch,tuple(uv[0]),tuple(end),color,2,tipLength=.2)
            cv2.putText(patch,camera,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(245,245,245),2)
            panel[row_i*500+45:row_i*500+493,ci*448:(ci+1)*448] = patch
        image_sources.append(dict(sim_time=sample['sim_time'],event=sample['event'],
                                  requested_port=report['requested_port'],requested_card=report.get('requested_card')))
    cv2.putText(panel,'Cyan: predicted reference landmarks | yellow: opening center | arrows: 2D reference directions',
                (15,1022),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1)
    cv2.putText(panel,'Original-image crops centered on predictions. No ground-truth points are drawn.',
                (15,1048),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1)
    image_path=Path('docs/images/cable_insertion/gazebo_predicted_port_landmarks.png')
    assert cv2.imwrite(str(image_path),panel)
    manifest=dict(source_rollout='/'.join(args.source.parts[-3:]), source_trace_sha256=sha(args.source/'runtime_trace.jsonl'),
        video=video.name,gif=gif.name,video_sha256=sha(video),gif_sha256=sha(gif),
        start_sim_s=start,end_sim_s=stop,video_fps=20,gif_playback_speed=2,
        max_source_frame_gap_s={c:max(b-a for a,b in zip(stamps[c],stamps[c][1:]) if start<=a and b<=stop) for c in ('center','side')},
        local_backoffs=1,hierarchical_route_resets=0,measured_net_retreat_mm=retreat_mm,
        measured_lateral_retry_mm=lateral_mm,trigger=trigger,retreat=retreat,lateral=lateral,
        oracle_input=False,RL_actor=False,pose_overlay=str(image_path),
        pose_overlay_sha256=sha(image_path),pose_overlay_sources=image_sources,
        pose_overlay_ground_truth_points=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
