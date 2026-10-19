#!/usr/bin/env python3
"""Publish selected scored recordings and motion-trimmed GIF previews."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


TRIALS = (
    ('sfp_one_card', 'eval_trial_1_sfp_1card_five_cameras_20fps.mp4'),
    ('sfp_two_cards', 'eval_trial_2_sfp_2cards_five_cameras_20fps.mp4'),
    ('sc_three_cards', 'eval_trial_3_sc_3cards_five_cameras_20fps.mp4'),
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Saved rollout directory')
    parser.add_argument('--output', type=Path, default=Path('docs/media/final_evaluation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = [json.loads(line) for line in (args.source / 'runtime_trace.jsonl').open()]
    terminals = [row for row in trace if row.get('event') == 'terminal_before_return'
                 and row.get('episode_index') is not None]
    assert [row['episode_index'] for row in terminals] == [1, 2, 3]
    scores = json.loads((args.source.parent / 'score_summary.json').read_text())['score']
    assert all(scores[f'trial_{i}']['tier_3']['score'] == 75 for i in (1, 2, 3))
    rows = []
    start = 0.002  # Common first camera stamp in the selected frozen recording.
    for index, ((name, filename), terminal) in enumerate(zip(TRIALS, terminals), 1):
        end = float(terminal['sim_time'])
        commands = [r for r in trace if r.get('event') == 'command' and 'state' in r
                    and start <= r['sim_time'] < end]
        origin = commands[0]['state'][:3]
        moving = []
        previous_stamp = None
        for row in commands:
            stamp = row['sim_time']
            if stamp == previous_stamp:
                continue
            previous_stamp = stamp
            state = row['state']
            speed = sum(v*v for v in state[7:10]) ** .5
            distance = sum((a-b)**2 for a, b in zip(state[:3], origin)) ** .5
            if speed > .003 and distance > .001:
                moving.append(stamp)
                if len(moving) >= 3 and moving[-1] - moving[0] <= .4:
                    break
            else:
                moving = []
        assert len(moving) >= 3, f'No sustained measured movement in trial {index}'
        trim_start = max(0., moving[0] - start - .2)
        video = args.output / f'{name}.mp4'
        gif = args.output / f'{name}.gif'
        shutil.copy2(args.source / filename, video)
        video.chmod(0o644)
        assert digest(video) == digest(args.source / filename)
        # Preserve the complete scored movement. Preview plays at 2x speed;
        # only initialization and the final post-terminal scene are excluded.
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-ss', str(trim_start), '-i', str(video), '-t', str((end-start-trim_start)/2),
            '-filter_complex',
            '[0:v]setpts=(PTS-STARTPTS)/2,fps=8,scale=720:-1:flags=lanczos,split[a][b];'
            '[a]palettegen=max_colors=96:stats_mode=diff[p];'
            '[b][p]paletteuse=dither=bayer:bayer_scale=3',
            '-threads', '2', '-filter_complex_threads', '2', '-loop', '0', str(gif)
        ], check=True)
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', str(gif)]))
        duration = float(probe['format']['duration'])
        expected_duration = (end-start-trim_start)/2
        assert abs(duration-expected_duration) < .3, 'GIF timing differs from declared 2x preview'
        rows.append(dict(trial=index, task=name, video=video.name, gif=gif.name,
                         score_100=sum(scores[f'trial_{index}'][k]['score']
                                       for k in ('tier_1', 'tier_2', 'tier_3')),
                         video_sha256=digest(video), gif_sha256=digest(gif),
                         video_bytes=video.stat().st_size, gif_bytes=gif.stat().st_size,
                         original_capture_fps=20, gif_fps=8, gif_playback_speed=2,
                         gif_duration_s=duration,
                         trial_start_sim_s=start, terminal_sim_s=end,
                         first_sustained_motion_sim_s=moving[0],
                         gif_trim_start_video_s=trim_start))
        start = end
    manifest = dict(schema='final_project_media/v1', source_rollout=str(args.source),
                    selected_repeat=10, total_score_300=scores['total'],
                    configuration='published qualification eval_config.yaml',
                    configuration_sha256=digest(args.source / 'eval_config.yaml'),
                    scope='Selected successful exposed-config demonstration; not unseen evaluation',
                    repeat_batch=dict(scored_runs=19, all_three_insertions=15,
                                      individual_insertions=53, individual_trials=57),
                    runtime='Rule transport, RGB perception, explicit insertion controller; no BC/RL actions',
                    motion_trim='Three distinct samples: speed >3mm/s and displacement >1mm; 0.2s preroll',
                    trials=rows)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
