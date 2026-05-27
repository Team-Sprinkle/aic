#!/usr/bin/env python3
"""Create a LeRobot dataset copy with videos transcoded to H.264."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder", default="h264_nvenc")
    parser.add_argument("--fallback-encoder", default="libx264")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--qp", type=int, default=23)
    parser.add_argument("--gop", type=int, default=2)
    return parser.parse_args()


def copy_non_video_tree(src: Path, dst: Path) -> None:
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "videos":
            target.mkdir(parents=True, exist_ok=True)
            continue
        if child.is_dir():
            shutil.copytree(child, target, copy_function=shutil.copy2)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def ffmpeg_command(src: Path, dst: Path, args: argparse.Namespace, encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        codec_args = ["-c:v", encoder, "-preset", "fast", "-rc", "constqp", "-qp", str(args.qp)]
    else:
        codec_args = ["-c:v", encoder, "-preset", "ultrafast", "-crf", str(args.crf)]
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-an",
        *codec_args,
        "-g",
        str(args.gop),
        "-keyint_min",
        str(args.gop),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]


def transcode_one(src: Path, dst: Path, args: argparse.Namespace) -> tuple[Path, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp.mp4")
    for encoder in (args.encoder, args.fallback_encoder):
        cmd = ffmpeg_command(src, tmp, args, encoder)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return dst, encoder
        tmp.unlink(missing_ok=True)
        if encoder == args.fallback_encoder:
            raise RuntimeError(f"ffmpeg failed for {src}:\n{proc.stderr}")
    raise AssertionError("unreachable")


def update_metadata(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for feature in info.get("features", {}).values():
        if feature.get("dtype") == "video":
            video_info = feature.setdefault("info", {})
            video_info["video.codec"] = "h264"
            video_info["video.pix_fmt"] = "yuv420p"
    info["aic_video_transcode"] = {
        "schema_version": "aic_video_transcode/v1",
        "codec": "h264",
        "reason": "faster random-access decoding for ACT training",
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    copy_non_video_tree(args.input_root, args.output_root)

    video_files = sorted((args.input_root / "videos").rglob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No mp4 files under {args.input_root / 'videos'}")
    encoders: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for src in video_files:
            rel = src.relative_to(args.input_root)
            futures.append(pool.submit(transcode_one, src, args.output_root / rel, args))
        for index, future in enumerate(as_completed(futures), start=1):
            dst, encoder = future.result()
            encoders[encoder] = encoders.get(encoder, 0) + 1
            if index % 10 == 0 or index == len(futures):
                print(f"transcoded {index}/{len(futures)} latest={dst} encoders={encoders}", flush=True)
    update_metadata(args.output_root)
    report = {"input_root": str(args.input_root), "output_root": str(args.output_root), "videos": len(video_files), "encoders": encoders}
    report_path = args.output_root / "meta" / "video_transcode_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
