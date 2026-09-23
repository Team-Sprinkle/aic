#!/usr/bin/env python3
"""Encode one-Hz audit frames as widely playable H.264 MP4 video."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def transcode(source: Path, destination: Path) -> None:
    """Keep each source frame on screen for one second at 10 playback fps."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".h264.tmp.mp4")
    if temporary == source:
        raise ValueError("Temporary output must differ from the input")
    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source), "-vf", "fps=10,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-profile:v", "baseline", "-level:v", "3.1", "-threads", "2",
            "-movflags", "+faststart", "-an", str(temporary),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i",
            str(temporary), "-f", "null", "-",
        ], check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = Path(name)
        transcode(path, path)
        print(path)
