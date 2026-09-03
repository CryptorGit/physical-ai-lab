"""Assemble the closeout reel with ffmpeg, fixed reset cards, and 30 fps output."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("clips", nargs=3)
args = parser.parse_args()
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("ffmpeg is required")
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
clips = [Path(value).resolve(strict=True) for value in args.clips]
font = "C\\:/Windows/Fonts/arial.ttf"


def card(path: Path, seconds: int, first: str, second: str, third: str = "") -> None:
    first = first.replace(":", r"\:").replace(";", r"\;")
    second = second.replace(":", r"\:").replace(";", r"\;")
    vf = (
        f"drawtext=fontfile='{font}':text='{first}':fontcolor=white:fontsize=56:x=(w-text_w)/2:y=470,"
        f"drawtext=fontfile='{font}':text='{second}':fontcolor=0x78cdff:fontsize=34:x=(w-text_w)/2:y=560"
    )
    if third:
        third = third.replace(":", r"\:").replace(";", r"\;")
        vf += f",drawtext=fontfile='{font}':text='{third}':fontcolor=0xf0d06f:fontsize=34:x=(w-text_w)/2:y=630"
    subprocess.run([
        ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x12161c:s=1920x1080:r=30:d={seconds}",
        "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True)


intro, reset, outro = output.parent / "intro.mp4", output.parent / "reset.mp4", output.parent / "outro.mp4"
card(intro, 2, "EXP_008 CLOSEOUT SHOWCASE", "EXP_007 FORMAL CAPABILITIES REPLAYED")
card(reset, 1, "NEW SCENE / RESET", "NOT A LOCOMOTION TRANSITION")
card(
    outro,
    3,
    "ACHIEVED: STAND <-> WALK; WALK -> RUN 2.6 / 2.8",
    "NOT ACHIEVED: RUN -> WALK; FULL BIDIRECTIONAL GRAPH",
    "NEXT: UNITREE GO2",
)
inputs = [intro, clips[0], reset, clips[1], reset, clips[2], outro]
command = [ffmpeg, "-y"]
for path in inputs:
    command += ["-i", str(path)]
filters = ";".join(f"[{i}:v]fps=30,scale=1920:1080[v{i}]" for i in range(7))
filters += ";" + "".join(f"[v{i}]" for i in range(7)) + "concat=n=7:v=1:a=0[outv]"
command += ["-filter_complex", filters, "-map", "[outv]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(output)]
subprocess.run(command, check=True)
print(f"combined_video={output}")
