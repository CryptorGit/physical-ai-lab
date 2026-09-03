"""Concatenate deterministic showcase scene clips without changing their timing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("clips", nargs="+")
args = parser.parse_args()

output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
clips = [Path(value).resolve(strict=True) for value in args.clips]
captures = [cv2.VideoCapture(str(path)) for path in clips]
first = captures[0]
fps = first.get(cv2.CAP_PROP_FPS)
width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
temporary = output.with_name(output.stem + "_assemble_tmp.mp4")
writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
frame_count = 0
for clip_index, (path, capture) in enumerate(zip(clips, captures)):
    if clip_index > 0:
        # An explicit one-second reset card prevents the independent running
        # and standing families from looking like one continuous transition.
        card = np.full((height, width, 3), (20, 24, 30), dtype=np.uint8)
        title = "SCENE CUT / RESET"
        next_scene = path.stem.upper().replace("_", " ")
        cv2.putText(
            card, title, (max(30, width // 2 - 250), height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 248, 252), 3, cv2.LINE_AA,
        )
        cv2.putText(
            card, next_scene, (max(30, width // 2 - 300), height // 2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 205, 255), 2, cv2.LINE_AA,
        )
        for _ in range(round(fps)):
            writer.write(card)
            frame_count += 1
    clip_fps = capture.get(cv2.CAP_PROP_FPS)
    clip_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if abs(clip_fps - fps) > 0.01 or clip_size != (width, height):
        raise RuntimeError(
            f"Incompatible clip {path}: fps={clip_fps}, size={clip_size}; "
            f"expected fps={fps}, size={(width, height)}"
        )
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
    capture.release()
writer.release()
if output.exists():
    output.unlink()
temporary.replace(output)
print(
    f"reel_result=PASS output={output} clips={len(clips)} frames={frame_count} "
    f"fps={fps} scene_cut_pause_s=1.0"
)
