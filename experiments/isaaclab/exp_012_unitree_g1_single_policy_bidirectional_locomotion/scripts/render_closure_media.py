"""Transcode the card-free demo to H.264 and create the exp_012 thumbnail."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent.parents[2]
RAW = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/closure/raw/exp_012_closure_sequence_raw_v6_floor.mp4"
TEMP = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/closure/raw/exp_012_linkedin_assembly_mp4v.mp4"
FINAL = REPO / "media/exp_012_g1_single_policy_sequence_linkedin.mp4"
THUMB = REPO / "media/exp_012_g1_single_policy_sequence_thumbnail.png"
WIDTH, HEIGHT, FPS = 1920, 1080, 30


def centered(frame, text, y, scale, color=(245, 248, 252), thickness=2):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    cv2.putText(
        frame, text, ((WIDTH - size[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def refresh_static_header(frame):
    cv2.rectangle(frame, (20, 18), (835, 88), (9, 14, 24), -1)
    cv2.rectangle(frame, (1210, 18), (1895, 120), (9, 14, 24), -1)
    cv2.putText(frame, "EXP_012 - ONE POLICY, TWO GAITS", (42, 62),
                cv2.FONT_HERSHEY_SIMPLEX, .88, (245, 248, 252), 2, cv2.LINE_AA)
    for index, line in enumerate(("ONE CHECKPOINT", "NO ROUTER", "NO ACTION BLENDING")):
        cv2.putText(frame, line, (1235, 52 + 28 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 248, 252), 2, cv2.LINE_AA)
    return frame


def thumbnail(frame):
    canvas = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 178), (8, 13, 23), -1)
    cv2.addWeighted(overlay, .84, canvas, .16, 0, canvas)
    cv2.putText(canvas, "ONE POLICY, TWO GAITS", (45, 86), cv2.FONT_HERSHEY_SIMPLEX,
                1.75, (245, 248, 252), 4, cv2.LINE_AA)
    cv2.putText(canvas, "Stand -> Walk -> Run -> Walk -> Stop", (48, 143),
                cv2.FONT_HERSHEY_SIMPLEX, .82, (70, 205, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(THUMB), canvas)


def main():
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(RAW))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != WIDTH or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != HEIGHT:
        raise RuntimeError("raw capture resolution mismatch")
    writer = cv2.VideoWriter(str(TEMP), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("unable to open assembly writer")
    selected_thumb = None
    for output_index in range(28 * FPS):
        source_index = min(int(round(output_index * source_fps / FPS)), source_frames - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"raw frame {source_index} unreadable")
        frame = refresh_static_header(frame)
        writer.write(frame)
        if output_index == 11 * FPS:
            selected_thumb = frame.copy()
    capture.release()
    writer.release()
    thumbnail(selected_thumb)
    command = [
        "ffmpeg", "-y", "-i", str(TEMP), "-c:v", "libx264", "-preset", "slow",
        "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
        "-an", str(FINAL),
    ]
    subprocess.run(command, check=True)
    print(FINAL)


if __name__ == "__main__":
    main()
