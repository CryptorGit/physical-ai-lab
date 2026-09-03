"""Build a candid LinkedIn research-retrospective video for OpenDuckMini exp_003.

The source clips are simulation recordings.  The generated video deliberately
labels v52/v59 status so that diagnostic locomotion is not presented as a
hardware-qualified controller.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = WORKSPACE / "experiments/mujoco/exp_003_openduckmini_calibrated_walk"
V59 = EXP / "artifacts/videos_v59_omnidirectional_diagnostic_not_qualified"
V52 = HERE / "source_clips/basic"
BUILD = HERE / "build"
OUTPUT = HERE / "openduck_exp003_linkedin_research_retrospective.mp4"
COVER = HERE / "openduck_exp003_linkedin_cover.png"
MANIFEST = HERE / "video_manifest.json"

WIDTH = 1080
HEIGHT = 1080
FPS = 30
FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    selected_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=selected_font)
    x = (WIDTH - (box[2] - box[0])) // 2
    draw.text((x, y), text, font=selected_font, fill=fill)


def make_card(
    path: Path,
    eyebrow: str,
    title_lines: list[str],
    body_lines: list[str],
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (8, 17, 31))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 72, 1010, 1008), radius=38, fill=(15, 30, 51))
    draw.rectangle((70, 72, 88, 1008), fill=(255, 183, 52))
    draw.text((125, 130), eyebrow.upper(), font=font(FONT_BOLD, 27), fill=(255, 190, 67))
    y = 245
    for line in title_lines:
        draw.text((125, y), line, font=font(FONT_BOLD, 64), fill=(248, 250, 252))
        y += 78
    y += 45
    for line in body_lines:
        draw.text((127, y), line, font=font(FONT_REGULAR, 31), fill=(184, 204, 225))
        y += 48
    draw.text(
        (125, 928),
        "MuJoCo simulation  |  No real-hardware deployment",
        font=font(FONT_REGULAR, 24),
        fill=(133, 157, 181),
    )
    image.save(path)


def card_segment(image: Path, output: Path, seconds: float) -> None:
    run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image),
            "-t", str(seconds), "-r", str(FPS),
            "-vf", f"fade=t=in:st=0:d=0.35,fade=t=out:st={seconds - 0.35}:d=0.35",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(output),
        ]
    )


def clip_segment(
    source: Path,
    output: Path,
    start: float,
    seconds: float,
    footer: str,
) -> None:
    escaped_footer = footer.replace("'", "\\'").replace(":", "\\:")
    escaped_font = FONT_REGULAR.as_posix().replace(":", "\\:")
    fade_out = max(0.0, seconds - 0.25)
    filters = (
        "scale=1080:810:flags=lanczos,"
        "pad=1080:1080:0:135:color=0x08111F,"
        f"drawtext=fontfile='{escaped_font}':"
        f"text='{escaped_footer}':fontcolor=white:fontsize=27:"
        "x=(w-text_w)/2:y=1010,"
        "fade=t=in:st=0:d=0.25,"
        f"fade=t=out:st={fade_out}:d=0.25"
    )
    run(
        [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(seconds),
            "-i", str(source), "-vf", filters, "-r", str(FPS), "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(output),
        ]
    )


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    title = BUILD / "title.png"
    finding = BUILD / "finding.png"
    close = BUILD / "close.png"
    make_card(
        title,
        "OpenDuckMini / exp_003",
        ["Learning to walk", "in every direction"],
        ["A reinforcement-learning research retrospective", "from controller parity to a reproducible No-Go"],
    )
    make_card(
        finding,
        "What the audit found",
        ["The simulator was", "part of the experiment"],
        ["Legacy routing changed the observed failures", "GPU batch physics was not bit-exact", "A yaw reward favored overshoot"],
    )
    make_card(
        close,
        "Scientific outcome",
        ["A useful negative", "result"],
        ["v52 remains the simulation parent", "v59/v60 and later pilots were not adopted", "Single-policy omnidirectional research: CLOSED / NO-GO"],
    )
    make_card(
        COVER,
        "OpenDuckMini / reinforcement learning",
        ["What it took to", "teach a duck to walk"],
        ["exp_003: locomotion, diagnostics, and the value", "of closing a research direction honestly"],
    )

    specs = [
        ("card", title, 3.2, ""),
        ("clip", V52 / "basic_03_backward.mp4", 3.7, "v52 hybrid controller | adopted simulation parent"),
        ("clip", V59 / "basic_02_forward.mp4", 4.2, "v59 omnidirectional diagnostic | not qualified"),
        ("clip", V59 / "basic_06_yaw_left.mp4", 3.8, "yaw left | diagnostic policy"),
        ("clip", V59 / "basic_07_yaw_right.mp4", 3.8, "yaw right | diagnostic policy"),
        ("card", finding, 3.8, ""),
        ("clip", V59 / "compound_03_forward_yaw_left.mp4", 4.2, "forward + yaw left | diagnostic policy"),
        ("clip", V59 / "compound_04_forward_yaw_right.mp4", 4.2, "forward + yaw right | diagnostic policy"),
        ("card", close, 4.5, ""),
    ]

    segment_paths: list[Path] = []
    sources: list[dict[str, object]] = []
    for index, (kind, source, seconds, footer) in enumerate(specs):
        output = BUILD / f"segment_{index:02d}.mp4"
        if kind == "card":
            card_segment(source, output, seconds)
            start = 0.0
        else:
            start = 0.35
            clip_segment(source, output, start, seconds, footer)
        segment_paths.append(output)
        sources.append(
            {
                "kind": kind,
                "path": str(source.relative_to(WORKSPACE)),
                "sha256": sha256(source),
                "start_seconds": start,
                "duration_seconds": seconds,
                "footer": footer,
            }
        )

    concat_file = BUILD / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    silent = BUILD / "video_only.mp4"
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(silent),
        ]
    )
    run(
        [
            "ffmpeg", "-y", "-i", str(silent),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(OUTPUT),
        ]
    )

    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=index,codec_name,width,height,r_frame_rate,pix_fmt",
            "-of", "json", str(OUTPUT),
        ],
        text=True,
    )
    payload = {
        "title": "OpenDuckMini exp_003 LinkedIn research retrospective",
        "output": str(OUTPUT.relative_to(WORKSPACE)),
        "output_sha256": sha256(OUTPUT),
        "cover": str(COVER.relative_to(WORKSPACE)),
        "cover_sha256": sha256(COVER),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
        ).strip(),
        "disclosure": {
            "simulation_only": True,
            "real_hardware_allowed": False,
            "v52_status": "adopted simulation parent",
            "v59_status": "diagnostic_not_qualified",
        },
        "sources": sources,
        "ffprobe": json.loads(probe),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(MANIFEST)


if __name__ == "__main__":
    main()
