"""
Convert a dataset of PNG frame sequences to MP4 video files.

Input structure:  <input_root>/<clip_name>/<frame>.png
Output structure: <output_root>/<clip_name>.mp4

Usage:
    python scripts/frames_to_videos.py \\
        --input  datasets/YouHQ40/SR_flash_bs_none \\
        --output datasets/YouHQ40/SR_flash_bs_none_videos
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _collect_clips(root):
    clips = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in os.listdir(full)
        ):
            clips.append((entry, full))
    return clips


def _frames_to_video(frames_dir, out_path, fps):
    pngs = sorted(Path(frames_dir).glob("*.png"))
    if not pngs:
        return False
    digits = len(pngs[0].stem)
    pattern = str(Path(frames_dir) / f"%0{digits}d.png")
    start = int(pngs[0].stem)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(start),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description="Convert frame sequences to MP4 files")
    p.add_argument("--input", required=True, help="Root dir with clip subdirectories (PNG frames)")
    p.add_argument("--output", required=True, help="Output dir for MP4 files")
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()

    clips = _collect_clips(args.input)
    if not clips:
        print(f"No clips found in {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"Found {len(clips)} clips, converting to {args.output}")

    failed = []
    for i, (clip_name, clip_dir) in enumerate(clips):
        out_path = os.path.join(args.output, f"{clip_name}.mp4")
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(clips)}] {clip_name} — skip (exists)")
            continue
        ok = _frames_to_video(clip_dir, out_path, fps=args.fps)
        status = "OK" if ok else "FAILED"
        print(f"  [{i+1}/{len(clips)}] {clip_name} {status}")
        if not ok:
            failed.append(clip_name)

    if failed:
        print(f"\nFailed: {failed}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
