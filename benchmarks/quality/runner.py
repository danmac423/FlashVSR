"""
Evaluate FlashVSR output quality using 6 metrics from the paper.

Reference-based (FR): PSNR ↑, SSIM ↑, LPIPS ↓    — require --gt
No-reference (NR):    NIQE ↓, MUSIQ ↑, CLIPIQA ↑  (per-frame, SR only)

Note: DOVER requires a separate environment due to torch<2 constraint.

Input structure:
    <sr_root>/<clip_name>/<frame>.png
    <gt_root>/<clip_name>/<frame>.png

Output: <output_dir>/<sr_name>.{csv,json}

Usage:
    uv run python -m benchmarks.quality.runner \\
        --sr  datasets/YouHQ40/SR_flash_bs_none \\
        --gt  datasets/YouHQ40/HQ \\
        --output benchmarks/quality/results
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import pyiqa
except ImportError:
    print("pyiqa not found. Install with:  pip install pyiqa", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import get_logger, setup_logger

logger = get_logger()

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

FR_METRICS = ["psnr", "ssim", "lpips"]
NR_METRICS = ["niqe", "musiq", "clipiqa"]
ALL_METRICS = FR_METRICS + NR_METRICS


@dataclass
class ClipMetrics:
    clip: str
    num_frames: int
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    niqe: float | None = None
    musiq: float | None = None
    clipiqa: float | None = None


def _sorted_frame_paths(clip_dir: str) -> list[str]:
    return sorted(
        os.path.join(clip_dir, f)
        for f in os.listdir(clip_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def _load_frame(path: str, device: torch.device) -> torch.Tensor:
    """Return (1, C, H, W) float32 [0, 1]."""
    arr = np.array(Image.open(path).convert("RGB"))
    t = torch.from_numpy(arr).to(torch.float32) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _collect_clips(root: str) -> list[tuple[str, str]]:
    clips = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in os.listdir(full)
        ):
            clips.append((entry, full))
    return clips


def eval_clip(
    sr_paths: list[str],
    gt_paths: list[str] | None,
    metric_fns: dict[str, Any],
    device: torch.device,
) -> dict[str, float | None]:
    scores: dict[str, list[float]] = {m: [] for m in ALL_METRICS}

    pairs = list(zip(sr_paths, gt_paths)) if gt_paths else [(p, None) for p in sr_paths]
    for sr_path, gt_path in tqdm(pairs, desc="  frames", leave=False):
        sr = _load_frame(sr_path, device)

        if gt_path is not None:
            gt = _load_frame(gt_path, device)
            for m in FR_METRICS:
                if m in metric_fns:
                    scores[m].append(metric_fns[m](sr, gt).item())

        for m in NR_METRICS:
            scores[m].append(metric_fns[m](sr).item())

    return {m: float(np.mean(v)) if v else None for m, v in scores.items()}


def _dataset_averages(results: list[ClipMetrics]) -> dict[str, float | None]:
    avgs: dict[str, float | None] = {}
    for m in ALL_METRICS:
        vals = [getattr(r, m) for r in results if getattr(r, m) is not None]
        avgs[m] = float(np.mean(vals)) if vals else None
    return avgs


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def save_results(results: list[ClipMetrics], output_dir: str, sr_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    avgs = _dataset_averages(results)
    stem = f"{sr_name}"

    csv_path = os.path.join(output_dir, f"{stem}.csv")
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
        writer.writerow(
            {"clip": "AVERAGE", "num_frames": sum(r.num_frames for r in results), **avgs}
        )

    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump({"clips": [asdict(r) for r in results], "average": avgs}, f, indent=2)

    logger.info("Saved → %s", output_dir)
    logger.info(
        "Average:  PSNR=%s  SSIM=%s  LPIPS=%s  NIQE=%s  MUSIQ=%s  CLIPIQA=%s",
        _fmt(avgs.get("psnr"), ".2f"),
        _fmt(avgs.get("ssim")),
        _fmt(avgs.get("lpips")),
        _fmt(avgs.get("niqe")),
        _fmt(avgs.get("musiq"), ".2f"),
        _fmt(avgs.get("clipiqa")),
    )


def main() -> None:
    setup_logger(logging.INFO)

    p = argparse.ArgumentParser(description="Evaluate FlashVSR quality metrics")
    p.add_argument("--sr", required=True, help="Root dir with SR clip subdirectories")
    p.add_argument(
        "--gt",
        default=None,
        help="Root dir with GT (HQ) clip subdirectories (required for PSNR/SSIM/LPIPS)",
    )
    p.add_argument("--output", default="benchmarks/quality/results")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    has_gt = args.gt is not None

    logger.info("Loading metric models...")
    metric_fns: dict[str, Any] = {}
    if has_gt:
        for m in FR_METRICS:
            kwargs = {"test_y_channel": True} if m == "psnr" else {}
            metric_fns[m] = pyiqa.create_metric(m, device=device, **kwargs)
    for m in NR_METRICS:
        metric_fns[m] = pyiqa.create_metric(m, device=device)
    logger.info("Loaded: %s", list(metric_fns.keys()))

    sr_clips = _collect_clips(args.sr)
    if not sr_clips:
        logger.error("No clips found in %s", args.sr)
        sys.exit(1)
    logger.info("Found %d clips in %s", len(sr_clips), args.sr)

    gt_clip_dirs = dict(_collect_clips(args.gt)) if has_gt else {}

    results: list[ClipMetrics] = []
    for i, (clip_name, sr_dir) in enumerate(sr_clips):
        logger.info("[%d/%d] %s", i + 1, len(sr_clips), clip_name)
        try:
            sr_paths = _sorted_frame_paths(sr_dir)
            gt_dir = gt_clip_dirs.get(clip_name)
            gt_paths = _sorted_frame_paths(gt_dir) if gt_dir else None

            if gt_paths is not None and len(sr_paths) != len(gt_paths):
                n = min(len(sr_paths), len(gt_paths))
                logger.warning(
                    "  Frame count mismatch SR=%d GT=%d — trimming to %d",
                    len(sr_paths),
                    len(gt_paths),
                    n,
                )
                sr_paths, gt_paths = sr_paths[:n], gt_paths[:n]

            scores = eval_clip(sr_paths, gt_paths, metric_fns, device)
            results.append(ClipMetrics(clip=clip_name, num_frames=len(sr_paths), **scores))

        except Exception as exc:
            logger.error("Failed on %s: %s", clip_name, exc, exc_info=True)

    if results:
        sr_name = os.path.basename(args.sr.rstrip("/"))
        save_results(results, args.output, sr_name)
    else:
        logger.error("No results collected.")


if __name__ == "__main__":
    main()
