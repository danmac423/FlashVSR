"""
Merge DOVER results with quality metrics from benchmarks/quality/runner.py.

Usage:
    python scripts/merge_dover_metrics.py \\
        --metrics benchmarks/quality/results/VideoLQ/SR_flash_bs_none_20260614_201957.csv \\
        --dover   benchmarks/quality/results/VideoLQ/SR_flash_bs_none_dover.csv \\
        --output  benchmarks/quality/results/VideoLQ/SR_flash_bs_none_merged.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def load_metrics(path):
    """Load runner.py CSV. Returns {clip: row_dict}, skips AVERAGE row."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"].strip()
            if clip == "AVERAGE":
                continue
            rows[clip] = {k.strip(): v.strip() for k, v in row.items()}
    return rows


def load_dover(path):
    """Load evaluate_a_set_of_videos.py CSV. Returns {clip: row_dict}."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # path column: ".../001.mp4" → "001"
            raw_path = list(row.values())[0].strip()
            clip = Path(raw_path).stem
            rows[clip] = {k.strip(): v.strip() for k, v in row.items()}
    return rows


def main():
    p = argparse.ArgumentParser(description="Merge DOVER results with quality metrics")
    p.add_argument("--metrics", required=True, help="CSV from benchmarks/quality/runner.py")
    p.add_argument("--dover", required=True, help="CSV from evaluate_a_set_of_videos.py")
    p.add_argument("--output", required=True, help="Output merged CSV path")
    args = p.parse_args()

    metrics = load_metrics(args.metrics)
    dover = load_dover(args.dover)

    clips_only_in_metrics = set(metrics) - set(dover)
    clips_only_in_dover = set(dover) - set(metrics)
    if clips_only_in_metrics:
        print(
            f"Warning: {len(clips_only_in_metrics)} clips in metrics but not in DOVER: {sorted(clips_only_in_metrics)}",
            file=sys.stderr,
        )
    if clips_only_in_dover:
        print(
            f"Warning: {len(clips_only_in_dover)} clips in DOVER but not in metrics: {sorted(clips_only_in_dover)}",
            file=sys.stderr,
        )

    common_clips = sorted(set(metrics) & set(dover))
    if not common_clips:
        print("No common clips found — check if clip names match.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "clip",
        "num_frames",
        "psnr",
        "ssim",
        "lpips",
        "niqe",
        "musiq",
        "clipiqa",
        "dover",
    ]

    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def fmt(val):
        return f"{val:.6f}" if val is not None else ""

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        totals = {k: [] for k in fieldnames if k not in ("clip", "num_frames")}
        total_frames = 0

        for clip in common_clips:
            m = metrics[clip]
            d = dover[clip]

            technical = safe_float(d.get("technical score"))

            row = {
                "clip": clip,
                "num_frames": m.get("num_frames", ""),
                "psnr": m.get("psnr", ""),
                "ssim": m.get("ssim", ""),
                "lpips": m.get("lpips", ""),
                "niqe": m.get("niqe", ""),
                "musiq": m.get("musiq", ""),
                "clipiqa": m.get("clipiqa", ""),
                "dover": fmt(technical),
            }
            writer.writerow(row)

            total_frames += int(m.get("num_frames") or 0)
            for key in ("psnr", "ssim", "lpips", "niqe", "musiq", "clipiqa"):
                v = safe_float(m.get(key))
                if v is not None:
                    totals[key].append(v)

            if technical is not None:
                totals["dover"].append(technical)

        avg_row = {"clip": "AVERAGE", "num_frames": total_frames}
        for key in fieldnames:
            if key in ("clip", "num_frames"):
                continue
            vals = totals[key]
            avg_row[key] = fmt(sum(vals) / len(vals)) if vals else ""
        writer.writerow(avg_row)

    print(f"Merged {len(common_clips)} clips → {args.output}")


if __name__ == "__main__":
    main()
