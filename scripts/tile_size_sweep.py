"""
Sweep spatial tile size and record time/VRAM for a single attention/quantization
combination, merging all results into one CSV with an explicit tile_size column.

Usage:
    uv run python scripts/tile_size_sweep.py \\
        --video inputs/example4.mp4 \\
        --tile-sizes 128 160 192 224 256 \\
        --tile-overlap 24 \\
        --measured-runs 5
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.performance.runner import Combination, benchmark
from src.config.processing import (
    ProcessingConfig,
    QuantizationMode,
    SpatialTilingConfig,
    TemporalTilingConfig,
)
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode
from src.utils.logger import get_logger, setup_logger

logger = get_logger()


def main() -> None:
    setup_logger(logging.INFO)

    p = argparse.ArgumentParser(description="Sweep spatial tile size for one attention combo")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--model", default="FlashVSR-v1.1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--tile-sizes", type=int, nargs="+", default=[128, 160, 192, 224, 256], dest="tile_sizes"
    )
    p.add_argument("--tile-overlap", type=int, default=24, dest="tile_overlap")
    p.add_argument("--warmup-runs", type=int, default=1, dest="warmup_runs")
    p.add_argument("--measured-runs", type=int, default=5, dest="measured_runs")
    p.add_argument(
        "--output-dir", default="benchmarks/performance/results/tile_size_sweep", dest="output_dir"
    )
    args = p.parse_args()

    device = torch.device(args.device)
    combo = Combination(AttentionMode.FLASH, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.NONE)

    proc_config = ProcessingConfig(
        scale=4, seed=0, sparse_ratio=2.0, kv_ratio=3.0, local_range=11, color_fix=True
    )
    temporal_config = TemporalTilingConfig(enabled=False, tile_size=100, tile_overlap=6)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"tile_sweep_{ts}.csv")
    json_path = os.path.join(args.output_dir, f"tile_sweep_{ts}.json")

    rows: list[dict] = []
    for tile_size in args.tile_sizes:
        logger.info("=== tile_size=%d ===", tile_size)
        spatial_config = SpatialTilingConfig(
            enabled=True,
            tile_size=(tile_size, tile_size),
            tile_overlap=args.tile_overlap,
        )
        try:
            results = benchmark(
                video_path=args.video,
                model=args.model,
                device=device,
                dtype=torch.bfloat16,
                combinations=[combo],
                num_warmup_runs=args.warmup_runs,
                num_measured_runs=args.measured_runs,
                proc_config=proc_config,
                spatial_config=spatial_config,
                temporal_config=temporal_config,
            )
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            logger.error("tile_size=%d OOM — recording and continuing: %s", tile_size, exc)
            rows.append({"tile_size": tile_size, "label": combo.label, "oom": True})
            results = []
        finally:
            torch.cuda.empty_cache()

        for r in results:
            row = asdict(r)
            row["tile_size"] = tile_size
            row["oom"] = False
            rows.append(row)

        if not rows:
            continue

        fieldnames = sorted({k for row in rows for k in row})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info("Saved %d results so far → %s", len(rows), csv_path)

    if not rows:
        logger.warning("No results to save.")


if __name__ == "__main__":
    main()
