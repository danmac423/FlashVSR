"""
Run FlashVSR on a dataset of PNG frame sequences.

Input structure:  <input_root>/<clip_name>/<frame>.png
Output structure: <output_root>/<clip_name>/<frame>.png

Usage:
    uv run python scripts/run_dataset.py \\
        --input datasets/VideoLQ/LQ \\
        --output datasets/VideoLQ/SR
"""

import argparse
import logging
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.processing import (
    AttentionConfig,
    IOConfig,
    OutputMode,
    ProcessingConfig,
    QuantizationConfig,
    QuantizationMode,
    SpatialTilingConfig,
    TemporalTilingConfig,
    VRAMConfig,
)
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode
from src.processing.pipeline import init_pipeline
from src.processing.video import flashvsr
from src.utils.logger import get_logger, setup_logger

logger = get_logger()

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _collect_clips(root: str) -> list[tuple[str, str]]:
    """Return sorted list of (clip_name, clip_dir) for all image-sequence subdirs."""
    clips = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in os.listdir(full)
        ):
            clips.append((entry, full))
    return clips


def _load_frames_tensor(clip_dir: str) -> torch.Tensor:
    """Read sorted PNG frames and return (T, H, W, 3) float16 [0,1] tensor."""
    paths = sorted(
        os.path.join(clip_dir, f)
        for f in os.listdir(clip_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    frames = [
        torch.from_numpy(__import__("numpy").array(Image.open(p).convert("RGB"))) for p in paths
    ]
    return torch.stack(frames).to(torch.float16) / 255.0


def process_clip(
    pipe,
    clip_name: str,
    clip_dir: str,
    out_root: str,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
    temporal_config: TemporalTilingConfig,
    skip_existing: bool,
) -> None:
    out_dir = os.path.join(out_root, clip_name)

    if skip_existing and os.path.isdir(out_dir) and os.listdir(out_dir):
        logger.info("Skipping %s (already processed)", clip_name)
        return

    logger.info("Processing clip: %s", clip_name)
    frames = _load_frames_tensor(clip_dir)
    logger.info("  %d frames  %dx%d", frames.shape[0], frames.shape[2], frames.shape[1])

    os.makedirs(out_dir, exist_ok=True)
    flashvsr(
        pipe=pipe,
        io_config=IOConfig(input_path="", output_mode=OutputMode.FRAMES, output_dir=out_dir),
        proc_config=proc_config,
        spatial_tiling_config=spatial_config,
        temporal_tiling_config=temporal_config,
        input_frames=frames,
    )


def main() -> None:
    setup_logger(logging.INFO)

    p = argparse.ArgumentParser(description="Run FlashVSR on a dataset of PNG frame sequences")
    p.add_argument("--input", required=True, help="Root dir with LQ clip subdirectories")
    p.add_argument("--output", required=True, help="Root dir for SR output")
    p.add_argument("--model", default="FlashVSR-v1.1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip_existing",
        help="Skip clips whose output directory already exists and is non-empty",
    )
    # spatial tiling
    p.add_argument("--tile-size", type=int, nargs=2, default=[192, 192], metavar=("H", "W"))
    p.add_argument("--tile-overlap", type=int, default=24)
    p.add_argument("--no-spatial-tiling", action="store_true", dest="no_spatial")
    # temporal tiling
    p.add_argument("--temporal-tile-size", type=int, default=100, dest="temporal_tile_size")
    p.add_argument("--temporal-tile-overlap", type=int, default=6, dest="temporal_tile_overlap")
    p.add_argument("--no-temporal-tiling", action="store_true", dest="no_temporal")
    # model config
    p.add_argument(
        "--quantization", default="none", choices=["none", "int8_weight_only", "int8_dynamic"]
    )
    p.add_argument("--attn-mode", default="flash", choices=["flash", "sage"], dest="attn_mode")
    p.add_argument(
        "--mask-attn-mode",
        default="block_sparse",
        choices=["block_sparse", "sparse_sage", "none"],
        dest="mask_attn_mode",
    )
    args = p.parse_args()

    clips = _collect_clips(args.input)
    if not clips:
        logger.error("No clips found in %s", args.input)
        sys.exit(1)
    logger.info("Found %d clips in %s", len(clips), args.input)

    device = torch.device(args.device)
    quant_mode = QuantizationMode(args.quantization)

    attn_mode = AttentionMode(args.attn_mode)
    mask_attn_mode = (
        None if args.mask_attn_mode == "none" else MaskAttentionMode(args.mask_attn_mode)
    )

    pipe = init_pipeline(
        args.model,
        device,
        torch.bfloat16,
        attention_config=AttentionConfig(attn_mode=attn_mode, mask_attn_mode=mask_attn_mode),
        quantization_config=QuantizationConfig(mode=quant_mode),
        vram_config=VRAMConfig(enabled=False),
    )

    proc_config = ProcessingConfig(
        scale=4,
        seed=0,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
    )
    spatial_config = SpatialTilingConfig(
        enabled=not args.no_spatial,
        tile_size=tuple(args.tile_size),
        tile_overlap=args.tile_overlap,
    )
    temporal_config = TemporalTilingConfig(
        enabled=not args.no_temporal,
        tile_size=args.temporal_tile_size,
        tile_overlap=args.temporal_tile_overlap,
    )

    for i, (clip_name, clip_dir) in enumerate(clips):
        logger.info("[%d/%d] %s", i + 1, len(clips), clip_name)
        try:
            process_clip(
                pipe,
                clip_name,
                clip_dir,
                args.output,
                proc_config,
                spatial_config,
                temporal_config,
                args.skip_existing,
            )
        except Exception as exc:
            logger.error("Failed on clip %s: %s", clip_name, exc, exc_info=True)


if __name__ == "__main__":
    main()
