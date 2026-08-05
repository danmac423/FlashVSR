import argparse
import logging
import os

import torch

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
from src.models.utils import get_device_list
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode
from src.processing.pipeline import init_pipeline
from src.processing.video import flashvsr
from src.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a single video recording with FlashVSR."
    )

    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for the output. Defaults to '<video_name>_output'.",
    )
    parser.add_argument(
        "--output-mode",
        choices=[m.value for m in OutputMode],
        default=OutputMode.VIDEO.value,
        help="Whether to save a video, individual frames, or nothing to disk.",
    )
    parser.add_argument(
        "--model",
        default="FlashVSR-v1.1",
        help="Model name to load (downloaded from the JunhaoZhuang HF repo).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on, e.g. 'cuda:0' or 'mps'. Autodetected if omitted.",
    )

    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--sparse-ratio", type=float, default=2.0, help="Sparse attention ratio."
    )
    parser.add_argument(
        "--kv-ratio", type=float, default=3.0, help="Key/value ratio."
    )
    parser.add_argument(
        "--local-range", type=int, default=11, help="Local attention range."
    )
    parser.add_argument(
        "--no-color-fix",
        action="store_false",
        dest="color_fix",
        help="Disable color fixing.",
    )

    parser.add_argument(
        "--attn-mode",
        choices=[m.value for m in AttentionMode],
        default=AttentionMode.FLASH.value,
        help="Attention implementation.",
    )
    parser.add_argument(
        "--mask-attn-mode",
        choices=[m.value for m in MaskAttentionMode] + ["none"],
        default=MaskAttentionMode.BLOCK_SPARSE.value,
        help="Masked attention mode, or 'none' to disable.",
    )

    parser.add_argument(
        "--quantization",
        choices=[m.value for m in QuantizationMode],
        default=QuantizationMode.NONE.value,
        help="Weight quantization mode.",
    )

    parser.add_argument(
        "--no-spatial-tiling",
        action="store_false",
        dest="spatial_tiling",
        help="Disable spatial tiling.",
    )
    parser.add_argument(
        "--spatial-tile-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(192, 192),
        help="Spatial tile size.",
    )
    parser.add_argument(
        "--spatial-tile-overlap", type=int, default=24, help="Spatial tile overlap."
    )

    parser.add_argument(
        "--temporal-tiling",
        action="store_true",
        help="Enable temporal tiling.",
    )
    parser.add_argument(
        "--temporal-tile-size", type=int, default=100, help="Temporal tile size."
    )
    parser.add_argument(
        "--temporal-tile-overlap",
        type=int,
        default=6,
        help="Temporal tile overlap.",
    )

    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        if requested not in get_device_list():
            raise RuntimeError(f"Device '{requested}' is not available!")
        return requested

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "auto"
    )
    if device == "auto" or device not in get_device_list():
        raise RuntimeError("No devices found to run FlashVSR!")
    return device


def main():
    setup_logger(logging.INFO)
    logger = get_logger()

    args = parse_args()

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or f"{video_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    proc_config = ProcessingConfig(
        scale=args.scale,
        seed=args.seed,
        sparse_ratio=args.sparse_ratio,
        kv_ratio=args.kv_ratio,
        local_range=args.local_range,
        color_fix=args.color_fix,
    )
    attention_config = AttentionConfig(
        attn_mode=AttentionMode(args.attn_mode),
        mask_attn_mode=(
            None
            if args.mask_attn_mode == "none"
            else MaskAttentionMode(args.mask_attn_mode)
        ),
    )
    spatial_tiling_config = SpatialTilingConfig(
        enabled=args.spatial_tiling,
        tile_size=tuple(args.spatial_tile_size),
        tile_overlap=args.spatial_tile_overlap,
    )
    temporal_tiling_config = TemporalTilingConfig(
        enabled=args.temporal_tiling,
        tile_size=args.temporal_tile_size,
        tile_overlap=args.temporal_tile_overlap,
    )
    io_config = IOConfig(
        input_path=args.input,
        output_mode=OutputMode(args.output_mode),
        output_dir=output_dir,
    )

    quantization_config = QuantizationConfig(mode=QuantizationMode(args.quantization))
    vram_config = VRAMConfig(enabled=False, num_persistent_param_in_dit=None)

    pipe = init_pipeline(
        args.model,
        device,
        torch.bfloat16,
        attention_config=attention_config,
        quantization_config=quantization_config,
        vram_config=vram_config,
    )

    flashvsr(
        pipe=pipe,
        io_config=io_config,
        proc_config=proc_config,
        spatial_tiling_config=spatial_tiling_config,
        temporal_tiling_config=temporal_tiling_config,
    )


if __name__ == "__main__":
    main()
