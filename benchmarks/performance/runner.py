"""FlashVSR performance benchmark runner.

Usage:
    python -m benchmarks.performance.runner --video inputs/example0.mp4
    python -m benchmarks.performance.runner --video inputs/example0.mp4 --warmup-runs 0 --measured-runs 5
    python -m benchmarks.performance.runner --video inputs/example0.mp4 --no-spatial-tiling
"""

import argparse
import csv
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
from torchcodec.decoders import VideoDecoder

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
from src.utils.tiling import calculate_spatial_tile_coords

logger = get_logger()

BENCHMARK_NUM_FRAMES = 101
BENCHMARK_HEIGHT = 192
BENCHMARK_WIDTH = 352


@dataclass
class Combination:
    attn_mode: AttentionMode
    mask_attn_mode: MaskAttentionMode | None
    quantization_mode: QuantizationMode

    @property
    def label(self) -> str:
        mask = self.mask_attn_mode.value if self.mask_attn_mode else "none"
        return f"{self.attn_mode.value}/{mask}/{self.quantization_mode.value}"


@dataclass
class RunResult:
    label: str
    attn_mode: str
    mask_attn_mode: str
    quantization_mode: str
    run_index: int
    input_height: int
    input_width: int
    num_frames: int
    num_spatial_tiles: int
    init_time_s: float
    inference_time_s: float
    time_per_frame_s: float
    peak_vram_mb: float
    vram_after_init_mb: float


DEFAULT_COMBINATIONS: list[Combination] = [
    Combination(AttentionMode.FLASH, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.NONE),
    Combination(
        AttentionMode.FLASH, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.INT8_WEIGHT_ONLY
    ),
    Combination(AttentionMode.FLASH, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.INT8_DYNAMIC),
    Combination(AttentionMode.FLASH, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.NONE),
    Combination(
        AttentionMode.FLASH, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.INT8_WEIGHT_ONLY
    ),
    Combination(AttentionMode.FLASH, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.INT8_DYNAMIC),
    Combination(AttentionMode.SAGE, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.NONE),
    Combination(
        AttentionMode.SAGE, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.INT8_WEIGHT_ONLY
    ),
    Combination(AttentionMode.SAGE, MaskAttentionMode.BLOCK_SPARSE, QuantizationMode.INT8_DYNAMIC),
    Combination(AttentionMode.SAGE, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.NONE),
    Combination(
        AttentionMode.SAGE, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.INT8_WEIGHT_ONLY
    ),
    Combination(AttentionMode.SAGE, MaskAttentionMode.SPARSE_SAGE, QuantizationMode.INT8_DYNAMIC),
]


def _get_video_metadata(video_path: str) -> tuple[int, int, int]:
    """Return (num_frames, height, width) for the given video."""
    decoder = VideoDecoder(video_path)
    meta = decoder.metadata
    num_frames = meta.num_frames
    if num_frames is None:
        num_frames = sum(1 for _ in decoder)
    return int(num_frames), int(meta.height), int(meta.width)


def _load_benchmark_frames(video_path: str) -> torch.Tensor:
    """Validate dimensions and load frames into memory as a float16 tensor.

    Raises ValueError if any dimension is smaller than required.
    Returns tensor (BENCHMARK_NUM_FRAMES, BENCHMARK_HEIGHT, BENCHMARK_WIDTH, 3) float16 [0, 1].
    """
    num_frames, height, width = _get_video_metadata(video_path)

    if num_frames < BENCHMARK_NUM_FRAMES or height < BENCHMARK_HEIGHT or width < BENCHMARK_WIDTH:
        raise ValueError(
            f"Video too small for benchmark: {num_frames}x{height}x{width}. "
            f"Required at least {BENCHMARK_NUM_FRAMES}x{BENCHMARK_HEIGHT}x{BENCHMARK_WIDTH}."
        )

    decoder = VideoDecoder(video_path, dimension_order="NHWC")
    frames = decoder.get_frames_in_range(0, BENCHMARK_NUM_FRAMES).data.to(torch.float16) / 255.0

    if height > BENCHMARK_HEIGHT or width > BENCHMARK_WIDTH:
        crop_y = (height - BENCHMARK_HEIGHT) // 2
        crop_x = (width - BENCHMARK_WIDTH) // 2
        frames = frames[:, crop_y : crop_y + BENCHMARK_HEIGHT, crop_x : crop_x + BENCHMARK_WIDTH, :]
        logger.info(
            "Trimmed input: %dx%dx%d → %dx%dx%d",
            num_frames,
            height,
            width,
            BENCHMARK_NUM_FRAMES,
            BENCHMARK_HEIGHT,
            BENCHMARK_WIDTH,
        )

    return frames


def _vram_reserved_mb(device: torch.device) -> float:
    return torch.cuda.memory_reserved(device.index) / 1024**2


def _peak_vram_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_reserved(device.index) / 1024**2


def _run_inference(
    pipe,
    frames: torch.Tensor,
    combo: Combination,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
    temporal_config: TemporalTilingConfig,
) -> None:
    run_proc = ProcessingConfig(
        scale=proc_config.scale,
        color_fix=proc_config.color_fix,
        seed=proc_config.seed,
        sparse_ratio=proc_config.sparse_ratio,
        kv_ratio=proc_config.kv_ratio,
        local_range=proc_config.local_range,
    )
    flashvsr(
        pipe=pipe,
        io_config=IOConfig(input_path="", output_mode=OutputMode.NONE, output_dir=""),
        proc_config=run_proc,
        spatial_tiling_config=spatial_config,
        temporal_tiling_config=temporal_config,
        input_frames=frames,
    )


def benchmark(
    video_path: str,
    model: str,
    device: torch.device,
    dtype: torch.dtype,
    combinations: list[Combination],
    num_warmup_runs: int,
    num_measured_runs: int,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
    temporal_config: TemporalTilingConfig,
) -> list[RunResult]:
    frames = _load_benchmark_frames(video_path)

    torch.cuda.set_device(device)

    num_spatial_tiles = (
        len(
            calculate_spatial_tile_coords(
                BENCHMARK_HEIGHT,
                BENCHMARK_WIDTH,
                spatial_config.tile_size,
                spatial_config.tile_overlap,
            )
        )
        if spatial_config.enabled
        else 1
    )
    logger.info(
        "Input: %s  |  %dx%d  |  frames: %d  |  spatial tiles/frame: %d",
        video_path,
        BENCHMARK_WIDTH,
        BENCHMARK_HEIGHT,
        BENCHMARK_NUM_FRAMES,
        num_spatial_tiles,
    )
    logger.info(
        "Combinations: %d  |  warmup: %d  |  measured: %d",
        len(combinations),
        num_warmup_runs,
        num_measured_runs,
    )

    results: list[RunResult] = []

    for combo in combinations:
        logger.info("=== %s ===", combo.label)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device.index)

        try:
            t0 = time.perf_counter()
            pipe = init_pipeline(
                model,
                device,
                dtype,
                attention_config=AttentionConfig(
                    attn_mode=combo.attn_mode, mask_attn_mode=combo.mask_attn_mode
                ),
                quantization_config=QuantizationConfig(mode=combo.quantization_mode),
                vram_config=VRAMConfig(enabled=False),
            )
            init_time = time.perf_counter() - t0
        except Exception as exc:
            logger.error("Init failed — skipping: %s", exc)
            continue

        vram_after_init = _vram_reserved_mb(device)
        logger.info("Init %.2fs  |  VRAM after init %.0f MB", init_time, vram_after_init)

        warmup_ok = True
        for i in range(num_warmup_runs):
            logger.info("Warmup %d/%d...", i + 1, num_warmup_runs)
            try:
                _run_inference(pipe, frames, combo, proc_config, spatial_config, temporal_config)
            except Exception as exc:
                logger.error("Warmup failed — skipping combination: %s", exc)
                warmup_ok = False
                break

        if warmup_ok:
            for i in range(num_measured_runs):
                torch.cuda.reset_peak_memory_stats(device.index)
                t0 = time.perf_counter()
                _run_inference(pipe, frames, combo, proc_config, spatial_config, temporal_config)
                inference_time = time.perf_counter() - t0

                peak = _peak_vram_mb(device)
                tpf = inference_time / BENCHMARK_NUM_FRAMES
                logger.info(
                    "Run %d/%d  |  %.2fs total  |  %.3fs/frame  |  peak VRAM %.0f MB",
                    i + 1,
                    num_measured_runs,
                    inference_time,
                    tpf,
                    peak,
                )
                results.append(
                    RunResult(
                        label=combo.label,
                        attn_mode=combo.attn_mode.value,
                        mask_attn_mode=combo.mask_attn_mode.value
                        if combo.mask_attn_mode
                        else "none",
                        quantization_mode=combo.quantization_mode.value,
                        run_index=i,
                        input_height=BENCHMARK_HEIGHT,
                        input_width=BENCHMARK_WIDTH,
                        num_frames=BENCHMARK_NUM_FRAMES,
                        num_spatial_tiles=num_spatial_tiles,
                        init_time_s=round(init_time, 4),
                        inference_time_s=round(inference_time, 4),
                        time_per_frame_s=round(tpf, 4),
                        peak_vram_mb=round(peak, 1),
                        vram_after_init_mb=round(vram_after_init, 1),
                    )
                )

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return results


def save_results(results: list[RunResult], output_dir: str) -> None:
    if not results:
        logger.warning("No results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_dir, f"benchmark_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    json_path = os.path.join(output_dir, f"benchmark_{ts}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    logger.info("Saved %d results → %s", len(results), output_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlashVSR performance benchmark")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--model", default="FlashVSR-v1.1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup-runs", type=int, default=1, dest="warmup_runs")
    p.add_argument("--measured-runs", type=int, default=3, dest="measured_runs")
    p.add_argument("--output-dir", default="benchmarks/performance/results", dest="output_dir")
    p.add_argument(
        "--tile-size", type=int, nargs=2, default=[192, 192], metavar=("H", "W"), dest="tile_size"
    )
    p.add_argument("--tile-overlap", type=int, default=24, dest="tile_overlap")
    p.add_argument("--no-spatial-tiling", action="store_true", dest="no_spatial")
    p.add_argument("--no-temporal-tiling", action="store_true", dest="no_temporal")
    return p.parse_args()


def main() -> None:
    setup_logger(logging.INFO)
    args = _parse_args()

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
        enabled=not args.no_temporal, tile_size=100, tile_overlap=6
    )

    results = benchmark(
        video_path=args.video,
        model=args.model,
        device=torch.device(args.device),
        dtype=torch.bfloat16,
        combinations=DEFAULT_COMBINATIONS,
        num_warmup_runs=args.warmup_runs,
        num_measured_runs=args.measured_runs,
        proc_config=proc_config,
        spatial_config=spatial_config,
        temporal_config=temporal_config,
    )

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
