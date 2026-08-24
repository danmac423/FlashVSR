"""Video processing functions for FlashVSR."""

import os

import torch
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from torchvision.io import write_png
from tqdm import tqdm

from src.config.processing import (
    IOConfig,
    OutputMode,
    ProcessingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
)
from src.models.utils import clean_vram
from src.pipelines.base import BasePipeline
from src.utils.dimension import calculate_next_frame_requirement
from src.utils.logger import get_logger
from src.utils.tensor import convert_tensor_to_video, prepare_input_tensor, remove_padding
from src.utils.tiling import (
    blend_temporal_overlap,
    calculate_spatial_tile_coords,
    calculate_temporal_tile_ranges,
    create_spatial_blend_mask,
)

logger = get_logger()


def process_single_temporal_chunk(
    pipe: BasePipeline,
    frames_chunk: torch.Tensor,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
) -> torch.Tensor:
    """Process a single temporal chunk of frames.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        frames_chunk (torch.Tensor): Input frames tensor of shape (N, H, W, C)
        proc_config (ProcessingConfig): Processing configuration
        spatial_config (SpatialTilingConfig): Spatial tiling configuration

    Returns:
        torch.Tensor: Processed frames tensor of shape (N, H, W, C)
    """
    _device = pipe.device
    dtype = pipe.torch_dtype

    add = calculate_next_frame_requirement(frames_chunk.shape[0]) - frames_chunk.shape[0]
    if add > 0:
        logger.debug("Padding frames: %d → %d", frames_chunk.shape[0], frames_chunk.shape[0] + add)
        padding_frames = frames_chunk[-1:, :, :, :].repeat(add, 1, 1, 1)
        _frames = torch.cat([frames_chunk, padding_frames], dim=0)
    else:
        _frames = frames_chunk

    if spatial_config.enabled:
        N, H, W, C = _frames.shape

        final_output_canvas = torch.zeros(
            (N, H * proc_config.scale, W * proc_config.scale, C), dtype=torch.float16, device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_spatial_tile_coords(
            H, W, spatial_config.tile_size, spatial_config.tile_overlap
        )

        for i, (x1, y1, x2, y2) in enumerate(tqdm(tile_coords, desc="Spatial tiles", leave=False)):
            logger.debug("Tile %d/%d: (%d,%d)-(%d,%d)", i + 1, len(tile_coords), x1, y1, x2, y2)
            input_tile = _frames[:, y1:y2, x1:x2, :]

            LQ_tile, sh, sw, th, tw, N, padding = prepare_input_tensor(
                input_tile, _device, scale=proc_config.scale, dtype=dtype
            )
            LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                seed=proc_config.seed,
                LQ_video=LQ_tile,
                num_frames=N,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
                kv_ratio=proc_config.kv_ratio,
                local_range=proc_config.local_range,
                color_fix=proc_config.color_fix,
            )

            processed_tile_cpu = convert_tensor_to_video(output_tile_gpu).to("cpu")

            processed_tile_cpu = remove_padding(processed_tile_cpu, sh, sw, th, tw, padding)

            mask = create_spatial_blend_mask(
                (sh, sw), spatial_config.tile_overlap * proc_config.scale
            ).to("cpu")

            out_x1, out_y1 = x1 * proc_config.scale, y1 * proc_config.scale
            out_x2, out_y2 = out_x1 + sw, out_y1 + sh

            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask

            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
        del final_output_canvas, weight_sum_canvas
    else:
        LQ, sh, sw, th, tw, N, padding = prepare_input_tensor(
            _frames, _device, scale=proc_config.scale, dtype=dtype
        )
        LQ = LQ.to(_device)

        video = pipe(
            seed=proc_config.seed,
            LQ_video=LQ,
            num_frames=N,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=proc_config.kv_ratio,
            local_range=proc_config.local_range,
            color_fix=proc_config.color_fix,
        )

        final_output = convert_tensor_to_video(video).to("cpu")

        final_output = remove_padding(final_output, sh, sw, th, tw, padding)

        del video, LQ
        clean_vram()

    return final_output[: frames_chunk.shape[0], :, :, :]


def flashvsr(
    pipe: BasePipeline,
    io_config: IOConfig,
    proc_config: ProcessingConfig,
    spatial_tiling_config: SpatialTilingConfig,
    temporal_tiling_config: TemporalTilingConfig,
    input_frames: torch.Tensor | None = None,
):
    """Process video using FlashVSR with temporal tiling support.
    Writes output incrementally to avoid RAM overflow.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        io_config (IOConfig): Input/output configuration
        proc_config (ProcessingConfig): Processing configuration
        spatial_tiling_config (SpatialTilingConfig): Spatial tiling configuration
        temporal_tiling_config (TemporalTilingConfig): Temporal tiling configuration
        input_frames: Optional pre-loaded tensor (T, H, W, C) float16 [0,1].
            When provided, skips VideoDecoder - no file I/O on input.
    """
    if input_frames is not None:
        num_frames = input_frames.shape[0]
        avg_fps = 30.0

        def _get_chunk(start: int, end: int) -> torch.Tensor:
            return input_frames[start:end]
    else:
        video_decoder = VideoDecoder(io_config.input_path, dimension_order="NHWC")
        meta = video_decoder.metadata
        num_frames = meta.num_frames
        avg_fps = meta.average_fps
        logger.info(
            "Input: %s | %dx%d | %.2f fps | %d frames",
            io_config.input_path,
            meta.width,
            meta.height,
            meta.average_fps,
            meta.num_frames,
        )

        def _get_chunk(start: int, end: int) -> torch.Tensor:
            return video_decoder.get_frames_in_range(start, end).data.to(torch.float16) / 255.0

    def _write_output(frames_to_write: torch.Tensor, chunk_idx: int, frame_counter: int) -> int:
        if io_config.output_mode == OutputMode.VIDEO:
            encoder = VideoEncoder(
                frames=rearrange(frames_to_write, "N H W C -> N C H W")
                .clamp(0, 1)
                .mul_(255)
                .to(torch.uint8),
                frame_rate=avg_fps,
            )
            out_path = os.path.join(io_config.output_dir, f"chunk_{chunk_idx:03d}.mp4")
            encoder.to_file(out_path, codec="libx264", crf=0, pixel_format="yuv420p")
            logger.info("Output written to: %s", out_path)
        elif io_config.output_mode == OutputMode.FRAMES:
            final = (
                rearrange(frames_to_write, "N H W C -> N C H W")
                .clamp(0, 1)
                .mul_(255)
                .to(torch.uint8)
            )
            for i in range(final.shape[0]):
                write_png(
                    final[i],
                    os.path.join(io_config.output_dir, f"{frame_counter:08d}.png"),
                    compression_level=0,
                )
                frame_counter += 1
        return frame_counter

    if temporal_tiling_config.enabled and num_frames > temporal_tiling_config.tile_size:
        chunks = calculate_temporal_tile_ranges(
            num_frames,
            temporal_tiling_config.tile_size,
            temporal_tiling_config.tile_overlap,
        )

        prev_chunk_tail = None
        frame_counter = 0

        for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Temporal Chunks", leave=False)):
            frames_chunk = _get_chunk(start, end)
            current_chunk_output = process_single_temporal_chunk(
                pipe=pipe,
                frames_chunk=frames_chunk,
                proc_config=proc_config,
                spatial_config=spatial_tiling_config,
            )
            del frames_chunk
            clean_vram()

            if chunk_idx == 0:
                frames_to_write = (
                    current_chunk_output[: -temporal_tiling_config.tile_overlap]
                    if temporal_tiling_config.tile_overlap > 0
                    else current_chunk_output
                )
                prev_chunk_tail = (
                    current_chunk_output[-temporal_tiling_config.tile_overlap :]
                    if temporal_tiling_config.tile_overlap > 0
                    else None
                )
            else:
                overlap_head = current_chunk_output[: temporal_tiling_config.tile_overlap]
                blended_overlap = blend_temporal_overlap(
                    prev_chunk_tail, overlap_head, temporal_tiling_config.tile_overlap
                )

                rest_of_chunk = current_chunk_output[temporal_tiling_config.tile_overlap :]
                if chunk_idx == len(chunks) - 1:
                    frames_to_write = torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    prev_chunk_tail = None
                else:
                    frames_to_write = (
                        torch.cat(
                            [
                                blended_overlap,
                                rest_of_chunk[: -temporal_tiling_config.tile_overlap],
                            ],
                            dim=0,
                        )
                        if temporal_tiling_config.tile_overlap > 0
                        else torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    )
                    prev_chunk_tail = (
                        rest_of_chunk[-temporal_tiling_config.tile_overlap :]
                        if temporal_tiling_config.tile_overlap > 0
                        and rest_of_chunk.shape[0] >= temporal_tiling_config.tile_overlap
                        else rest_of_chunk
                    )

            frame_counter = _write_output(frames_to_write, chunk_idx, frame_counter)
            del current_chunk_output, frames_to_write
            clean_vram()

    else:
        logger.info("Processing entire video as one chunk")
        frames = _get_chunk(0, num_frames)

        final_output = process_single_temporal_chunk(
            pipe=pipe,
            frames_chunk=frames,
            proc_config=proc_config,
            spatial_config=spatial_tiling_config,
        )

        if io_config.output_mode == OutputMode.FRAMES:
            logger.info(
                "Output written to: %s (%d frames)", io_config.output_dir, final_output.shape[0]
            )
        _write_output(final_output, 0, 0)

        del frames, final_output
        clean_vram()
