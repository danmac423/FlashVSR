"""Tiling utilities for spatial and temporal video processing."""

import math

import torch
from einops import rearrange


def calculate_spatial_tile_coords(
    height: int, width: int, tile_size: tuple[int, int], overlap: int
) -> list[tuple[int, int, int, int]]:
    """Calculate spatial tile coordinates with overlap for tiled processing.

    Args:
        height (int): Height of the frame
        width (int): Width of the frame
        tile_size (tuple[int, int]): (tile_width, tile_height)
        overlap (int): Overlap size

    Returns:
        list: list of tile coordinates (x1, y1, x2, y2)
    """
    coords = []
    tile_w, tile_h = tile_size

    stride_w = tile_w - overlap
    stride_h = tile_h - overlap

    num_rows = math.ceil((height - overlap) / stride_h)
    num_cols = math.ceil((width - overlap) / stride_w)

    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride_h
            x1 = c * stride_w

            y2 = min(y1 + tile_h, height)
            x2 = min(x1 + tile_w, width)

            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)
            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)

            coords.append((x1, y1, x2, y2))

    return coords


def calculate_temporal_tile_ranges(
    total_frames: int, chunk_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Calculate temporal chunk ranges with overlap for tiled processing.

    Args:
        total_frames (int): Total number of frames in the video
        chunk_size (int): Size of each temporal chunk
        overlap (int): Overlap size between chunks

    Returns:
        list: list of temporal chunk ranges (start_frame, end_frame)
    """
    chunks = []
    stride = chunk_size - overlap

    num_chunks = math.ceil((total_frames - overlap) / stride)

    for i in range(num_chunks):
        start = i * stride
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))

    return chunks


def create_spatial_blend_mask(size: tuple[int, int], overlap: int) -> torch.Tensor:
    """Create feather mask for blending spatial tiles.

    Args:
        size (tuple[int, int]): Size of the mask (height, width)
        overlap (int): Overlap size for blending

    Returns:
        torch.Tensor: Blend mask of shape (1, H, W, 1)
    """
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)

    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(
        mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1)
    )

    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(
        mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1)
    )

    return rearrange(mask, "N C H W -> N H W C")


def blend_temporal_overlap(
    prev_chunk_tail: torch.Tensor, current_chunk_head: torch.Tensor, overlap: int
) -> torch.Tensor:
    """Blend overlapping region of two temporal chunks.

    Args:
        prev_chunk_tail (torch.Tensor): Last 'overlap' frames from previous chunk
        current_chunk_head (torch.Tensor): First 'overlap' frames from current chunk
        overlap (int): Number of overlapping frames

    Returns:
        torch.Tensor: Blended frames
    """
    if overlap <= 0 or prev_chunk_tail is None:
        return current_chunk_head

    actual_overlap = min(overlap, prev_chunk_tail.shape[0], current_chunk_head.shape[0])

    if actual_overlap <= 0:
        return current_chunk_head

    blend_weight = torch.linspace(1, 0, actual_overlap).view(-1, 1, 1, 1)

    blended = prev_chunk_tail[-actual_overlap:] * blend_weight + current_chunk_head[
        :actual_overlap
    ] * (1 - blend_weight)

    return blended
