"""Tensor manipulation utilities for video processing."""

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models.utils import clean_vram
from src.utils.dimension import (
    calculate_padded_frame_count,
    compute_scaled_and_target_dims,
)


def convert_tensor_to_video(frames: torch.Tensor) -> torch.Tensor:
    """Convert tensor from model output format to video format.

    Args:
        frames (torch.Tensor): Input tensor of shape (C, N, H, W)

    Returns:
        torch.Tensor: Output tensor of shape (N, H, W, C) with values in [0, 1]
    """
    video_permuted = rearrange(frames, "C N H W -> N H W C")
    video_final = (video_permuted.to(torch.float16) + 1.0) / 2.0
    return video_final


def upscale_and_normalize_tensor(
    frame_tensor: torch.Tensor,
    sh: int,
    sw: int,
    th: int,
    tw: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Upscale and normalize a single frame tensor.

    Args:
        frame_tensor (torch.Tensor): Input frame tensor of shape (H, W, C)
        scaled_height (int): Scaled height before padding
        scaled_width (int): Scaled width before padding
        target_height (int): Target height after padding
        target_width (int): Target width after padding

    Returns:
        tuple: Upscaled and normalized tensor of shape (H, W, C)
        and padding applied (left, right, top, bottom)
    """
    tensor_bchw = rearrange(frame_tensor, "H W C -> 1 C H W")

    upscaled_tensor = F.interpolate(tensor_bchw, size=(sh, sw), mode="bicubic", align_corners=False)

    pad_h = th - sh
    pad_w = tw - sw
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_tensor = F.pad(
        upscaled_tensor, [pad_left, pad_right, pad_top, pad_bottom], mode="reflect"
    )

    normalized_tensor = padded_tensor.clamp(0.0, 1.0).mul(255).round().div(255).mul(2).sub(1.0)

    return rearrange(normalized_tensor, "1 C H W -> H W C"), (
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
    )


def prepare_input_tensor(
    image_tensor: torch.Tensor,
    device: torch.device,
    scale: int = 4,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, int, int, int, int, int, tuple[int, int, int, int]]:
    """Prepare input tensor for model processing. Upscales, normalizes, and pads frames
    temporally and spatially to meet model requirements.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (N, H, W, C)
        device (torch.device): Device to load tensors onto
        scale (int): Upscaling factor
        dtype (torch.dtype): Data type for tensors

    Returns:
        tuple: Prepared tensor and related dimensions
            - torch.Tensor: Prepared tensor of shape (1, C, N, H, W)
            - int: Scaled height before padding
            - int: Scaled width before padding
            - int: Target height after padding
            - int: Target width after padding
            - int: Number of frames after temporal padding
            - tuple: Padding applied (left, right, top, bottom)
    """
    N0, h0, w0, _ = image_tensor.shape

    multiple = 128
    sh, sw, th, tw = compute_scaled_and_target_dims(h0, w0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    N = calculate_padded_frame_count(num_frames_with_padding)

    if N == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")

    frames = []
    for i in range(N):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor, padding = upscale_and_normalize_tensor(frame_slice, sh, sw, th, tw)
        tensor = tensor.to("cpu").to(dtype)

        frames.append(tensor)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    del frames
    vid_final = rearrange(vid_stacked, "N H W C -> 1 C N H W")

    del vid_stacked
    clean_vram()

    return vid_final, sh, sw, th, tw, N, padding


def remove_padding(
    frames: torch.Tensor, sh: int, sw: int, th: int, tw: int, padding: tuple[int, int, int, int]
) -> torch.Tensor:
    """Remove padding from processed frames.

    Args:
        frames (torch.Tensor): Processed frames tensor of shape (N, H, W, C)
        sh (int): Scaled height before padding
        sw (int): Scaled width before padding
        th (int): Target height after padding
        tw (int): Target width after padding
        padding (tuple[int, int, int, int]): Padding applied (left, right, top, bottom)

    Returns:
        torch.Tensor: Frames tensor with padding removed of shape (N, sh, sw, C)
    """
    pad_left, pad_right, pad_top, pad_bottom = padding

    if th > sh:
        frames = frames[:, pad_top : th - pad_bottom, :, :]
    if tw > sw:
        frames = frames[:, :, pad_left : tw - pad_right, :]

    return frames
