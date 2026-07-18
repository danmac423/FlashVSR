"""Dimension calculation utilities for video processing."""


def calculate_padded_frame_count(n: int) -> int:
    """Calculate largest frame count in format 8n+1 that is <= n.

    Args:
        n (int): Original frame count

    Returns:
        int: Padded frame count
    """
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def calculate_next_frame_requirement(n: int) -> int:
    """Calculate next frame count in format 8n+5.

    Args:
        n (int): Original frame count

    Returns:
        int: Next frame count
    """
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5


def compute_scaled_and_target_dims(
    h0: int, w0: int, scale: int = 4, multiple: int = 128
) -> tuple[int, int, int, int]:
    """Compute scaled dimensions and target dimensions aligned to multiple.

    Args:
        h0 (int): Original height
        w0 (int): Original width
        scale (int): Upscaling factor
        multiple (int): Multiple to align target dimensions

    Returns:
        tuple: scaled_height, scaled_width, target_height, target_width
    """
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sw, sh = w0 * scale, h0 * scale
    th = ((sh + multiple - 1) // multiple) * multiple
    tw = ((sw + multiple - 1) // multiple) * multiple
    return sh, sw, th, tw
