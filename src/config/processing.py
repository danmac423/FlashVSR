"""Configuration dataclasses for video processing."""

from dataclasses import dataclass
from enum import Enum

from src.models.wan_video_dit import AttentionMode, MaskAttentionMode


class OutputMode(Enum):
    """Output mode for processed video."""

    VIDEO = "video"
    FRAMES = "frames"
    NONE = "none"


class QuantizationMode(Enum):
    NONE = "none"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    INT8_DYNAMIC = "int8_dynamic"


@dataclass
class QuantizationConfig:
    mode: QuantizationMode = QuantizationMode.NONE


@dataclass
class VRAMConfig:
    """VRAM offloading config. Only effective when quantization is disabled."""

    enabled: bool = False
    # None keeps all parameters in VRAM; set an int to limit persistent params in DiT
    num_persistent_param_in_dit: int | None = None


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""

    scale: int = 4
    color_fix: bool = True
    seed: int = 0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""

    attn_mode: AttentionMode = AttentionMode.FLASH
    mask_attn_mode: MaskAttentionMode | None = None


@dataclass
class SpatialTilingConfig:
    """Configuration for spatial tiling."""

    enabled: bool = True
    tile_size: tuple[int, int] = (192, 192)
    tile_overlap: int = 24


@dataclass
class TemporalTilingConfig:
    """Configuration for temporal tiling."""

    enabled: bool = True
    tile_size: int = 100
    tile_overlap: int = 6


@dataclass
class IOConfig:
    """Configuration for input/output."""

    input_path: str = "input/video.mp4"
    output_mode: OutputMode = OutputMode.VIDEO
    output_dir: str = "output"
