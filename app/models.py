from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from src.config.processing import (
    AttentionConfig,
    AttentionMode,
    MaskAttentionMode,
    ProcessingConfig,
    QuantizationConfig,
    QuantizationMode,
    SpatialTilingConfig,
    TemporalTilingConfig,
    VRAMConfig,
)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobParams(BaseModel):
    proc_config: ProcessingConfig = ProcessingConfig(
        scale=4,
        seed=0,
        sparse_ratio=2.0,
        kv_ratio=3,
        local_range=11,
        color_fix=True,
    )
    attention_config: AttentionConfig = AttentionConfig(
        attn_mode=AttentionMode.FLASH,
        mask_attn_mode=MaskAttentionMode.BLOCK_SPARSE,
    )
    spatial_tiling_config: SpatialTilingConfig = SpatialTilingConfig(
        enabled=True,
        tile_size=(192, 192),
        tile_overlap=24,
    )
    temporal_tiling_config: TemporalTilingConfig = TemporalTilingConfig(
        enabled=False,
        tile_size=100,
        tile_overlap=6,
    )
    quantization_config: QuantizationConfig = QuantizationConfig(mode=QuantizationMode.NONE)
    vram_config: VRAMConfig = VRAMConfig(enabled=False, num_persistent_param_in_dit=None)


class Job(BaseModel):
    id: str
    status: JobStatus
    progress: int = 0
    params: JobParams
    input_path: str
    output_path: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
