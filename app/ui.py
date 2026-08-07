import time
from pathlib import Path

import gradio as gr
import httpx

from app.models import JobParams
from src.config.processing import (
    AttentionConfig,
    ProcessingConfig,
    QuantizationConfig,
    QuantizationMode,
    SpatialTilingConfig,
    TemporalTilingConfig,
    VRAMConfig,
)
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode

API_URL = "http://localhost:8000"
OUTPUTS_DIR = Path(__file__).parent.parent / "data" / "outputs"

_ATTN_MODES = [m.value for m in AttentionMode]
_MASK_ATTN_MODES = [m.value for m in MaskAttentionMode]
_QUANT_MODES = [m.value for m in QuantizationMode]

# Fixed to the values used across all experiments/benchmarks (see main.py,
# scripts/run_dataset.py, scripts/tile_size_sweep.py, benchmarks/performance/runner.py).
# Not exposed in the UI.
_FIXED_SPARSE_RATIO = 2.0
_FIXED_KV_RATIO = 3.0
_FIXED_LOCAL_RANGE = 11
_FIXED_COLOR_FIX = True
_FIXED_SEED = 0
_FIXED_VRAM_ENABLED = False
_FIXED_NUM_PERSISTENT = None


def _build_params(
    spatial_enabled,
    spatial_tile_h,
    spatial_tile_w,
    spatial_overlap,
    temporal_enabled,
    temporal_tile_size,
    temporal_overlap,
    attn_mode,
    mask_attn_mode,
    quant_mode,
) -> JobParams:
    return JobParams(
        proc_config=ProcessingConfig(
            sparse_ratio=_FIXED_SPARSE_RATIO,
            kv_ratio=_FIXED_KV_RATIO,
            local_range=_FIXED_LOCAL_RANGE,
            color_fix=_FIXED_COLOR_FIX,
            seed=_FIXED_SEED,
        ),
        attention_config=AttentionConfig(
            attn_mode=AttentionMode(attn_mode),
            mask_attn_mode=MaskAttentionMode(mask_attn_mode) if mask_attn_mode != "none" else None,
        ),
        spatial_tiling_config=SpatialTilingConfig(
            enabled=spatial_enabled,
            tile_size=(int(spatial_tile_h), int(spatial_tile_w)),
            tile_overlap=int(spatial_overlap),
        ),
        temporal_tiling_config=TemporalTilingConfig(
            enabled=temporal_enabled,
            tile_size=int(temporal_tile_size),
            tile_overlap=int(temporal_overlap),
        ),
        quantization_config=QuantizationConfig(mode=QuantizationMode(quant_mode)),
        vram_config=VRAMConfig(
            enabled=_FIXED_VRAM_ENABLED,
            num_persistent_param_in_dit=_FIXED_NUM_PERSISTENT,
        ),
    )


def process_video(
    video_path,
    spatial_enabled,
    spatial_tile_h,
    spatial_tile_w,
    spatial_overlap,
    temporal_enabled,
    temporal_tile_size,
    temporal_overlap,
    attn_mode,
    mask_attn_mode,
    quant_mode,
):
    if video_path is None:
        raise gr.Error("Upload a video first.")

    params = _build_params(
        spatial_enabled,
        spatial_tile_h,
        spatial_tile_w,
        spatial_overlap,
        temporal_enabled,
        temporal_tile_size,
        temporal_overlap,
        attn_mode,
        mask_attn_mode,
        quant_mode,
    )

    filename = Path(video_path).name
    with open(video_path, "rb") as f:
        resp = httpx.post(
            f"{API_URL}/jobs",
            files={"file": (filename, f, "video/mp4")},
            data={"params": params.model_dump_json()},
            timeout=30,
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    yield "pending", None

    while True:
        time.sleep(4)
        resp = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]

        if status == "done":
            yield "done", job["output_path"]
            return
        elif status == "error":
            raise gr.Error(f"Job failed: {job['error']}")
        else:
            yield status, None


with gr.Blocks(title="FlashVSR") as demo:
    gr.Markdown("# FlashVSR — Video Super Resolution")

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input (LQ)")

            # with gr.Accordion("Spatial Tiling", open=False):
            #     spatial_enabled = gr.Checkbox(value=True, label="Enabled")
            #     spatial_tile_h = gr.Slider(64, 512, value=192, step=32, label="Tile Height")
            #     spatial_tile_w = gr.Slider(64, 512, value=192, step=32, label="Tile Width")
            #     spatial_overlap = gr.Slider(0, 64, value=24, step=4, label="Overlap")

            # with gr.Accordion("Temporal Tiling", open=False):
            #     temporal_enabled = gr.Checkbox(value=False, label="Enabled")
            #     temporal_tile_size = gr.Slider(
            #         20, 200, value=100, step=10, label="Tile Size (frames)"
            #     )
            #     temporal_overlap = gr.Slider(0, 20, value=6, step=1, label="Overlap (frames)")

            # with gr.Accordion("Attention", open=False):
            #     attn_mode = gr.Dropdown(_ATTN_MODES, value="flash", label="Attention Mode")
            #     mask_attn_mode = gr.Dropdown(
            #         _MASK_ATTN_MODES, value="block_sparse", label="Mask Attention Mode"
            #     )

            # with gr.Accordion("Quantization", open=False):
            #     quant_mode = gr.Dropdown(_QUANT_MODES, value="none", label="Quantization Mode")

            # run_btn = gr.Button("Run Super Resolution", variant="primary")

        with gr.Column():
            video_out = gr.Video(label="Output (4× SR)")

    with gr.Row():
        status_box = gr.Textbox(label="Status", interactive=False, value="")

    with gr.Row():
        with gr.Accordion("Spatial Tiling", open=False):
            spatial_enabled = gr.Checkbox(value=True, label="Enabled")
            spatial_tile_h = gr.Slider(64, 512, value=192, step=32, label="Tile Height")
            spatial_tile_w = gr.Slider(64, 512, value=192, step=32, label="Tile Width")
            spatial_overlap = gr.Slider(0, 64, value=24, step=4, label="Overlap")

        with gr.Accordion("Temporal Tiling", open=False):
            temporal_enabled = gr.Checkbox(value=False, label="Enabled")
            temporal_tile_size = gr.Slider(20, 200, value=100, step=10, label="Tile Size (frames)")
            temporal_overlap = gr.Slider(0, 20, value=6, step=1, label="Overlap (frames)")

        with gr.Accordion("Attention", open=False):
            attn_mode = gr.Dropdown(_ATTN_MODES, value="flash", label="Attention Mode")
            mask_attn_mode = gr.Dropdown(
                _MASK_ATTN_MODES, value="block_sparse", label="Mask Attention Mode"
            )

        with gr.Accordion("Quantization", open=False):
            quant_mode = gr.Dropdown(_QUANT_MODES, value="none", label="Quantization Mode")

    with gr.Row():
        run_btn = gr.Button("Run Super Resolution", variant="primary")

    inputs = [
        video_in,
        spatial_enabled,
        spatial_tile_h,
        spatial_tile_w,
        spatial_overlap,
        temporal_enabled,
        temporal_tile_size,
        temporal_overlap,
        attn_mode,
        mask_attn_mode,
        quant_mode,
    ]

    run_btn.click(
        fn=process_video,
        inputs=inputs,
        outputs=[status_box, video_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(OUTPUTS_DIR)],
    )
