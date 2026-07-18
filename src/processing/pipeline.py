"""Pipeline initialization for FlashVSR."""

import os

import torch
from huggingface_hub import snapshot_download
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from src.config.processing import AttentionConfig, QuantizationConfig, QuantizationMode, VRAMConfig
from src.models.model_manager import ModelManager
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Causal_LQ4x_Proj
from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from src.utils.logger import get_logger

logger = get_logger()

_QUANTIZATION_CONFIGS = {
    QuantizationMode.INT8_WEIGHT_ONLY: Int8WeightOnlyConfig(version=2),
    QuantizationMode.INT8_DYNAMIC: Int8DynamicActivationInt8WeightConfig(version=2),
}


def model_download(model_name: str = "JunhaoZhuang/FlashVSR", base_dir: str = "models") -> str:
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])

    if os.path.exists(model_dir):
        logger.info("Model directory found: %s", model_dir)
        return model_dir

    logger.warning("Model directory '%s' not found. Attempting to download...", model_dir)

    try:
        logger.info("Downloading model '%s' from Hugging Face Hub...", model_name)
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
        )
        logger.info("Successfully downloaded model '%s' to '%s'", model_name, model_dir)
        return model_dir
    except Exception as e:
        logger.error("Failed to download model '%s': %s", model_name, e)
        raise RuntimeError(
            f"Could not download model '{model_name}'. Please check your internet connection and Hugging Face Hub access."
        )


def init_pipeline(
    model: str,
    device: torch.device,
    dtype: torch.dtype,
    attention_config: AttentionConfig,
    quantization_config: QuantizationConfig = None,
    vram_config: VRAMConfig = None,
) -> FlashVSRTinyPipeline:
    """Initialize FlashVSR pipeline with given model and device.

    Args:
        model (str): Model name.
        device (torch.device): Device to load the model on.
        dtype (torch.dtype): Data type for model weights.
        attention_config (AttentionConfig): Attention settings for DIT mdoel.
        quantization_config (QuantizationConfig): Quantization settings.
        vram_config (VRAMConfig): VRAM offloading settings (ignored when quantization is active).

    Returns: FlashVSRTinyPipeline: Initialized pipeline instance.

    Raises:
        RuntimeError: If model directory or required files do not exist.
    """
    if quantization_config is None:
        quantization_config = QuantizationConfig()
    if vram_config is None:
        vram_config = VRAMConfig()
    base_dir = "models"
    model_path = model_download(model_name="JunhaoZhuang/" + model, base_dir=base_dir)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f'Model directory does not exist!\nPlease save all weights to "{model_path}"'
        )
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"'
        )
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    prompt_path = os.path.join(base_dir, "posi_prompt.pth")
    if not os.path.exists(prompt_path):
        raise RuntimeError(f'"posi_prompt.pth" does not exist!\nPlease save it to "{base_dir}"')

    mm = ModelManager(
        torch_dtype=dtype,
        device="cpu",
        attn_mode=attention_config.attn_mode,
        mask_attn_mode=attention_config.mask_attn_mode,
    )
    logger.info(
        "Attention mode: %s | Mask attention mode: %s",
        attention_config.attn_mode.value,
        attention_config.mask_attn_mode.value if attention_config.mask_attn_mode else "none",
    )

    logger.info("Loading DiT weights...")
    mm.load_models([ckpt_path])

    pipe = FlashVSRTinyPipeline.from_model_manager(mm)

    logger.info("Building TCDecoder...")
    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(
        new_channels=multi_scale_channels,
        device="cpu",
        dtype=dtype,
        new_latent_channels=16 + 768,
    )
    pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location="cpu"), strict=False)
    pipe.TCDecoder.clean_mem()

    logger.info("Loading LQ projection...")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        dtype=dtype
    )
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(lq_path, map_location="cpu"), strict=True
    )

    pipe.to(device, dtype=dtype)

    if quantization_config.mode != QuantizationMode.NONE:
        quant_cfg = _QUANTIZATION_CONFIGS[quantization_config.mode]
        logger.info("Quantizing denoising model (%s)...", quantization_config.mode.value)
        quantize_(pipe.dit, quant_cfg)
        logger.info("Quantization complete.")
        if vram_config.enabled:
            logger.warning("VRAM management is not supported with quantization — skipping.")
    elif vram_config.enabled:
        logger.info(
            "Enabling VRAM management (num_persistent_param_in_dit=%s)...",
            vram_config.num_persistent_param_in_dit,
        )
        pipe.enable_vram_management(
            num_persistent_param_in_dit=vram_config.num_persistent_param_in_dit
        )

    logger.info("Initializing cross-attention KV cache...")
    pipe.init_cross_kv(prompt_path=prompt_path)

    logger.info("Pipeline ready.")
    return pipe
