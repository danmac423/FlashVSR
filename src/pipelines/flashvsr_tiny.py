from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.model_manager import ModelManager
from src.models.TCDecoder import TAEHV
from src.models.wan_video_dit import RMSNorm, WanModel, sinusoidal_embedding_1d
from src.pipelines.base import BasePipeline
from src.schedulers.flow_match import FlowMatchScheduler
from src.utils.logger import get_logger

logger = get_logger()


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, "feat must be (N, C, H, W)"
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2]
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, "x must be (N, C, H, W)"
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    # dilation=radius implements the à-trous trick: receptive field doubles each level
    # without downsampling, so spatial resolution is preserved throughout decomposition
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C)


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    # Stationary (undecimated) wavelet transform via the à-trous algorithm.
    # high accumulates residuals at each scale; low is the coarse approximation.
    assert x.dim() == 4, "x must be (N, C, H, W)"
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2**i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(
    content: torch.Tensor, style: torch.Tensor, levels: int = 5
) -> torch.Tensor:
    # Keep sharp details (high-freq) from HQ content, color/tone (low-freq) from LQ style.
    # This transfers the LQ color statistics without softening edges from the HQ output.
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        assert x.dim() == 5, "input must be (B, C, f, H, W)"
        B, C, f, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * f, C, H, W)
        return y, B, f

    @staticmethod
    def _unflatten_time(y: torch.Tensor, B: int, f: int) -> torch.Tensor:
        BF, C, H, W = y.shape
        assert BF == B * f
        return y.reshape(B, f, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        assert hq_image.shape == lq_image.shape
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3

        B, C, f, H, W = hq_image.shape
        if chunk_size is None or chunk_size >= f:
            hq4, B, f = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == "adain":
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"unknown method: {method}")
            return self._unflatten_time(torch.clamp(out4, *clip_range), B, f)

        outs = []
        for start in range(0, f, chunk_size):
            end = min(start + chunk_size, f)
            hq4, B_, f_ = self._flatten_time(hq_image[:, :, start:end])
            lq4, _, _ = self._flatten_time(lq_image[:, :, start:end])
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == "adain":
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"unknown method: {method}")
            outs.append(self._unflatten_time(torch.clamp(out4, *clip_range), B_, f_))
        return torch.cat(outs, dim=2)


class FlashVSRTinyPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        """Initialize the FlashVSR Tiny pipeline.

        Args:
            device: Target compute device (e.g. ``"cuda"``, ``"cpu"``).
            torch_dtype: Inference precision; ``bfloat16`` is the recommended default.
        """
        super().__init__(device=device, torch_dtype=torch_dtype)
        # shift=5 matches WanVideo's training schedule; extra_one_step extends
        # the trajectory by one so 1-step inference lands at sigma=0 (clean image).
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.TCDecoder: TAEHV = None
        self.model_names = ["dit", "TCDecoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        """Wrap DiT layers for CPU offloading to reduce peak VRAM usage.

        All layers are wrapped for offloading: parameters are stored on CPU and
        moved to the compute device only during their forward pass. Layers within
        the parameter budget use GPU for computation; layers that exceed it fall
        back to CPU-only computation via ``overflow_module_config``.

        Args:
            num_persistent_param_in_dit: Cumulative parameter count threshold
                before switching to CPU-only computation. ``None`` means all
                layers compute on GPU (but are still offloaded to CPU when idle).
        """
        dtype = next(iter(self.dit.parameters())).dtype
        from src.vram_management.layers import (
            AutoWrappedLinear,
            AutoWrappedModule,
            enable_vram_management,
        )

        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        """Load model weights from a ModelManager into this pipeline.

        Args:
            model_manager: Source of pre-loaded model weights.
        """
        self.dit = model_manager.fetch_model("wan_video_dit")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        """Construct a pipeline directly from a ModelManager.

        Convenience factory that calls ``__init__`` and ``fetch_models`` in one
        step, inferring device and dtype from the manager when not supplied.

        Args:
            model_manager: Manager holding pre-loaded model weights.
            torch_dtype: Override the manager's default dtype. ``None`` inherits
                from the manager.
            device: Override the manager's default device. ``None`` inherits
                from the manager.

        Returns:
            A fully initialized ``FlashVSRTinyPipeline`` instance.
        """
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRTinyPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        """Return the underlying DiT denoising model.

        Returns:
            The ``WanModel`` instance used for denoising.
        """
        return self.dit

    def init_cross_kv(
        self,
        context_tensor: torch.Tensor | None = None,
        prompt_path=None,
    ):
        """Pre-compute and cache cross-attention KV pairs from the text context.

        Must be called once before ``__call__``. Loads the context tensor,
        injects it into the DiT via ``reinit_cross_kv``, and pre-computes
        the time embeddings at T=1000 for single-step inference.

        Args:
            context_tensor: Pre-encoded text context of shape
                ``(1, seq_len, dim)``. Takes priority over ``prompt_path``.
            prompt_path: Path to a ``.pt`` file containing a saved context
                tensor. Used only when ``context_tensor`` is ``None``.

        Raises:
            RuntimeError: If ``self.dit`` has not been initialized.
            ValueError: If neither ``context_tensor`` nor ``prompt_path`` is
                provided.
            AttributeError: If ``WanModel`` is missing ``reinit_cross_kv``.
        """
        self.load_models_to_device(["dit"])

        if self.dit is None:
            raise RuntimeError(
                "Please initialize self.dit via fetch_models / from_model_manager first"
            )

        if context_tensor is None:
            if prompt_path is None:
                raise ValueError("init_cross_kv: provide either prompt_path or context_tensor")
            ctx = torch.load(prompt_path, map_location=self.device)
        else:
            ctx = context_tensor

        ctx = ctx.to(dtype=self.torch_dtype, device=self.device)

        if self.prompt_emb_posi is None:
            self.prompt_emb_posi = {}
        self.prompt_emb_posi["context"] = ctx
        self.prompt_emb_posi["stats"] = "load"

        if hasattr(self.dit, "reinit_cross_kv"):
            self.dit.reinit_cross_kv(ctx)
        else:
            raise AttributeError("WanModel is missing reinit_cross_kv(ctx)")

        self.timestep = torch.tensor([1000.0], device=self.device, dtype=self.torch_dtype)
        self.t = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))
        self.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
        self.load_models_to_device([])

    @torch.no_grad()
    def __call__(
        self,
        LQ_video: torch.Tensor,
        seed: int = 0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        is_full_block: bool = False,
        if_buffer: bool = False,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 9,
        color_fix: bool = True,
    ):
        """Run the streaming FlashVSR super-resolution pipeline on an LQ video.

        Denoises latent chunks in a sliding-window fashion using a single-step
        flow-matching update, decodes with TAEHV, and optionally applies AdaIN
        color correction to match the LQ color distribution.

        Args:
            LQ_video: Low-quality input video of shape ``(1, 3, T, H, W)``
                in the range ``[-1, 1]``.
            seed: Random seed for noise generation.
            height: Output height in pixels; snapped to a multiple of 16.
            width: Output width in pixels; snapped to a multiple of 16.
            num_frames: Desired output frame count; rounded up to satisfy
                ``(T - 1) % 4 == 0`` for the temporal VAE.
            is_full_block: Use full (non-windowed) self-attention in the DiT
                blocks when ``True``.
            if_buffer: Omit the anchor latent frame when ``True``; used when
                continuing from a previously decoded streaming window.
            topk_ratio: Multiplier on the sparse cross-window token budget.
                Higher values attend to more context at greater cost.
            kv_ratio: Number of past streaming windows retained in the rolling
                KV cache.
            local_range: Number of local windows attended to without sparsity.
            color_fix: Apply AdaIN color correction to align the output color
                distribution with the LQ input when ``True``.

        Returns:
            Decoded HQ video tensor of shape ``(3, T_out, H, W)`` in ``[-1, 1]``.

        Raises:
            RuntimeError: If ``init_cross_kv`` has not been called beforehand.
        """
        if self.prompt_emb_posi is None or "context" not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV is not initialized. Please call pipe.init_cross_kv() before __call__."
            )

        height, width = self.check_resize_height_width(height, width)
        # WanVideo's temporal VAE compresses time by 4×, so valid counts satisfy (T-1) % 4 == 0.
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            logger.warning("num_frames %% 4 != 1 is required; rounded up to %d.", num_frames)

        # buffer mode omits the +1 anchor frame because the previous window already provides it
        if if_buffer:
            noise = self.generate_noise(
                (1, 16, (num_frames - 1) // 4, height // 8, width // 8),
                seed=seed,
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            noise = self.generate_noise(
                (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                seed=seed,
                device=self.device,
                dtype=self.torch_dtype,
            )

        latents = noise
        # Each streaming step advances 8 frames (2 latent tokens); -2 reserves
        # the warmup window, so this is the number of incremental decode steps.
        process_total_num = (num_frames - 1) // 8 - 2

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()

        latents_total = []
        LQ_cur_idx = 0

        logger.debug(
            "DiT inference: %d steps, %dx%d, %d frames",
            process_total_num,
            width,
            height,
            num_frames,
        )
        self.load_models_to_device(["dit"])
        with torch.no_grad():
            for cur_process_idx in tqdm(range(process_total_num), desc="DiT", leave=False):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    # Warmup: stream-encode 7 LQ temporal chunks to fill the initial
                    # KV context before any output latents are produced.
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        # max(0, ...) clamps the first slice so we don't index negatively;
                        # the -3 offset creates a 3-frame overlap for temporal context.
                        cur = (
                            self.denoising_model().LQ_proj_in.stream_forward(
                                LQ_video[
                                    :, :, max(0, inner_idx * 4 - 3) : (inner_idx + 1) * 4 - 3, :, :
                                ]
                            )
                            if LQ_video is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            # Concatenate along the token/time dimension (dim=1) across blocks
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                            del cur
                    LQ_cur_idx = (inner_loop_num - 1) * 4 - 3
                    # First output chunk covers the first 6 latent frames (warmup window)
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    # Each incremental step encodes 2 new LQ chunks (8 new frames each)
                    # to match the 2-latent-frame output window produced per step.
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        # Absolute LQ frame offset: 17 = warmup offset (7 chunks × 4 - 11 overlap)
                        cur = (
                            self.denoising_model().LQ_proj_in.stream_forward(
                                LQ_video[
                                    :,
                                    :,
                                    cur_process_idx * 8 + 17 + inner_idx * 4 : cur_process_idx * 8
                                    + 21
                                    + inner_idx * 4,
                                    :,
                                    :,
                                ]
                            )
                            if LQ_video is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                            del cur
                    LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    # Slide the 2-latent window forward by 2 each step (4 frames in pixel space)
                    cur_latents = latents[
                        :, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2, :, :
                    ]

                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=True,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range=local_range,
                )

                # Single-step flow matching update: x_0 ≈ x_T - v(x_T, T)
                # Valid because the scheduler runs one step at maximum noise (T=1000).
                cur_latents = cur_latents - noise_pred_posi
                latents_total.append(cur_latents)

            del cur_latents, pre_cache_k, pre_cache_v, LQ_latents
            latents = torch.cat(latents_total, dim=2)
            del latents_total

            logger.debug("TCDecoder: decoding %d latent frames", latents.shape[2])
            self.load_models_to_device(["TCDecoder"])
            self.TCDecoder.clean_mem()
            # TAEHV outputs pixels in [0, 1]; rescale to [-1, 1] for downstream consistency.
            frames = (
                self.TCDecoder.decode_video(
                    latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=True,
                    cond=LQ_video[:, :, :LQ_cur_idx, :, :],
                )
                .transpose(1, 2)
                .mul_(2)
                .sub_(1)
            )
            del latents
            self.load_models_to_device([])

            try:
                if color_fix:
                    frames = self.ColorCorrector(
                        frames.to(device=LQ_video.device),
                        LQ_video[:, :, : frames.shape[2], :, :],
                        clip_range=(-1, 1),
                        chunk_size=16,
                        method="adain",
                    )
            except:  # noqa: E722
                pass

        return frames[0]


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    LQ_latents: torch.Tensor | None = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: list[torch.Tensor] | None = None,
    pre_cache_v: list[torch.Tensor] | None = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod: torch.Tensor = None,
    t: torch.Tensor = None,
    local_range: int = 9,
):
    """Run one forward pass of the WanVideo DiT for a single streaming chunk.

    Patchifies the latent chunk, constructs absolute-position RoPE frequencies,
    injects per-block LQ features, propagates through all transformer blocks
    with the rolling KV cache, and unpatchifies the output velocity field.

    Args:
        dit: The ``WanModel`` transformer to evaluate.
        x: Noisy latent chunk of shape ``(B, 16, f, h, w)``.
        timestep: Scalar diffusion timestep tensor, typically ``[1000.0]``.
        context: Cross-attention context tensor. Pass ``None`` when KV pairs
            are pre-cached via ``reinit_cross_kv``.
        LQ_latents: Per-block LQ feature maps from ``LQ_proj_in``, a list of
            length ``num_blocks`` where each entry has shape ``(B, T, dim)``.
        is_full_block: Use full (non-windowed) self-attention when ``True``.
        is_stream: Enable streaming KV-cache accumulation across chunks.
        pre_cache_k: Per-block cached key tensors from the previous chunk,
            length ``num_blocks``. ``None`` entries indicate a cold start.
        pre_cache_v: Per-block cached value tensors from the previous chunk,
            length ``num_blocks``.
        topk_ratio: Ratio controlling the sparse cross-window token budget.
        kv_ratio: Number of past windows retained in the rolling KV cache.
        cur_process_idx: Index of the current streaming step (0 = warmup).
        t_mod: Pre-computed time modulation tensor of shape ``(1, 6, dim)``,
            cached from ``init_cross_kv``.
        t: Pre-computed time embedding of shape ``(1, dim)``,
            cached from ``init_cross_kv``.
        local_range: Number of local windows included without sparse selection.

    Returns:
        A tuple ``(x, pre_cache_k, pre_cache_v)`` where ``x`` is the predicted
        velocity field of shape ``(B, 16, f, h, w)``, and ``pre_cache_k`` /
        ``pre_cache_v`` are the updated per-block KV caches.
    """
    x, (f, h, w) = dit.patchify(x)

    # (temporal, height, width) window shape for partitioned attention
    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    # Number of spatial windows per side; square_num is total spatial windows in a temporal slice
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    # Top-k token budget for sparse cross-window attention; -1 excludes the self-window
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    # Temporal RoPE frequencies must reflect the absolute position in the full video,
    # not the local chunk index, so that cross-window attention is positionally consistent.
    if cur_process_idx == 0:
        # Warmup chunk starts at temporal position 0
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
    else:
        # +4 skips the warmup window (6 latent frames); each subsequent step advances by 2
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][4 + cur_process_idx * 2 : 4 + cur_process_idx * 2 + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

    for block_id, block in enumerate(dit.blocks):
        # Multi-scale LQ injection: each transformer block receives its own LQ feature map,
        # extracted by LQ_proj_in at the matching depth, rather than only at the input.
        if LQ_latents is not None and block_id < len(LQ_latents):
            x = x + LQ_latents[block_id]
        # pre_cache_k/v carry the KV state from the previous streaming window;
        # each block's cache is updated in-place after the forward pass.
        x, last_pre_cache_k, last_pre_cache_v = block(
            x,
            context,
            t_mod,
            freqs,
            f,
            h,
            w,
            local_num,
            topk,
            block_id=block_id,
            kv_len=kv_len,
            is_full_block=is_full_block,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
            pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
            local_range=local_range,
        )
        if pre_cache_k is not None:
            pre_cache_k[block_id] = last_pre_cache_k
        if pre_cache_v is not None:
            pre_cache_v[block_id] = last_pre_cache_v

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x, pre_cache_k, pre_cache_v
