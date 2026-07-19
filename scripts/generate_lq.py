#!/usr/bin/env python3
"""
Generate LQ frames from HQ frames using the RealBasicVSR degradation pipeline.

Replicates the train_pipeline from:
  realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py

Usage:
  # Single video (flat directory of frames):
  python scripts/generate_lq.py path/to/hq_frames path/to/lq_frames

  # Dataset with multiple videos (one subdir per video):
  python scripts/generate_lq.py path/to/hq_root path/to/lq_root --multi-video

  # Reproducible output:
  python scripts/generate_lq.py hq/ lq/ --seed 42
"""

import argparse
import io
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.logger import get_logger, setup_logger

logger = get_logger()

# ── Kernel generators ──────────────────────────────────────────────────────────

def _grid(ks: int):
    half = ks // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    return torch.meshgrid(x, x, indexing="ij")  # Y, X


def _norm(k: torch.Tensor) -> torch.Tensor:
    return k / k.sum()


def iso_gaussian(ks: int, sigma: float) -> torch.Tensor:
    Y, X = _grid(ks)
    return _norm(torch.exp(-(X**2 + Y**2) / (2 * sigma**2)))


def aniso_gaussian(ks: int, sx: float, sy: float, angle: float) -> torch.Tensor:
    Y, X = _grid(ks)
    ca, sa = math.cos(angle), math.sin(angle)
    Xr, Yr = ca * X - sa * Y, sa * X + ca * Y
    return _norm(torch.exp(-(Xr**2 / (2 * sx**2) + Yr**2 / (2 * sy**2))))


def gen_iso_gaussian(ks: int, sigma: float, beta: float) -> torch.Tensor:
    Y, X = _grid(ks)
    return _norm(torch.exp(-((X**2 + Y**2) / (2 * sigma**2)) ** beta))


def gen_aniso_gaussian(ks: int, sx: float, sy: float, angle: float, beta: float) -> torch.Tensor:
    Y, X = _grid(ks)
    ca, sa = math.cos(angle), math.sin(angle)
    Xr, Yr = ca * X - sa * Y, sa * X + ca * Y
    return _norm(torch.exp(-((Xr**2 / sx**2 + Yr**2 / sy**2) / 2) ** beta))


def plateau_iso(ks: int, sigma: float, beta: float) -> torch.Tensor:
    Y, X = _grid(ks)
    return _norm(1.0 / (1.0 + ((X**2 + Y**2) / (2 * sigma**2)) ** beta))


def plateau_aniso(ks: int, sx: float, sy: float, angle: float, beta: float) -> torch.Tensor:
    Y, X = _grid(ks)
    ca, sa = math.cos(angle), math.sin(angle)
    Xr, Yr = ca * X - sa * Y, sa * X + ca * Y
    return _norm(1.0 / (1.0 + ((Xr**2 / sx**2 + Yr**2 / sy**2) / 2) ** beta))


def sinc_kernel(ks: int, omega: float) -> torch.Tensor:
    """2-D isotropic sinc (circular low-pass) with Hamming window."""
    Y, X = _grid(ks)
    r = (X**2 + Y**2).sqrt()
    k = torch.zeros(ks, ks)
    m = r > 0
    k[m] = omega * torch.special.bessel_j1(omega * r[m]) / (2 * math.pi * r[m])
    k[~m] = omega**2 / (4 * math.pi)
    win = torch.hamming_window(ks, periodic=False)
    return _norm(k * win.unsqueeze(0) * win.unsqueeze(1))


def sample_kernel(p: dict) -> torch.Tensor:
    ks = random.choice(p["kernel_size"])
    ktype = random.choices(p["kernel_list"], weights=p["kernel_prob"])[0]
    sx = random.uniform(*p["sigma_x"]) if "sigma_x" in p else 1.0
    sy = random.uniform(*p["sigma_y"]) if "sigma_y" in p else sx
    angle = random.uniform(*p["rotate_angle"]) if "rotate_angle" in p else 0.0
    bg = random.uniform(*p["beta_gaussian"]) if "beta_gaussian" in p else 2.0
    bp = random.uniform(*p["beta_plateau"]) if "beta_plateau" in p else 1.5

    if ktype == "iso":
        return iso_gaussian(ks, sx)
    if ktype == "aniso":
        return aniso_gaussian(ks, sx, sy, angle)
    if ktype == "generalized_iso":
        return gen_iso_gaussian(ks, sx, bg)
    if ktype == "generalized_aniso":
        return gen_aniso_gaussian(ks, sx, sy, angle, bg)
    if ktype == "plateau_iso":
        return plateau_iso(ks, sx, bp)
    if ktype == "plateau_aniso":
        return plateau_aniso(ks, sx, sy, angle, bp)
    if ktype == "sinc":
        omega = random.uniform(*p.get("omega", [math.pi / 3, math.pi]))
        return sinc_kernel(ks, omega)
    raise ValueError(f"Unknown kernel type: {ktype}")


# ── Per-frame image transforms ─────────────────────────────────────────────────

def conv_blur(img: np.ndarray, k: torch.Tensor) -> np.ndarray:
    """Apply 2-D separable convolution with kernel k. img: H×W×3 float32 [0,1]."""
    ks = k.shape[0]
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    kf = k.view(1, 1, ks, ks).expand(3, -1, -1, -1).contiguous()
    t = F.pad(t, [ks // 2] * 4, mode="reflect")
    out = F.conv2d(t, kf, groups=3)
    return out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def _resize_frame(img: np.ndarray, tw: int, th: int, interp) -> np.ndarray:
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((tw, th), interp), dtype=np.float32) / 255.0


def _apply_gaussian_noise(img: np.ndarray, sigma: float, gray_prob: float) -> np.ndarray:
    h, w, c = img.shape
    gray = random.random() < gray_prob
    shape = (h, w, 1) if gray else (h, w, c)
    noise = np.random.randn(*shape).astype(np.float32) * sigma
    return img + (np.repeat(noise, c, axis=2) if gray else noise)


def _apply_poisson_noise(img: np.ndarray, scale: float, gray_prob: float) -> np.ndarray:
    c = img.shape[2]
    gray = random.random() < gray_prob
    if gray:
        base = img.mean(axis=2, keepdims=True)
        noisy = np.random.poisson(np.maximum(base * 255 * scale, 0)).astype(np.float32) / (255 * scale)
        return img + np.repeat(noisy - base, c, axis=2)
    else:
        noisy = np.random.poisson(np.maximum(img * 255 * scale, 0)).astype(np.float32) / (255 * scale)
        return noisy


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.float32) / 255.0


# ── Clip-level operations (consistent params across all frames) ────────────────

_PIL_INTERP = {"bilinear": Image.BILINEAR, "area": Image.BOX, "bicubic": Image.BICUBIC}
_CODEC_MAP = {"libx264": "libx264", "h264": "libx264", "mpeg4": "mpeg4"}


def blur_clip(frames: list, params: dict, prob: float = 1.0) -> list:
    if random.random() >= prob:
        return frames
    k = sample_kernel(params)
    return [conv_blur(f, k) for f in frames]


def resize_clip(frames: list, params: dict, target_size: tuple | None = None) -> list:
    h, w = frames[0].shape[:2]
    interp = random.choices(
        list(_PIL_INTERP.values()), weights=[1 / 3] * 3
    )[0]

    if target_size is not None:
        th, tw = target_size
    else:
        mode = random.choices(["up", "down", "keep"], weights=params["resize_mode_prob"])[0]
        if mode == "keep":
            return frames
        lo, hi = params["resize_scale"]
        if mode == "up":
            scale = random.uniform(max(lo, 1.0), hi)
        else:
            scale = random.uniform(lo, min(hi, 1.0))
        th = int(h * scale)
        tw = int(w * scale)
        if params.get("is_size_even"):
            th = max(2, th - th % 2)
            tw = max(2, tw - tw % 2)

    return [_resize_frame(f, tw, th, interp) for f in frames]


def noise_clip(frames: list, params: dict) -> list:
    ntype = random.choices(params["noise_type"], weights=params["noise_prob"])[0]
    if ntype == "gaussian":
        sigma = random.uniform(*params["gaussian_sigma"]) / 255.0
        gray_prob = params["gaussian_gray_noise_prob"]
        return [_apply_gaussian_noise(f, sigma, gray_prob) for f in frames]
    else:
        scale = random.uniform(*params["poisson_scale"])
        gray_prob = params["poisson_gray_noise_prob"]
        return [_apply_poisson_noise(f, scale, gray_prob) for f in frames]


def jpeg_clip(frames: list, quality_range: list) -> list:
    quality = random.randint(*quality_range)
    return [_apply_jpeg(f, quality) for f in frames]


_REF_AREA = 256 * 256  # original config was tuned for 256×256 training patches


def video_compress_clip(frames: list, params: dict) -> list:
    if not frames:
        return frames
    codec_name = random.choices(params["codec"], weights=params["codec_prob"])[0]
    codec = _CODEC_MAP.get(codec_name, codec_name)
    h, w = frames[0].shape[:2]
    # scale bitrate proportionally to frame area so bits-per-pixel stays constant
    area_scale = (h * w) / _REF_AREA
    bitrate = int(random.uniform(*params["bitrate"]) * area_scale)

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        for i, f in enumerate(frames):
            Image.fromarray((f * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(in_dir, f"{i+1:05d}.png")
            )

        encoded = os.path.join(tmp, "vid.mp4")
        ret = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-r", "25", "-i", os.path.join(in_dir, "%05d.png"),
                "-vcodec", codec, "-b:v", str(bitrate),
                "-vf", f"scale={w}:{h}",
                encoded,
            ],
            capture_output=True,
        )
        if ret.returncode != 0:
            # codec not available, skip silently
            return frames

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", encoded,
             os.path.join(out_dir, "%05d.png")],
            capture_output=True,
            check=True,
        )

        result = []
        for i in range(len(frames)):
            p = os.path.join(out_dir, f"{i+1:05d}.png")
            if os.path.exists(p):
                result.append(
                    np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                )
            else:
                result.append(frames[i])
        return result


def unsharp_masking_clip(frames: list, ks: int = 51, sigma: float = 0.0,
                          weight: float = 0.5, threshold: int = 10) -> list:
    if sigma == 0.0:
        # OpenCV formula: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
    k = iso_gaussian(ks, sigma)
    thresh = threshold / 255.0

    def _usm(img):
        blurred = conv_blur(img, k)
        diff = img - blurred
        mask = np.abs(diff) > thresh
        return (img + weight * diff * mask).clip(0, 1)

    return [_usm(f) for f in frames]


def quantize_clip(frames: list) -> list:
    return [(f * 255.0).clip(0, 255).round() / 255.0 for f in frames]


# ── Pipeline constants (from RealBasicVSR config) ─────────────────────────────

_BLUR1 = dict(
    kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
    kernel_list=["iso", "aniso", "generalized_iso", "generalized_aniso",
                 "plateau_iso", "plateau_aniso", "sinc"],
    kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
    sigma_x=[0.2, 3.0], sigma_y=[0.2, 3.0],
    rotate_angle=[-math.pi, math.pi],
    beta_gaussian=[0.5, 4.0], beta_plateau=[1.0, 2.0],
)
_RESIZE1 = dict(resize_mode_prob=[0.2, 0.7, 0.1], resize_scale=[0.15, 1.5], is_size_even=True)
_NOISE1 = dict(
    noise_type=["gaussian", "poisson"], noise_prob=[0.5, 0.5],
    gaussian_sigma=[1, 30], gaussian_gray_noise_prob=0.4,
    poisson_scale=[0.05, 3.0], poisson_gray_noise_prob=0.4,
)
_JPEG1 = [30, 95]
_VIDEO = dict(codec=["libx264", "h264", "mpeg4"], codec_prob=[1/3, 1/3, 1/3], bitrate=[1e4, 1e5])

_BLUR2 = dict(
    kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
    kernel_list=["iso", "aniso", "generalized_iso", "generalized_aniso",
                 "plateau_iso", "plateau_aniso", "sinc"],
    kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
    sigma_x=[0.2, 1.5], sigma_y=[0.2, 1.5],
    rotate_angle=[-math.pi, math.pi],
    beta_gaussian=[0.5, 4.0], beta_plateau=[1.0, 2.0],
)
_RESIZE2 = dict(resize_mode_prob=[0.3, 0.4, 0.3], resize_scale=[0.3, 1.2], is_size_even=True)
_NOISE2 = dict(
    noise_type=["gaussian", "poisson"], noise_prob=[0.5, 0.5],
    gaussian_sigma=[1, 25], gaussian_gray_noise_prob=0.4,
    poisson_scale=[0.05, 2.5], poisson_gray_noise_prob=0.4,
)
_JPEG2 = [30, 95]
_SINC = dict(
    kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
    kernel_list=["sinc"], kernel_prob=[1.0],
    omega=[math.pi / 3, math.pi],
)


def degrade(frames: list, scale: int = 4) -> list:
    """
    Apply the full RealBasicVSR degradation pipeline to a clip.

    Args:
        frames: list of H×W×3 numpy arrays, float32 in [0, 1]
        scale:  downscaling factor (default 4)

    Returns:
        list of (H//scale)×(W//scale)×3 numpy arrays
    """
    h, w = frames[0].shape[:2]
    lq_h, lq_w = h // scale, w // scale
    if lq_h < 1 or lq_w < 1:
        raise ValueError(f"scale={scale} too large for {h}×{w} frames")

    # UnsharpMasking → initial LQ (gt_unsharp)
    lqs = unsharp_masking_clip(frames, ks=51, sigma=0, weight=0.5, threshold=10)

    # ── Round 1 ───────────────────────────────────────────────────────────────
    lqs = blur_clip(lqs, _BLUR1)
    lqs = resize_clip(lqs, _RESIZE1)
    lqs = noise_clip(lqs, _NOISE1)
    lqs = jpeg_clip(lqs, _JPEG1)
    lqs = video_compress_clip(lqs, _VIDEO)

    # ── Round 2 ───────────────────────────────────────────────────────────────
    lqs = blur_clip(lqs, _BLUR2, prob=0.8)
    lqs = resize_clip(lqs, _RESIZE2)
    lqs = noise_clip(lqs, _NOISE2)
    lqs = jpeg_clip(lqs, _JPEG2)

    # ── DegradationsWithShuffle ───────────────────────────────────────────────
    # Two degradation groups applied in random order:
    #   A) video compression
    #   B) resize to target LQ size + sinc blur (prob=0.8)
    def apply_video(fs):
        return video_compress_clip(fs, _VIDEO)

    def apply_resize_sinc(fs):
        fs = resize_clip(fs, {}, target_size=(lq_h, lq_w))
        fs = blur_clip(fs, _SINC, prob=0.8)
        return fs

    if random.random() < 0.5:
        lqs = apply_video(lqs)
        lqs = apply_resize_sinc(lqs)
    else:
        lqs = apply_resize_sinc(lqs)
        lqs = apply_video(lqs)

    # ── Quantize ──────────────────────────────────────────────────────────────
    return quantize_clip(lqs)


# ── I/O helpers ────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_frames(folder: Path) -> tuple[list, list]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image files found in {folder}")
    frames = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
              for p in paths]
    return frames, paths


def save_frames(frames: list, paths: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, f in zip(paths, frames):
        arr = (f * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / p.name)


def degrade_video(frames: list, paths: list, out_dir: Path,
                  scale: int, clip_length: int) -> None:
    n = len(frames)
    n_clips = math.ceil(n / clip_length)
    for clip_idx, start in enumerate(range(0, n, clip_length)):
        clip_frames = frames[start:start + clip_length]
        clip_paths = paths[start:start + clip_length]
        logger.info("  clip %d/%d (frames %d–%d)",
                    clip_idx + 1, n_clips, start + 1, start + len(clip_frames))
        lqs = degrade(clip_frames, scale=scale)
        save_frames(lqs, clip_paths, out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LQ frames using the RealBasicVSR degradation pipeline."
    )
    parser.add_argument("hq_dir", type=Path, help="HQ frames directory")
    parser.add_argument("lq_dir", type=Path, help="Output LQ frames directory")
    parser.add_argument("--scale", type=int, default=4, help="Downscaling factor (default: 4)")
    parser.add_argument(
        "--clip-length", type=int, default=89,
        help="Frames per clip — parameters are re-sampled each clip (default: 89, same as FlashVSR training)"
    )
    parser.add_argument(
        "--multi-video", action="store_true",
        help="Treat hq_dir as a root with one sub-directory per video"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    setup_logger()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.multi_video:
        video_dirs = sorted(d for d in args.hq_dir.iterdir() if d.is_dir())
        if not video_dirs:
            logger.error("No subdirectories found in %s", args.hq_dir)
            return
        logger.info("Found %d videos in %s", len(video_dirs), args.hq_dir)
        for i, vdir in enumerate(video_dirs):
            out_dir = args.lq_dir / vdir.name
            frames, paths = load_frames(vdir)
            logger.info("[%d/%d] %s  (%d frames)", i + 1, len(video_dirs), vdir.name, len(frames))
            degrade_video(frames, paths, out_dir, args.scale, args.clip_length)
        logger.info("Done — %d videos → %s", len(video_dirs), args.lq_dir)
    else:
        frames, paths = load_frames(args.hq_dir)
        logger.info("[1/1] %s  (%d frames)", args.hq_dir.name, len(frames))
        degrade_video(frames, paths, args.lq_dir, args.scale, args.clip_length)
        logger.info("Done — %s", args.lq_dir)


if __name__ == "__main__":
    main()
