# FlashVSR 

Video super-resolution using FlashVSR with configurable attention modes, sparse masking, and quantization.

## Setup

```bash
uv sync --all-groups
```

If the build fails due to compiler issues or CUDA kernel compilation errors, the following environment variables were used during development (tested on Ampere GPUs, compute capability 8.0 / 8.6):

```bash
CC=gcc-11 CXX=g++-11 NVCC_CCBIN=gcc-11 \
MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST="8.0;8.6" \
BLOCK_SPARSE_ATTN_CUDA_ARCHS="8.0;8.6" \
uv sync --all-groups
```

## Quick start: process a single video

`main.py` runs FlashVSR end-to-end on one video file. The model weights are downloaded automatically from the [JunhaoZhuang/FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR) Hugging Face repo on first run (cached under `models/`):

```bash
uv run python main.py inputs/example.mp4
```

The upscaled video is written to `<video_name>_output/` by default. Key options:

| Flag | Default | Description |
|---|---|---|
| `-o, --output-dir` | `<video_name>_output` | Output directory |
| `--output-mode` | `video` | `video`, `frames`, or `none` |
| `--model` | `FlashVSR-v1.1` | Model variant (suffix of the HF repo name) |
| `--device` | autodetect | e.g. `cuda:0`, `mps` |
| `--scale` | `4` | Upscaling factor |
| `--attn-mode` | `flash` | `flash` or `sage` |
| `--mask-attn-mode` | `block_sparse` | `block_sparse`, `sparse_sage`, or `none` |
| `--quantization` | `none` | `none`, `int8_weight_only`, `int8_dynamic` |
| `--no-spatial-tiling` | enabled | Disable spatial tiling (⚠️ needs much more VRAM) |
| `--spatial-tile-size H W` | `192 192` | Spatial tile size |
| `--temporal-tiling` | disabled | Enable temporal tiling (for long videos) |

Example combining sage attention, sparse-sage masking and int8 dynamic quantization:
```bash
uv run python main.py inputs/example.mp4 \
    --attn-mode sage --mask-attn-mode sparse_sage \
    --quantization int8_dynamic
```

Run `uv run python main.py --help` for the full list of flags.

## Datasets

```
datasets/
├── VideoLQ/
│   └── LQ/          # LQ frames (external source, no GT)
└── YouHQ40/
    ├── HQ/          # ground truth frames
    └── LQ/          # generated from HQ (see below)
```

### Generate LQ frames for YouHQ40

```bash
uv run python scripts/generate_lq.py \
    datasets/YouHQ40/HQ \
    datasets/YouHQ40/LQ \
    --multi-video --seed 42
```

## Running FlashVSR on a dataset

`scripts/run_dataset.py` runs FlashVSR over every clip subdirectory in a dataset (batch equivalent of `main.py`). All variants write SR frames to `datasets/<dataset>/<variant>/`.

### Variants

**SR** — flash attention, block-sparse mask, no quantization, no spatial tiling (⚠️ high VRAM — run on a machine with sufficient memory):
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR \
    --no-spatial-tiling
```

**SR\_flash\_bs\_none** — flash attention, block-sparse mask, no quantization:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_flash_bs_none
```

**SR\_sage\_bs\_none** — sage attention, block-sparse mask:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_bs_none \
    --attn-mode sage --mask-attn-mode block_sparse
```

**SR\_sage\_ss\_none** — sage attention, sparse-sage mask:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_ss_none \
    --attn-mode sage --mask-attn-mode sparse_sage
```

**SR\_sage\_ss\_int8dyn** — sage attention, sparse-sage mask, int8 dynamic quantization:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_ss_int8dyn \
    --attn-mode sage --mask-attn-mode sparse_sage \
    --quantization int8_dynamic
```

Replace `YouHQ40` with `VideoLQ` to run on the other dataset.

## Quality evaluation

### Reference-based + no-reference metrics (PSNR / SSIM / LPIPS / NIQE / MUSIQ / CLIPIQA)

With ground truth (YouHQ40):
```bash
uv run python -m benchmarks.quality.runner \
    --sr  datasets/YouHQ40/SR_flash_bs_none \
    --gt  datasets/YouHQ40/HQ \
    --output benchmarks/quality/results/YouHQ40
```

Without ground truth (VideoLQ — NR metrics only):
```bash
uv run python -m benchmarks.quality.runner \
    --sr  datasets/VideoLQ/SR_flash_bs_none \
    --output benchmarks/quality/results/VideoLQ
```

Results: `benchmarks/quality/results/<dataset>/<variant>.{csv,json}`

### DOVER (separate environment)

DOVER requires PyTorch < 2 so it runs in its own venv at `../DOVER/`.

**Setup** (once):
```bash
cd ..
git clone https://github.com/VQAssessment/DOVER.git
cd DOVER
python -m venv .venv
source .venv/bin/activate
pip install -e .
deactivate
cd ../FlashVSR
```

**Step 1 — convert frame sequences to MP4** (run from this repo):
```bash
uv run python scripts/frames_to_videos.py \
    --input  datasets/YouHQ40/SR_flash_bs_none \
    --output datasets/YouHQ40/SR_flash_bs_none_videos
```

**Step 2 — run DOVER evaluation** (run from `../DOVER/`):
```bash
cd ../DOVER
source .venv/bin/activate

python evaluate_a_set_of_videos.py \
    --input_video_dir ../FlashVSR/datasets/YouHQ40/SR_flash_bs_none_videos \
    --output_result_csv ../FlashVSR/benchmarks/quality/results/YouHQ40/SR_flash_bs_none_dover.csv

deactivate
cd ../FlashVSR
```

### Merge all metrics into one CSV

```bash
uv run python scripts/merge_dover_metrics.py \
    --metrics benchmarks/quality/results/YouHQ40/SR_flash_bs_none.csv \
    --dover   benchmarks/quality/results/YouHQ40/SR_flash_bs_none_dover.csv \
    --output  benchmarks/quality/results/YouHQ40/SR_flash_bs_none_merged.csv
```

Output columns: `clip, num_frames, psnr, ssim, lpips, niqe, musiq, clipiqa, dover`

## Performance benchmarking

`benchmarks/performance/runner.py` measures wall-clock time and peak VRAM for one or more attention/mask/quantization combinations on a single video:

```bash
# sweep over the default set of combinations
uv run python -m benchmarks.performance.runner --video inputs/example0.mp4

# a single combination, with custom warmup/measured run counts
uv run python -m benchmarks.performance.runner \
    --video inputs/example0.mp4 \
    --attn-mode sage --mask-attn-mode sparse_sage --quantization int8_dynamic \
    --warmup-runs 0 --measured-runs 5

# disable spatial/temporal tiling, override tile/input size
uv run python -m benchmarks.performance.runner \
    --video inputs/example0.mp4 --no-spatial-tiling
```

Results: `benchmarks/performance/results/*.{csv,json}`.

### Spatial tile size sweep

`scripts/tile_size_sweep.py` sweeps spatial tile size (flash attention, block-sparse mask, no quantization) and records time/VRAM per size, recording OOMs instead of aborting the sweep:

```bash
uv run python scripts/tile_size_sweep.py \
    --video inputs/example4.mp4 \
    --tile-sizes 128 160 192 224 256 \
    --tile-overlap 24 \
    --measured-runs 5
```

Results: `benchmarks/performance/results/tile_size_sweep/*.{csv,json}`.

## Running the API

The `app/api.py` module exposes a FastAPI service for submitting videos and retrieving upscaled results asynchronously via a job queue.

Start the server:
```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
    -F "file=@input.mp4" \
    -F 'params={}'
```

Response: `{"job_id": "<id>"}`

### Check job status

```bash
curl http://localhost:8000/jobs/<job_id>
```

### List all jobs

```bash
curl http://localhost:8000/jobs
```

### Download the result (once status is `DONE`)

```bash
curl -OJ http://localhost:8000/jobs/<job_id>/download
```

## Running the frontend

`app/ui.py` is a Gradio UI that talks to the API above (`http://localhost:8000` by default), so make sure the API server is running first.

```bash
uv run python -m app.ui
```

The UI is served at `http://localhost:7860`. Upload a video, adjust the processing/tiling/attention/quantization settings, and click "Run Super Resolution" — the output plays once the job finishes.

## Development

```bash
make lint      # ruff check
make format    # ruff format
make test      # pytest
make clean     # remove __pycache__ and egg-info
```
