# FlashVSR

Video super-resolution using [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR) with configurable attention modes, sparse masking, and quantization. Built as the engineering base for an engineering thesis (benchmarking and evaluating FlashVSR's performance/quality trade-offs); besides the CLI/API/UI it also includes the dataset, benchmarking, and report-generation tooling used to produce the thesis's results.

## Repository layout

```
main.py                  # CLI entry point: run FlashVSR on a single video
src/                      # library code
├── config/                # dataclasses/enums (attention, quantization, tiling, I/O)
├── models/                # DiT (WanModel), VAE, TCDecoder, ModelManager, LQ projection
├── pipelines/              # FlashVSRTinyPipeline (denoising loop, tiling, color fix)
├── processing/             # pipeline init/model download, video I/O
├── schedulers/             # flow-matching scheduler
├── utils/                  # tiling, tensor/dimension helpers, logging
└── vram_management/         # optional CPU/GPU offloading layers
app/                      # FastAPI job-queue service + Gradio frontend
├── api.py, models.py, ui.py
scripts/                  # dataset prep, batch runs, benchmarking, report generation
├── generate_lq.py, run_dataset.py, frames_to_videos.py, merge_dover_metrics.py,
├── tile_size_sweep.py
└── report/                 # aggregates raw results into thesis tables/figures
    ├── aggregate_performance.py, aggregate_quality.py, common.py
    └── make_crops.py         # standalone tool: extracts qualitative comparison crops
benchmarks/               # quality/performance measurement code + raw results
├── manifest.toml           # links raw result files to the thesis chapters/experiments
├── performance/runner.py, performance/results/
└── quality/runner.py, quality/results/
tests/                    # pytest unit tests for src/utils
inputs/                   # example input videos used in docs/benchmarks
models/                   # downloaded model weights (gitignored, except posi_prompt.pth)
assets/                   # finalized figures/tables committed for the thesis (see below)
report/tables/            # freshly generated .typ tables (output of scripts/report/*)
```

`datasets/` (VideoLQ, YouHQ40) is not tracked in git - see [Datasets](#datasets) to obtain/generate it.

## Requirements

- Linux Python 3.11 (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU with CUDA for `sage`/`sparse_sage`/`block_sparse` attention (built as source extensions - see below)

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
uv run python main.py inputs/example0.mp4
```

The upscaled video is written to `<video_name>_output/` by default. Key options:

| Flag | Default | Description |
|---|---|---|
| `-o, --output-dir` | `<video_name>_output` | Output directory |
| `--output-mode` | `video` | `video`, `frames`, or `none` |
| `--model` | `FlashVSR-v1.1` | Model variant (suffix of the HF repo name) |
| `--device` | autodetect | e.g. `cuda:0`, `mps` |
| `--scale` | `4` | Upscaling factor |
| `--seed` | `0` | Random seed |
| `--attn-mode` | `flash` | `flash` or `sage` |
| `--mask-attn-mode` | `block_sparse` | `block_sparse`, `sparse_sage`, or `none` |
| `--quantization` | `none` | `none`, `int8_weight_only`, `int8_dynamic` |
| `--no-spatial-tiling` | enabled | Disable spatial tiling (⚠️ needs much more VRAM) |
| `--spatial-tile-size H W` | `192 192` | Spatial tile size |
| `--spatial-tile-overlap` | `24` | Spatial tile overlap |
| `--temporal-tiling` | disabled | Enable temporal tiling (for long videos) |
| `--temporal-tile-size` | `100` | Temporal tile size (frames) |
| `--temporal-tile-overlap` | `6` | Temporal tile overlap (frames) |
| `--sparse-ratio` | `2.0` | Sparse attention ratio |
| `--kv-ratio` | `3.0` | Key/value ratio |
| `--local-range` | `11` | Local attention range |
| `--no-color-fix` | enabled | Disable wavelet color fixing |

Example combining sage attention, sparse-sage masking and int8 dynamic quantization:
```bash
uv run python main.py inputs/example0.mp4 \
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

- **VideoLQ** (real-world LQ clips, no ground truth) is from [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) - see the "VideoLQ Dataset" section of its README for the Dropbox/Google Drive/OneDrive download links. Extract it into `datasets/VideoLQ/LQ/<clip>/`.
- **YouHQ40** (ground-truth HQ clips, LQ generated locally) is the YouHQ40-Test split from [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) - see its README's dataset section for the Google Drive link. Extract it into `datasets/YouHQ40/HQ/<clip>/`, then generate `LQ/` as below.

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

**SR** - flash attention, block-sparse mask, no quantization, no spatial tiling (⚠️ high VRAM - run on a machine with sufficient memory):
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR \
    --no-spatial-tiling
```

**SR\_flash\_bs\_none** - flash attention, block-sparse mask, no quantization:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_flash_bs_none
```

**SR\_sage\_bs\_none** - sage attention, block-sparse mask:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_bs_none \
    --attn-mode sage --mask-attn-mode block_sparse
```

**SR\_sage\_ss\_none** - sage attention, sparse-sage mask:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_ss_none \
    --attn-mode sage --mask-attn-mode sparse_sage
```

**SR\_sage\_ss\_int8dyn** - sage attention, sparse-sage mask, int8 dynamic quantization:
```bash
uv run python scripts/run_dataset.py \
    --input  datasets/YouHQ40/LQ \
    --output datasets/YouHQ40/SR_sage_ss_int8dyn \
    --attn-mode sage --mask-attn-mode sparse_sage \
    --quantization int8_dynamic
```

Replace `YouHQ40` with `VideoLQ` to run on the other dataset. `run_dataset.py` also accepts `--tile-size`, `--tile-overlap`, `--no-temporal-tiling`, `--temporal-tile-size`, `--temporal-tile-overlap`, `--skip-existing`, `--model`, and `--device` - see `--help`.

## Quality evaluation

### Reference-based + no-reference metrics (PSNR / SSIM / LPIPS / NIQE / MUSIQ / CLIPIQA)

With ground truth (YouHQ40):
```bash
uv run python -m benchmarks.quality.runner \
    --sr  datasets/YouHQ40/SR_flash_bs_none \
    --gt  datasets/YouHQ40/HQ \
    --output benchmarks/quality/results/YouHQ40
```

Without ground truth (VideoLQ - NR metrics only):
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

**Step 1 - convert frame sequences to MP4** (run from this repo):
```bash
uv run python scripts/frames_to_videos.py \
    --input  datasets/YouHQ40/SR_flash_bs_none \
    --output datasets/YouHQ40/SR_flash_bs_none_videos
```

**Step 2 - run DOVER evaluation** (run from `../DOVER/`):
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

`params` accepts a JSON body overriding any field of `JobParams` (processing/attention/quantization/tiling/VRAM config - see `app/models.py`); `{}` uses the defaults.

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

The UI is served at `http://localhost:7860`. Upload a video, adjust the processing/tiling/attention/quantization settings, and click "Run Super Resolution" - the output plays once the job finishes.

## Generating report tables (thesis chapter 8)

`scripts/report/aggregate_performance.py` and `scripts/report/aggregate_quality.py` aggregate raw results (indexed by `benchmarks/manifest.toml`) into Typst tables:

```bash
uv run python -m scripts.report.aggregate_performance   # -> report/tables/*.typ
uv run python -m scripts.report.aggregate_quality        # -> report/tables/*.typ
```

## Generating qualitative comparison crops

`scripts/report/make_crops.py` is a standalone tool (unrelated to the aggregation scripts above) that crops the same region of the same frame out of several pipeline outputs - e.g. LR input, a baseline run, and an optimized run - and saves them as individual PNGs for a side-by-side comparison figure. Sources can be a single image, a directory of frames, or a video file:

```bash
uv run python -m scripts.report.make_crops \
    --source lr=frames/lr_042.png \
    --source baz=frames/baseline_042.png \
    --source opt=frames/optimized_042.png \
    --lr-source lr \
    --crop A=1280,600,400,400 \
    --crop B=568,288,300,300 \
    --out rysunki/porownanie
```

## Thesis assets

- `assets/` holds the finalized figures (`assets/diagrams/`), sample visual results (`assets/visual_results/`), and Typst tables (`assets/tables/`) actually committed for use in the thesis document - treat these as frozen artifacts, not regeneration output.
