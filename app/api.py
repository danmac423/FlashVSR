import asyncio
import functools
import subprocess
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.models import Job, JobParams, JobStatus
from src.config.processing import (
    AttentionConfig,
    IOConfig,
    OutputMode,
    QuantizationConfig,
)
from src.processing.pipeline import init_pipeline
from src.processing.video import flashvsr

MODEL_NAME = "FlashVSR-v1.1"
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"


def _concat_chunks(output_dir: Path) -> Path:
    chunks = sorted(output_dir.glob("chunk_*.mp4"))
    if not chunks:
        raise FileNotFoundError(f"No output chunks in {output_dir}")
    final = output_dir / "output.mp4"
    if len(chunks) == 1:
        chunks[0].rename(final)
        return final
    list_file = output_dir / "concat.txt"
    list_file.write_text("\n".join(f"file '{c.resolve()}'" for c in chunks))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(final)],
        check=True,
        capture_output=True,
    )
    list_file.unlink()
    for c in chunks:
        c.unlink()
    return final


async def _worker(queue: asyncio.Queue, jobs: dict[str, Job]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipe = None
    current_attn_cfg: AttentionConfig | None = None
    current_quant_cfg: QuantizationConfig | None = None

    while True:
        job_id = await queue.get()
        job = jobs[job_id]
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now()

        try:
            attn_cfg = job.params.attention_config
            quant_cfg = job.params.quantization_config
            vram_cfg = job.params.vram_config

            if pipe is None or attn_cfg != current_attn_cfg or quant_cfg != current_quant_cfg:
                pipe = await asyncio.to_thread(
                    functools.partial(
                        init_pipeline,
                        MODEL_NAME,
                        device,
                        dtype,
                        attention_config=attn_cfg,
                        quantization_config=quant_cfg,
                        vram_config=vram_cfg,
                    )
                )
                current_attn_cfg = attn_cfg
                current_quant_cfg = quant_cfg

            output_dir = OUTPUTS_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(
                functools.partial(
                    flashvsr,
                    pipe=pipe,
                    io_config=IOConfig(
                        input_path=job.input_path,
                        output_mode=OutputMode.VIDEO,
                        output_dir=str(output_dir),
                    ),
                    proc_config=job.params.proc_config,
                    spatial_tiling_config=job.params.spatial_tiling_config,
                    temporal_tiling_config=job.params.temporal_tiling_config,
                )
            )

            output_path = _concat_chunks(output_dir)
            job.status = JobStatus.DONE
            job.output_path = str(output_path)

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error = str(e)

        job.updated_at = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs: dict[str, Job] = {}
    queue: asyncio.Queue = asyncio.Queue()

    app.state.jobs = jobs
    app.state.queue = queue
    app.state.worker_task = asyncio.create_task(_worker(queue, jobs))

    yield

    app.state.worker_task.cancel()
    try:
        await app.state.worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="FlashVSR API", lifespan=lifespan)


def _get_jobs(request: Request) -> dict[str, Job]:
    return request.app.state.jobs


def _get_queue(request: Request) -> asyncio.Queue:
    return request.app.state.queue


@app.post("/jobs", status_code=202)
async def submit_job(
    file: UploadFile,
    params: str = Form(default="{}"),
    jobs: dict = Depends(_get_jobs),
    queue: asyncio.Queue = Depends(_get_queue),
) -> dict:
    job_params = JobParams.model_validate_json(params)
    job_id = uuid.uuid4().hex

    input_dir = UPLOADS_DIR / job_id
    input_dir.mkdir(parents=True)
    input_path = input_dir / (file.filename or "input.mp4")

    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    now = datetime.now()
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        params=job_params,
        input_path=str(input_path),
        created_at=now,
        updated_at=now,
    )
    jobs[job_id] = job
    await queue.put(job_id)

    return {"job_id": job_id}


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, jobs: dict = Depends(_get_jobs)) -> Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/download")
async def download_job(job_id: str, jobs: dict = Depends(_get_jobs)) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}")
    return FileResponse(job.output_path, media_type="video/mp4", filename="output.mp4")


@app.get("/jobs", response_model=list[Job])
async def list_jobs(jobs: dict = Depends(_get_jobs)) -> list[Job]:
    return list(jobs.values())
