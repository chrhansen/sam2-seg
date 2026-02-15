from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from sam2_poc.config import load_settings
from sam2_poc.pipeline import extract_frame0, run_sam2_click_track


@dataclass
class JobState:
    job_id: str
    run_dir: Path
    input_video_path: Path
    source_filename: str
    width: int
    height: int
    fps: float
    num_frames: int
    state: str
    progress: float
    message: str
    error: str | None


class PromptPayload(BaseModel):
    x: int
    y: int


app = FastAPI(title="SAM2 Skier Tracker POC")
settings = load_settings()
settings.runs_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("uvicorn.error")

_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()



def _video_meta(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    return width, height, fps, num_frames



def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)



def _get_job(job_id: str) -> JobState:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job



def _job_payload(job: JobState) -> dict:
    payload = {
        "job_id": job.job_id,
        "state": job.state,
        "progress": round(job.progress, 4),
        "message": job.message,
    }
    if job.error:
        payload["error"] = job.error
    if job.state == "done":
        payload["results"] = {
            "overlay_url": f"/files/{job.job_id}/overlay.mp4",
            "masks_url": f"/files/{job.job_id}/masks.json",
            "frame0_url": f"/files/{job.job_id}/frame0.jpg",
        }
    return payload



def _run_job(job_id: str, click_x: int, click_y: int) -> None:
    job = _get_job(job_id)
    logger.info("[sam2][%s] job start click=(%d,%d) video=%s", job_id, click_x, click_y, job.source_filename)

    def progress_cb(progress: float, message: str) -> None:
        _update_job(job_id, progress=max(0.0, min(1.0, progress)), message=message)

    try:
        video_info, _ = run_sam2_click_track(
            settings=settings,
            video_path=job.input_video_path,
            source_filename=job.source_filename,
            run_dir=job.run_dir,
            click_x=click_x,
            click_y=click_y,
            progress_cb=progress_cb,
        )
        _update_job(
            job_id,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            num_frames=video_info.num_frames,
            state="done",
            progress=1.0,
            message="complete",
            error=None,
        )
        logger.info("[sam2][%s] job complete", job_id)
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            state="failed",
            message="processing failed",
            error=str(exc),
        )
        logger.exception("[sam2][%s] job failed: %s", job_id, exc)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SAM2 Skier Track POC</title>
  <style>
    :root {
      --bg: #0d1117;
      --fg: #e6edf3;
      --muted: #8b949e;
      --card: #161b22;
      --line: #30363d;
      --accent: #d2ff4d;
    }
    body {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: radial-gradient(circle at 20% -5%, #1a2435 0%, var(--bg) 40%);
      color: var(--fg);
      margin: 0;
      padding: 20px;
    }
    .wrap { max-width: 960px; margin: 0 auto; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin: 0 0 16px 0;
    }
    button {
      background: var(--accent);
      color: #111;
      border: 0;
      border-radius: 8px;
      padding: 8px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    input[type=file] { color: var(--fg); }
    .muted { color: var(--muted); }
    #frame0 {
      width: 100%;
      max-width: 820px;
      border: 1px solid var(--line);
      border-radius: 10px;
      cursor: crosshair;
      display: none;
    }
    #links a { color: var(--accent); margin-right: 12px; }
    #overlay {
      width: 100%;
      max-width: 820px;
      border: 1px solid var(--line);
      border-radius: 10px;
      display: none;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SAM2 skier track POC</h1>
    <div class="card">
      <form id="uploadForm">
        <input id="videoInput" type="file" name="video" accept="video/*" required />
        <button id="uploadBtn" type="submit">Upload</button>
      </form>
      <p class="muted">Upload clip, click skier on frame 0, wait for overlay + masks.json.</p>
    </div>

    <div class="card">
      <p id="status">Idle</p>
      <img id="frame0" alt="frame0" />
      <video id="overlay" controls playsinline></video>
      <div id="links"></div>
    </div>
  </div>

  <script>
    let current = null;
    let pollTimer = null;

    const statusEl = document.getElementById('status');
    const frame0El = document.getElementById('frame0');
    const overlayEl = document.getElementById('overlay');
    const linksEl = document.getElementById('links');
    const uploadBtn = document.getElementById('uploadBtn');

    function setStatus(msg) {
      statusEl.textContent = msg;
    }

    function stopPolling() {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    async function pollStatus() {
      if (!current) return;
      const res = await fetch(`/status/${current.job_id}`);
      if (!res.ok) return;
      const s = await res.json();
      const pct = Math.round((s.progress || 0) * 100);
      setStatus(`${s.state} ${pct}% ${s.message || ''}`.trim());

      if (s.state === 'done') {
        stopPolling();
        overlayEl.src = s.results.overlay_url;
        overlayEl.style.display = 'block';
        linksEl.innerHTML = `<a href="${s.results.overlay_url}" download>overlay.mp4</a> <a href="${s.results.masks_url}" download>masks.json</a>`;
      }
      if (s.state === 'failed') {
        stopPolling();
        setStatus(`failed: ${s.error || 'unknown error'}`);
      }
    }

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = document.getElementById('videoInput').files[0];
      if (!file) return;

      overlayEl.style.display = 'none';
      linksEl.innerHTML = '';
      frame0El.style.display = 'none';
      setStatus('uploading...');
      uploadBtn.disabled = true;

      const body = new FormData();
      body.append('video', file);
      const res = await fetch('/upload', { method: 'POST', body });
      uploadBtn.disabled = false;

      if (!res.ok) {
        const t = await res.text();
        setStatus(`upload failed: ${t}`);
        return;
      }

      current = await res.json();
      frame0El.src = current.frame0_url + `?t=${Date.now()}`;
      frame0El.style.display = 'block';
      setStatus('uploaded. click skier on frame0.');
    });

    frame0El.addEventListener('click', async (e) => {
      if (!current) return;

      const rect = frame0El.getBoundingClientRect();
      const xUi = e.clientX - rect.left;
      const yUi = e.clientY - rect.top;
      const x = Math.round((xUi / rect.width) * current.width);
      const y = Math.round((yUi / rect.height) * current.height);

      setStatus(`starting at click (${x}, ${y})`);
      const res = await fetch(`/prompt/${current.job_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y }),
      });

      if (!res.ok) {
        const t = await res.text();
        setStatus(`start failed: ${t}`);
        return;
      }

      stopPolling();
      pollTimer = setInterval(pollStatus, 1000);
      pollStatus();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/upload")
async def upload(video: UploadFile = File(...)) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    run_dir = settings.runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = video.filename or "input.mp4"
    suffix = Path(filename).suffix or ".mp4"
    input_video_path = run_dir / f"input{suffix}"

    with input_video_path.open("wb") as out:
        while chunk := await video.read(1024 * 1024):
            out.write(chunk)

    frame0_path = run_dir / "frame0.jpg"
    try:
        width, height, fps, num_frames = _video_meta(input_video_path)
        extract_frame0(input_video_path, frame0_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"video decode failed: {exc}") from exc

    job = JobState(
        job_id=job_id,
        run_dir=run_dir,
        input_video_path=input_video_path,
        source_filename=filename,
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        state="ready",
        progress=0.0,
        message="uploaded",
        error=None,
    )

    with _jobs_lock:
        _jobs[job_id] = job

    return JSONResponse(
        {
            "job_id": job_id,
            "frame0_url": f"/files/{job_id}/frame0.jpg",
            "width": width,
            "height": height,
        }
    )


@app.post("/prompt/{job_id}")
def prompt(job_id: str, payload: PromptPayload) -> JSONResponse:
    job = _get_job(job_id)

    if job.state == "processing":
        raise HTTPException(status_code=409, detail="job already processing")
    if job.state == "done":
        return JSONResponse({"ok": True, "already_done": True})

    x = max(0, min(payload.x, job.width - 1))
    y = max(0, min(payload.y, job.height - 1))

    _update_job(job_id, state="processing", progress=0.01, message="starting", error=None)
    worker = threading.Thread(target=_run_job, args=(job_id, x, y), daemon=True)
    worker.start()
    return JSONResponse({"ok": True})


@app.get("/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    return JSONResponse(_job_payload(job))


@app.get("/files/{job_id}/{name}")
def files(job_id: str, name: str):
    if name not in {"overlay.mp4", "masks.json", "frame0.jpg"}:
        raise HTTPException(status_code=404, detail="file not found")

    job = _get_job(job_id)
    path = (job.run_dir / name).resolve()
    if not path.exists() or path.parent != job.run_dir.resolve():
        raise HTTPException(status_code=404, detail="file not found")

    media_type = {
        "overlay.mp4": "video/mp4",
        "masks.json": "application/json",
        "frame0.jpg": "image/jpeg",
    }[name]
    return FileResponse(path=path, media_type=media_type, filename=name)
