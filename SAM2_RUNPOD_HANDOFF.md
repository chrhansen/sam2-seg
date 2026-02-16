# SAM2 RunPod Handoff (Tiny Model, No UI)

Purpose: give another coding agent a minimal baseline for integrating SAM2 into `poser.pro` production backend.

Scope:
- Run on RunPod GPU (A40 class, or similar).
- No browser UI.
- HTTP API only.
- Input: video metadata + click coordinates.
- Output: per-frame segmentation masks in COCO RLE.
- Defaults hard-coded to `sam2.1-hiera-tiny` + CUDA + VOS optimized.

## What We Validated
- SAM2 works reliably on RunPod A40 with CUDA.
- On Mac M1, MPS works but much slower than A40.
- For speed-sensitive tracking, tiny model is practical for short clips, 15-30 seconds.
- VOS optimized mode should be enabled for better video throughput.

## File 1: `Dockerfile` (minimal, RunPod-oriented)
Use this as the container image for the service. No runtime env vars required.

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SAM2_DEVICE=cuda \
    SAM2_HF_MODEL_ID=facebook/sam2.1-hiera-tiny \
    SAM2_VOS_OPTIMIZED=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    git build-essential ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 torchvision==0.20.1 && \
    python -m pip install -r /app/requirements.txt && \
    SAM2_BUILD_ALLOW_ERRORS=1 python -m pip install -v git+https://github.com/facebookresearch/sam2.git

COPY service.py /app/service.py

EXPOSE 8000
CMD ["bash", "-lc", "uvicorn service:app --host 0.0.0.0 --port ${PORT}"]
```

## File 2: `requirements.txt`
Keep this minimal.

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
numpy>=1.26.0
opencv-python>=4.10.0
pycocotools>=2.0.8
```

## File 3: `service.py` (thin API, no UI)
This endpoint processes one video request and returns per-frame RLE masks directly.

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pycocotools import mask as mask_utils
from sam2.sam2_video_predictor import SAM2VideoPredictor

app = FastAPI(title="sam2-runpod-service")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/sam2.1-hiera-tiny"
VOS_OPTIMIZED = True

predictor = SAM2VideoPredictor.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    vos_optimized=VOS_OPTIMIZED,
)


class TrackRequest(BaseModel):
    # Expected path visible from inside container (e.g. mounted /workspace/clip.mp4)
    video_path: str = Field(..., description="Container-visible input video path")
    click_x: int
    click_y: int
    object_id: int = 1
    max_frames: int | None = Field(default=None, ge=1)


def _decode_video_to_jpegs(video_path: Path, out_dir: Path, max_frames: int | None) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    idx = 0
    width = 0
    height = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx == 0:
            height, width = frame.shape[:2]
        cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
        if max_frames and idx >= max_frames:
            break

    cap.release()
    if idx == 0:
        raise RuntimeError("No frames decoded from input video")
    return width, height, idx


def _to_obj_ids_list(out_obj_ids) -> list[int]:
    if hasattr(out_obj_ids, "detach"):
        return [int(v) for v in out_obj_ids.detach().cpu().tolist()]
    return [int(v) for v in out_obj_ids]


def _mask_from_logits(out_obj_ids, out_mask_logits, target_obj_id: int) -> np.ndarray:
    ids = _to_obj_ids_list(out_obj_ids)
    obj_idx = ids.index(target_obj_id) if target_obj_id in ids else 0
    logits = out_mask_logits[obj_idx]
    while getattr(logits, "ndim", 0) > 2:
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    return (logits > 0.0).to(dtype=torch.uint8).detach().cpu().numpy()


def _encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {
        "encoding": "coco_rle",
        "size_hw": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, "device": DEVICE, "model_id": MODEL_ID, "vos_optimized": VOS_OPTIMIZED}


@app.post("/track")
def track(req: TrackRequest) -> dict[str, object]:
    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"video_path not found: {video_path}")

    with tempfile.TemporaryDirectory(prefix="sam2-track-") as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        width, height, num_frames = _decode_video_to_jpegs(video_path, frames_dir, req.max_frames)
        masks: list[np.ndarray | None] = [None] * num_frames

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(frames_dir))
            points = np.array([[req.click_x, req.click_y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=req.object_id,
                points=points,
                labels=labels,
                normalize_coords=True,
            )
            masks[int(out_frame_idx)] = _mask_from_logits(out_obj_ids, out_mask_logits, req.object_id)

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                masks[int(out_frame_idx)] = _mask_from_logits(out_obj_ids, out_mask_logits, req.object_id)

            predictor.reset_state(state)

        frames_out: list[dict[str, object]] = []
        for idx, mask in enumerate(masks):
            if mask is None or mask.max() == 0:
                frames_out.append({"index": idx, "present": False, "mask": None})
            else:
                frames_out.append({"index": idx, "present": True, "mask": _encode_coco_rle(mask)})

        return {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frames": frames_out,
        }
```

## API Contract (example)
- `POST /track`
- Body:

```json
{
  "video_path": "/workspace/input.mp4",
  "click_x": 585,
  "click_y": 174,
  "object_id": 1
}
```

- Response:
  - `frames[i].mask` is COCO RLE (`encoding`, `size_hw`, `counts`)
  - `frames[i].present=false` when no mask for frame

## RunPod Wiring
- Expose HTTP port `8000`.
- Mount volume at `/workspace` (or any path you use in `video_path`).
- No required runtime env vars for model/device defaults.

## Integration Notes for `poser.pro`
- Keep this as isolated segmentation microservice.
- Caller flow:
  1. Save/resolve video clip path accessible by the pod.
  2. Send click coordinates from your upstream interaction flow.
  3. Consume `frames[]` RLE output.
  4. Convert RLE to internal track/mask format used by `poser.pro`.
