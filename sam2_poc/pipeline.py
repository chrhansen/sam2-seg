from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .config import Settings


ProgressCb = Callable[[float, str], None]
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class VideoInfo:
    source_filename: str
    width: int
    height: int
    fps: float
    num_frames: int


@dataclass(frozen=True)
class RunArtifacts:
    frame0_path: Path
    overlay_path: Path
    masks_path: Path


_predictor_cache: dict[tuple[str, str, str, str, bool], object] = {}
_predictor_lock = threading.Lock()


@contextmanager
def _sam_inference_context(device: str):
    if device.startswith("cuda"):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    with torch.inference_mode():
        yield



def pick_device(preferred: str | None) -> str:
    if preferred:
        if preferred == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def _get_predictor(settings: Settings, device: str):
    key = (
        settings.sam2_hf_model_id or "",
        settings.sam2_model_cfg,
        settings.sam2_checkpoint,
        device,
        settings.sam2_vos_optimized,
    )
    with _predictor_lock:
        cached = _predictor_cache.get(key)
        if cached is not None:
            return cached

        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.build_sam import build_sam2_video_predictor
        except Exception as exc:
            raise RuntimeError(
                "SAM2 import failed. Install SAM2 first (pip install -e . in segment-anything-2)."
            ) from exc

        if settings.sam2_hf_model_id:
            predictor = SAM2VideoPredictor.from_pretrained(
                settings.sam2_hf_model_id,
                device=device,
                vos_optimized=settings.sam2_vos_optimized,
            )
        else:
            ckpt_path = Path(settings.sam2_checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {ckpt_path}. Set SAM2_CHECKPOINT or SAM2_HF_MODEL_ID."
                )
            predictor = build_sam2_video_predictor(
                config_file=settings.sam2_model_cfg,
                ckpt_path=str(ckpt_path),
                device=device,
                vos_optimized=settings.sam2_vos_optimized,
            )

        _predictor_cache[key] = predictor
        return predictor



def extract_frame0(video_path: Path, frame0_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to decode first frame")
    frame0_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame0_path), frame)
    h, w = frame.shape[:2]
    return w, h



def decode_frames(
    video_path: Path,
    frames_orig_dir: Path,
    frames_infer_dir: Path,
    resize_wh: tuple[int, int] | None,
) -> tuple[int, int, int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frames_orig_dir.mkdir(parents=True, exist_ok=True)
    frames_infer_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    orig_w = 0
    orig_h = 0
    infer_w = 0
    infer_h = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if idx == 0:
            orig_h, orig_w = frame.shape[:2]

        frame_name = f"{idx:06d}.jpg"
        orig_path = frames_orig_dir / frame_name
        cv2.imwrite(str(orig_path), frame)

        if resize_wh:
            target_w, target_h = resize_wh
            infer_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            infer_frame = frame

        if idx == 0:
            infer_h, infer_w = infer_frame.shape[:2]

        if not (frames_infer_dir == frames_orig_dir and resize_wh is None):
            cv2.imwrite(str(frames_infer_dir / frame_name), infer_frame)

        idx += 1

    cap.release()

    if idx == 0:
        raise RuntimeError("No frames decoded from video")

    return orig_w, orig_h, infer_w, infer_h, float(fps), idx



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

    mask = (logits > 0.0).to(dtype=torch.uint8).detach().cpu().numpy()
    return mask



def _scale_mask_to_original(mask: np.ndarray, original_w: int, original_h: int) -> np.ndarray:
    if mask.shape[1] == original_w and mask.shape[0] == original_h:
        return mask
    resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)



def _bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]



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


def _predictor_device_name(predictor: object) -> str:
    model = getattr(predictor, "model", None)
    if model is None:
        return "unknown"
    try:
        first_param = next(model.parameters())
    except Exception:  # noqa: BLE001
        return "unknown"
    return str(first_param.device)


def _mps_memory_mb() -> float | None:
    if not torch.backends.mps.is_available():
        return None
    try:
        return round(float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0), 2)
    except Exception:  # noqa: BLE001
        return None


def _mask_logits_device_name(out_mask_logits: object) -> str:
    logits = out_mask_logits
    if isinstance(logits, (list, tuple)) and logits:
        logits = logits[0]
    device = getattr(logits, "device", None)
    return str(device) if device is not None else "unknown"



def _overlay_frame(frame_bgr: np.ndarray, mask: np.ndarray | None, bbox: list[int] | None) -> np.ndarray:
    if mask is None or bbox is None:
        return frame_bgr

    out = frame_bgr.copy()
    tint = np.array([20, 220, 20], dtype=np.uint8)
    masked = mask > 0
    out[masked] = (0.55 * out[masked] + 0.45 * tint).astype(np.uint8)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 240), 2)
    return out



def run_sam2_click_track(
    *,
    settings: Settings,
    video_path: Path,
    source_filename: str,
    run_dir: Path,
    click_x: int,
    click_y: int,
    progress_cb: ProgressCb,
) -> tuple[VideoInfo, RunArtifacts]:
    total_start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    frame0_path = run_dir / "frame0.jpg"
    masks_path = run_dir / "masks.json"
    overlay_path = run_dir / "overlay.mp4"
    frames_orig_dir = run_dir / "frames_orig"

    resize_wh: tuple[int, int] | None = None
    if settings.infer_resize_width and settings.infer_resize_height:
        resize_wh = (settings.infer_resize_width, settings.infer_resize_height)
    frames_infer_dir = run_dir / ("frames_infer" if resize_wh else "frames_orig")

    job_ref = run_dir.name
    requested_device = settings.sam2_device or "auto"
    model_ref = settings.sam2_hf_model_id or settings.sam2_checkpoint
    logger.info(
        "[sam2][%s] start source=%s model=%s requested_device=%s mps_available=%s mps_built=%s",
        job_ref,
        source_filename,
        model_ref,
        requested_device,
        torch.backends.mps.is_available(),
        torch.backends.mps.is_built(),
    )

    stage_start = time.perf_counter()
    progress_cb(0.02, "extracting frame0")
    extract_frame0(video_path, frame0_path)
    logger.info("[sam2][%s] frame0 extracted in %.2fs", job_ref, time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    progress_cb(0.08, "decoding frames")
    orig_w, orig_h, infer_w, infer_h, fps, num_frames = decode_frames(
        video_path,
        frames_orig_dir,
        frames_infer_dir,
        resize_wh,
    )
    logger.info(
        "[sam2][%s] decoded frames=%d size=%dx%d fps=%.2f in %.2fs",
        job_ref,
        num_frames,
        orig_w,
        orig_h,
        fps,
        time.perf_counter() - stage_start,
    )

    video_info = VideoInfo(
        source_filename=source_filename,
        width=orig_w,
        height=orig_h,
        fps=fps,
        num_frames=num_frames,
    )

    device = pick_device(settings.sam2_device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        torch.set_float32_matmul_precision("high")
    logger.info(
        "[sam2][%s] resolved_device=%s mps_fallback_env=%s",
        job_ref,
        device,
        os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
    )

    stage_start = time.perf_counter()
    progress_cb(0.15, f"loading sam2 on {device}")
    predictor = _get_predictor(settings, device)
    logger.info(
        "[sam2][%s] predictor loaded in %.2fs predictor_device=%s",
        job_ref,
        time.perf_counter() - stage_start,
        _predictor_device_name(predictor),
    )

    sx = infer_w / float(orig_w)
    sy = infer_h / float(orig_h)
    click_infer = np.array([[click_x * sx, click_y * sy]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    target_obj_id = 1
    masks_infer: list[np.ndarray | None] = [None] * num_frames

    progress_cb(0.2, "running sam2")
    inference_start = time.perf_counter()
    with _sam_inference_context(device):
        init_state_start = time.perf_counter()
        inference_state = predictor.init_state(video_path=str(frames_infer_dir))
        logger.info(
            "[sam2][%s] init_state in %.2fs mps_mem_mb=%s",
            job_ref,
            time.perf_counter() - init_state_start,
            _mps_memory_mb(),
        )

        add_prompt_start = time.perf_counter()
        out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=target_obj_id,
            points=click_infer,
            labels=labels,
            normalize_coords=True,
        )
        logger.info(
            "[sam2][%s] add_prompt frame=%d in %.2fs logits_device=%s mps_mem_mb=%s",
            job_ref,
            int(out_frame_idx),
            time.perf_counter() - add_prompt_start,
            _mask_logits_device_name(out_mask_logits),
            _mps_memory_mb(),
        )
        masks_infer[int(out_frame_idx)] = _mask_from_logits(out_obj_ids, out_mask_logits, target_obj_id)

        tracked_frames = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            frame_idx = int(out_frame_idx)
            masks_infer[frame_idx] = _mask_from_logits(out_obj_ids, out_mask_logits, target_obj_id)
            tracked_frames += 1
            progress = 0.2 + 0.5 * ((frame_idx + 1) / max(1, num_frames))
            progress_cb(min(progress, 0.7), f"tracking frame {frame_idx + 1}/{num_frames}")
            if tracked_frames == 1 or tracked_frames % 100 == 0:
                elapsed = time.perf_counter() - inference_start
                fps_track = tracked_frames / max(elapsed, 1e-6)
                logger.info(
                    "[sam2][%s] propagate frame=%d/%d tracked=%d speed_fps=%.3f logits_device=%s mps_mem_mb=%s",
                    job_ref,
                    frame_idx + 1,
                    num_frames,
                    tracked_frames,
                    fps_track,
                    _mask_logits_device_name(out_mask_logits),
                    _mps_memory_mb(),
                )

        predictor.reset_state(inference_state)
        total_infer_s = time.perf_counter() - inference_start
        logger.info(
            "[sam2][%s] inference complete tracked=%d frames in %.2fs (%.3f fps)",
            job_ref,
            tracked_frames,
            total_infer_s,
            tracked_frames / max(total_infer_s, 1e-6),
        )

    stage_start = time.perf_counter()
    progress_cb(0.75, "building outputs")
    frame_paths = sorted(frames_orig_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (orig_w, orig_h),
    )

    frames_json: list[dict[str, object]] = []
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        mask_infer = masks_infer[idx]
        mask_orig = None
        bbox = None
        if mask_infer is not None:
            mask_orig = _scale_mask_to_original(mask_infer, orig_w, orig_h)
            if mask_orig.max() == 0:
                mask_orig = None
            else:
                bbox = _bbox_xyxy(mask_orig)

        present = mask_orig is not None and bbox is not None
        if present:
            mask_payload = _encode_coco_rle(mask_orig)
            frame_payload = {
                "frame_idx": idx,
                "present": True,
                "bbox_xyxy": bbox,
                "bbox_score": None,
                "mask": mask_payload,
                "track": None,
            }
        else:
            frame_payload = {
                "frame_idx": idx,
                "present": False,
                "bbox_xyxy": None,
                "bbox_score": None,
                "mask": None,
                "track": None,
            }

        frames_json.append(frame_payload)
        writer.write(_overlay_frame(frame, mask_orig, bbox))

    writer.release()
    present_count = sum(1 for frame in frames_json if frame["present"])
    logger.info(
        "[sam2][%s] rendered overlay+frames_json in %.2fs present_frames=%d/%d",
        job_ref,
        time.perf_counter() - stage_start,
        present_count,
        num_frames,
    )

    output = {
        "schema_version": "masktrack.v1",
        "video": {
            "source_filename": source_filename,
            "width": orig_w,
            "height": orig_h,
            "fps": fps,
            "num_frames": num_frames,
        },
        "processing": {
            "method": "sam2_vos",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_resize": {
                "applied": bool(resize_wh),
                "width": infer_w if resize_wh else None,
                "height": infer_h if resize_wh else None,
            },
            "notes": f"POC single-click SAM2 tracker; device={device}; model={model_ref}",
        },
        "target": {
            "target_id": "target-0",
            "init": {
                "type": "click",
                "frame_idx": 0,
                "click_xy": [int(click_x), int(click_y)],
                "init_box_xyxy": None,
            },
        },
        "frames": frames_json,
    }

    with masks_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True)

    logger.info("[sam2][%s] done total_time=%.2fs", job_ref, time.perf_counter() - total_start)
    progress_cb(1.0, "done")
    artifacts = RunArtifacts(frame0_path=frame0_path, overlay_path=overlay_path, masks_path=masks_path)
    return video_info, artifacts
