from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    sam2_model_cfg: str
    sam2_checkpoint: str
    sam2_hf_model_id: str | None
    sam2_device: str | None
    sam2_vos_optimized: bool
    sam2_fill_hole_area: int
    infer_resize_width: int | None
    infer_resize_height: int | None



def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _env_nonneg_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    return max(0, value)



def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    runs_dir = Path(os.getenv("RUNS_DIR", root / "runs")).resolve()
    return Settings(
        runs_dir=runs_dir,
        sam2_model_cfg=os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        sam2_checkpoint=os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_tiny.pt"),
        sam2_hf_model_id=os.getenv("SAM2_HF_MODEL_ID", "facebook/sam2.1-hiera-tiny") or None,
        sam2_device=os.getenv("SAM2_DEVICE", "mps"),
        sam2_vos_optimized=_env_bool("SAM2_VOS_OPTIMIZED", False),
        sam2_fill_hole_area=_env_nonneg_int("SAM2_FILL_HOLE_AREA", 48),
        infer_resize_width=_env_int("INFER_RESIZE_WIDTH"),
        infer_resize_height=_env_int("INFER_RESIZE_HEIGHT"),
    )
