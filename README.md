# SAM2 skier segmentation POC

Upload short ski clip, click skier on frame 0, get:
- `overlay.mp4`
- `masks.json` (`masktrack.v1` schema)

## Quickstart

1. Create env + install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install SAM2 (official repo):

```bash
git clone https://github.com/facebookresearch/sam2.git /tmp/sam2
pip install -e /tmp/sam2
```

3. Download checkpoint (or use Hugging Face model id).

Local checkpoint mode (default envs expected):
- `SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_t.yaml`
- `SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_tiny.pt`

Hugging Face mode:

```bash
export SAM2_HF_MODEL_ID=facebook/sam2.1-hiera-tiny
```

4. Run app:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## RunPod A40 (Docker + GHCR)

This repo includes `Dockerfile.runpod` plus GitHub Actions workflow
`.github/workflows/docker-ghcr.yml` that builds and publishes a public image to GHCR.

Published tags:
- `ghcr.io/chrhansen/sam2-seg:latest`
- `ghcr.io/chrhansen/sam2-seg:runpod`
- `ghcr.io/chrhansen/sam2-seg:sha-<short_sha>`

Trigger build:

```bash
gh workflow run build-and-push-runpod-image
```

Check build:

```bash
gh run list --workflow build-and-push-runpod-image
```

Run locally with GPU:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e SAM2_DEVICE=cuda \
  -e SAM2_HF_MODEL_ID=facebook/sam2.1-hiera-large \
  -e SAM2_VOS_OPTIMIZED=0 \
  ghcr.io/chrhansen/sam2-seg:latest
```

For RunPod Pod image field, use: `ghcr.io/chrhansen/sam2-seg:latest`.

## Optional knobs

- `SAM2_DEVICE`: `cuda`, `mps`, `cpu` (default: `mps`)
- `SAM2_HF_MODEL_ID`: default `facebook/sam2.1-hiera-tiny`
- `SAM2_VOS_OPTIMIZED`: `1|0`
- `SAM2_FILL_HOLE_AREA`: official SAM2 hole-fill postprocess area (default: `200`; set `0` to disable)
- `INFER_RESIZE_WIDTH`, `INFER_RESIZE_HEIGHT`: optional inference downscale
- `RUNS_DIR`: output directory

For explicit CPU run:

```bash
export SAM2_DEVICE=cpu
```

## Notes

- POC quality > polish. Single-object tracking only.
- `frames[].present=false` when no mask in frame.
- Outputs stored under `runs/<job_id>/`.
- Server logs include `[sam2][<job_id>]` lines with requested/resolved device and inference speed.
- RunPod large-model default uses `SAM2_VOS_OPTIMIZED=0`; `1` can spend many minutes in first-run Triton/Torch compile/autotune and appear stuck at `20%`.
