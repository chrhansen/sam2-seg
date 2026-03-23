# Shared Mask Track JSON Schema (v1)

This schema is designed so **both pipelines** (YOLO segmentation+tracking and SAM2 click-to-track) write the **same** `masks.json` format.

The file should be **one per processed video**.

---

## File: `masks.json`

### Top-level fields

```json
{
  "schema_version": "masktrack.v1",
  "video": {
    "source_filename": "input.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "num_frames": 600
  },
  "processing": {
    "method": "sam2_vos",
    "created_at": "2026-02-15T21:00:00Z",
    "frame_resize": {
      "applied": false,
      "width": null,
      "height": null
    },
    "notes": "freeform string"
  },
  "target": {
    "target_id": "target-0",
    "init": {
      "type": "click | box | auto",
      "frame_idx": 0,
      "click_xy": [960, 540],
      "init_box_xyxy": [800, 300, 1100, 1000]
    }
  },
  "frames": [
    {
      "frame_idx": 0,
      "present": true,
      "bbox_xyxy": [800, 300, 1100, 1000],
      "bbox_score": 0.92,
      "mask": {
        "encoding": "coco_rle",
        "size_hw": [1080, 1920],
        "counts": "eXj... (COCO RLE counts)"
      },
      "track": {
        "source_track_id": "7",
        "source_score": 0.88
      }
    }
  ]
}
```

---

## Semantics / requirements

### `schema_version`
- Must be exactly: `"masktrack.v1"`

### `video`
- `width`, `height` are the **original** video dimensions (before any resize).
- `fps` can be float.
- `num_frames` is the total frames decoded from the input.

### `processing.method`
- Use:
  - `"yolo_seg_track"` for Ultralytics YOLO segmentation + tracker (BoT-SORT/ByteTrack)
  - `"sam2_vos"` for SAM2 video object segmentation (prompted)

### `processing.frame_resize`
If you resize frames for inference:
- Set `applied: true`
- Set `width/height` to the dimensions used for mask inference.
- **All masks and bboxes in `frames[]` must still be stored in original video coordinates**.
  - If inference used resized frames, upsample mask back to original and transform bbox back to original.

### `target.init`
- For YOLO approach:
  - `type` typically `"auto"` (or `"click"` if you add a click-to-pick baseline)
- For SAM2 approach:
  - `type` should be `"click"` and `click_xy` required

### `frames[]`
One entry per frame from `0..num_frames-1`.

Fields:
- `present`:
  - `true` if target mask is available
  - `false` if target not visible / not confidently tracked
- `bbox_xyxy`:
  - `[x1, y1, x2, y2]` in **pixel coordinates** of original video
  - Omit or set to `null` if `present=false`
- `bbox_score`:
  - Float confidence (0..1) for the target in that frame (best-effort)
  - Can be `null` if not available
- `mask`:
  - If `present=true`, must include a binary mask encoded as **COCO RLE**:
    - `encoding`: `"coco_rle"`
    - `size_hw`: `[height, width]` of the original video
    - `counts`: RLE `counts` payload (string or list, depending on encoder)
  - If `present=false`, set `mask: null`

### `track` (optional)
If your method has track IDs/scores, store them for debugging:
- `source_track_id`: tracker ID as string (e.g., BoT-SORT ID)
- `source_score`: tracker association/confidence if available

---

## RLE notes (COCO RLE)

- Use `pycocotools.mask.encode()` / `decode()` for compatibility.
- `encode()` expects the mask array in **Fortran order** (column-major). Typical pattern:

```python
rle = pycocotools.mask.encode(np.asfortranarray(mask_uint8))
```

- Store `rle["counts"]` and `rle["size"]` (mapped to `size_hw`).

---

## Minimal “present=false” example

```json
{
  "frame_idx": 83,
  "present": false,
  "bbox_xyxy": null,
  "bbox_score": null,
  "mask": null,
  "track": null
}
```
