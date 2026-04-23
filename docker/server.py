"""
SAM3 Inference Server — FastAPI
Runs inside the Docker container on Jetson Orin.

Endpoints:
  GET  /health          — liveness check + GPU info
  GET  /model/info      — model metadata
  POST /predict/image   — segment image with text or point/box prompt
  POST /predict/video   — segment video with text prompt (returns per-frame results)

Usage examples (from anywhere on your network):
  curl http://<jetson-ip>:8000/health
  curl -X POST http://<jetson-ip>:8000/predict/image \
       -F "file=@photo.jpg" -F "text_prompt=red car"
"""

import io
import os
import time
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam3-server")

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="SAM3 Inference Server",
    description="Segment Anything Model 3 REST API running on Jetson Orin",
    version="1.0.0",
)

# ── Model globals ────────────────────────────────────────────
MODEL = None
MODEL_PATH = os.environ.get("SAM3_MODEL_PATH", "/app/models/sam3.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}. "
            "Mount your sam3.pt via: -v /path/to/sam3.pt:/app/models/sam3.pt"
        )
    log.info(f"Loading SAM3 from {MODEL_PATH} on {DEVICE} ...")
    from ultralytics import SAM
    MODEL = SAM(MODEL_PATH)
    MODEL.to(DEVICE)
    log.info("SAM3 loaded successfully.")
    return MODEL


@app.on_event("startup")
async def startup_event():
    """Pre-load model at startup so first request isn't slow."""
    try:
        load_model()
    except RuntimeError as e:
        log.warning(f"Model not loaded at startup: {e}")


# ── Helpers ──────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_b64(mask: np.ndarray) -> str:
    """Convert a boolean/float mask to a base64 PNG."""
    mask_uint8 = (mask.squeeze().astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8)
    return pil_to_b64(img)


def results_to_dict(results, include_masks: bool = True) -> dict:
    """Convert Ultralytics Results object to a JSON-serialisable dict."""
    output = {"detections": []}

    for r in results:
        boxes = r.boxes
        masks = r.masks if hasattr(r, "masks") and r.masks is not None else None

        if boxes is None:
            continue

        for i in range(len(boxes)):
            det = {
                "bbox_xyxy": boxes.xyxy[i].tolist(),   # [x1, y1, x2, y2]
                "bbox_xywh": boxes.xywh[i].tolist(),   # [cx, cy, w, h]
                "score": float(boxes.conf[i]),
                "class_id": int(boxes.cls[i]) if boxes.cls is not None else -1,
            }
            if include_masks and masks is not None:
                try:
                    det["mask_b64"] = mask_to_b64(masks.data[i].cpu().numpy())
                except Exception as e:
                    log.warning(f"Could not encode mask {i}: {e}")
            output["detections"].append(det)

    return output


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "gpu": gpu_info,
        "timestamp": time.time(),
    }


@app.get("/model/info")
def model_info():
    try:
        m = load_model()
        return {
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "type": type(m).__name__,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    text_prompt: Optional[str] = Form(None, description="Text concept, e.g. 'red car'"),
    points: Optional[str] = Form(None, description="JSON list of [x,y] points, e.g. '[[320,240]]'"),
    point_labels: Optional[str] = Form(None, description="JSON list of labels (1=fg, 0=bg), e.g. '[1]'"),
    boxes: Optional[str] = Form(None, description="JSON list of [x1,y1,x2,y2], e.g. '[[10,10,200,200]]'"),
    include_masks: bool = Form(True, description="Include base64-encoded masks in response"),
):
    """
    Segment objects in an image.

    Prompt options (use one):
      - text_prompt: finds all instances of a concept ("yellow school bus")
      - points + point_labels: classic SAM point prompt
      - boxes: bounding-box prompt

    Returns:
      detections[]: bbox_xyxy, bbox_xywh, score, class_id, mask_b64 (optional)
    """
    import json

    try:
        m = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Read image
    img_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    t0 = time.time()

    try:
        if text_prompt:
            results = m.predict(pil_img, texts=[text_prompt], verbose=False)

        elif points:
            pts = json.loads(points)
            lbls = json.loads(point_labels) if point_labels else [1] * len(pts)
            results = m.predict(
                pil_img,
                points=[pts],
                labels=[lbls],
                verbose=False,
            )

        elif boxes:
            bxs = json.loads(boxes)
            results = m.predict(pil_img, bboxes=bxs, verbose=False)

        else:
            # No prompt — auto-segment everything
            results = m.predict(pil_img, verbose=False)

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    elapsed = round(time.time() - t0, 3)
    output = results_to_dict(results, include_masks=include_masks)
    output["inference_time_s"] = elapsed
    output["image_size"] = list(pil_img.size)
    output["prompt"] = text_prompt or points or boxes or "auto"
    output["device"] = DEVICE

    return JSONResponse(content=output)


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(..., description="Video file (mp4/avi)"),
    text_prompt: Optional[str] = Form(None, description="Text concept to track"),
    max_frames: int = Form(300, description="Maximum frames to process"),
    include_masks: bool = Form(False, description="Include masks (increases response size)"),
):
    """
    Track a concept across video frames.
    Returns per-frame detections with bboxes and scores.
    """
    try:
        m = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Save video to temp file
    video_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    t0 = time.time()
    frame_results = []

    try:
        kwargs = {"verbose": False, "stream": True}
        if text_prompt:
            kwargs["texts"] = [text_prompt]

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        results_gen = m.track(tmp_path, persist=True, **kwargs)

        for frame_idx, r in enumerate(results_gen):
            if frame_idx >= max_frames:
                break
            frame_data = results_to_dict([r], include_masks=include_masks)
            frame_data["frame_idx"] = frame_idx
            frame_results.append(frame_data)

    except Exception as e:
        log.exception("Video inference error")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {e}")
    finally:
        os.unlink(tmp_path)

    elapsed = round(time.time() - t0, 3)

    return JSONResponse(content={
        "frames_processed": len(frame_results),
        "total_video_frames": total_frames,
        "fps": fps,
        "inference_time_s": elapsed,
        "text_prompt": text_prompt,
        "device": DEVICE,
        "frames": frame_results,
    })
