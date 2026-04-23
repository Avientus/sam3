"""
SAM3 Inference Server — FastAPI
Runs inside the Docker container on Jetson Orin.

SAM3 has TWO distinct predictor interfaces:
  - SAM3SemanticPredictor  → text prompts  (finds ALL instances of a concept)
  - SAM("sam3.pt").predict → point / box prompts (segments specific location)

Endpoints:
  GET  /health          — liveness + GPU info
  GET  /model/info      — model metadata
  POST /predict/image   — segment image (text, point, or box prompt)
  POST /predict/video   — track concept across video frames
"""

import io
import os
import time
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam3-server")

app = FastAPI(
    title="SAM3 Inference Server",
    description="Segment Anything Model 3 REST API on Jetson Orin",
    version="2.0.0",
)

MODEL_PATH = os.environ.get("SAM3_MODEL_PATH", "/app/models/sam3.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Two separate predictor objects — loaded lazily
_visual_model    = None   # SAM("sam3.pt")          — points / boxes
_semantic_pred   = None   # SAM3SemanticPredictor   — text prompts


def get_visual_model():
    """SAM3 visual predictor for point/box prompts."""
    global _visual_model
    if _visual_model is not None:
        return _visual_model
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    log.info(f"Loading SAM3 visual model from {MODEL_PATH} on {DEVICE} ...")
    from ultralytics import SAM
    _visual_model = SAM(MODEL_PATH)
    _visual_model.to(DEVICE)
    log.info("SAM3 visual model loaded.")
    return _visual_model


def get_semantic_predictor():
    """SAM3SemanticPredictor for text prompts."""
    global _semantic_pred
    if _semantic_pred is not None:
        return _semantic_pred
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    log.info(f"Loading SAM3SemanticPredictor from {MODEL_PATH} on {DEVICE} ...")
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=MODEL_PATH,
        half=(DEVICE == "cuda"),   # FP16 on GPU, FP32 on CPU
        verbose=False,
        save=False,
    )
    _semantic_pred = SAM3SemanticPredictor(overrides=overrides)
    log.info("SAM3SemanticPredictor loaded.")
    return _semantic_pred


@app.on_event("startup")
async def startup_event():
    try:
        get_visual_model()
        get_semantic_predictor()
    except RuntimeError as e:
        log.warning(f"Could not pre-load models: {e}")


# ── Helpers ──────────────────────────────────────────────────

def mask_to_b64(mask: np.ndarray) -> str:
    m = (mask.squeeze().astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def results_to_dict(results, include_masks: bool = True) -> dict:
    detections = []
    for r in results:
        boxes = r.boxes
        masks = r.masks if hasattr(r, "masks") and r.masks is not None else None
        if boxes is None:
            continue
        for i in range(len(boxes)):
            det = {
                "bbox_xyxy": boxes.xyxy[i].tolist(),
                "bbox_xywh": boxes.xywh[i].tolist(),
                "score":     float(boxes.conf[i]),
                "class_id":  int(boxes.cls[i]) if boxes.cls is not None else -1,
            }
            if include_masks and masks is not None:
                try:
                    det["mask_b64"] = mask_to_b64(masks.data[i].cpu().numpy())
                except Exception as ex:
                    log.warning(f"Mask encode error: {ex}")
            detections.append(det)
    return {"detections": detections}


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name":                 torch.cuda.get_device_name(0),
            "memory_total_mb":      round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
            "memory_allocated_mb":  round(torch.cuda.memory_allocated(0) / 1e6, 1),
        }
    return {
        "status":        "ok",
        "device":        DEVICE,
        "visual_loaded":   _visual_model is not None,
        "semantic_loaded": _semantic_pred is not None,
        "model_path":    MODEL_PATH,
        "gpu":           gpu_info,
        "timestamp":     time.time(),
    }


@app.get("/model/info")
def model_info():
    return {
        "model_path": MODEL_PATH,
        "device":     DEVICE,
        "note":       "text_prompt uses SAM3SemanticPredictor; points/boxes use SAM3Predictor",
    }


@app.post("/predict/image")
async def predict_image(
    file:          UploadFile        = File(...),
    text_prompt:   Optional[str]     = Form(None,  description="Text concept e.g. 'person', 'red car'"),
    points:        Optional[str]     = Form(None,  description="JSON [[x,y], ...] point prompt"),
    point_labels:  Optional[str]     = Form(None,  description="JSON [1,0,...] — 1=fg 0=bg"),
    boxes:         Optional[str]     = Form(None,  description="JSON [[x1,y1,x2,y2], ...]"),
    include_masks: bool              = Form(True,  description="Return base64 mask PNGs"),
):
    """
    Segment objects in an image using SAM3.

    Prompt types:
      text_prompt  → SAM3SemanticPredictor — finds ALL instances of a concept
      points       → SAM3 visual predictor — segment object at clicked location
      boxes        → SAM3 visual predictor — segment object inside bounding box
      (none)       → auto-segment everything with visual predictor
    """
    import json

    img_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Save to temp file (needed by SemanticPredictor.set_image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp.name, format="JPEG")
        tmp_path = tmp.name

    t0 = time.time()
    try:
        if text_prompt:
            # ── Text prompt path ─────────────────────────────
            predictor = get_semantic_predictor()
            predictor.set_image(tmp_path)
            results = predictor(text=[text_prompt])

        elif points:
            # ── Point prompt path ────────────────────────────
            m = get_visual_model()
            pts  = json.loads(points)
            lbls = json.loads(point_labels) if point_labels else [1] * len(pts)
            results = m.predict(tmp_path, points=[pts], labels=[lbls], verbose=False)

        elif boxes:
            # ── Box prompt path ──────────────────────────────
            m = get_visual_model()
            bxs = json.loads(boxes)
            results = m.predict(tmp_path, bboxes=bxs, verbose=False)

        else:
            # ── No prompt — segment everything ───────────────
            m = get_visual_model()
            results = m.predict(tmp_path, verbose=False)

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)

    elapsed = round(time.time() - t0, 3)
    output  = results_to_dict(results, include_masks=include_masks)
    output["inference_time_s"] = elapsed
    output["image_size"]       = list(pil_img.size)
    output["prompt_type"]      = "text" if text_prompt else ("points" if points else ("boxes" if boxes else "auto"))
    output["prompt_value"]     = text_prompt or points or boxes or "auto"
    output["device"]           = DEVICE
    return JSONResponse(content=output)


@app.post("/predict/video")
async def predict_video(
    file:          UploadFile    = File(...),
    text_prompt:   Optional[str] = Form(None,  description="Text concept to track"),
    max_frames:    int           = Form(300,   description="Max frames to process"),
    include_masks: bool          = Form(False, description="Include masks in response"),
):
    """
    Track a concept across video frames using SAM3VideoSemanticPredictor.
    Returns per-frame detections with bboxes and scores.
    """
    import json

    video_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    t0 = time.time()
    frame_results = []

    try:
        cap          = cv2.VideoCapture(tmp_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if text_prompt:
            from ultralytics.models.sam import SAM3SemanticPredictor
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=MODEL_PATH, half=(DEVICE == "cuda"),
                verbose=False, save=False,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
            results_gen = predictor(source=tmp_path, text=[text_prompt], stream=True)
        else:
            m = get_visual_model()
            results_gen = m.track(tmp_path, persist=True, stream=True, verbose=False)

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

    return JSONResponse(content={
        "frames_processed":  len(frame_results),
        "total_video_frames": total_frames,
        "fps":               fps,
        "inference_time_s":  round(time.time() - t0, 3),
        "text_prompt":       text_prompt,
        "device":            DEVICE,
        "frames":            frame_results,
    })
