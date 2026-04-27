"""
SAM3 Inference Server — FastAPI
Uses the native Meta SAM3 API (no Ultralytics dependency).

SAM3 high-level predictor: build_sam3_predictor()
  - handle_request()        → single-response operations
  - handle_stream_request() → streaming (video propagation)

Endpoints:
  GET  /health          — liveness + GPU info
  GET  /model/info      — model metadata
  POST /predict/image   — segment image (text, point, or box prompt)
  POST /predict/video   — track objects across video frames
"""

import io
import os
import time
import base64
import shutil
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
    description="Segment Anything Model 3 REST API",
    version="3.0.0",
)

MODEL_PATH   = os.environ.get("SAM3_MODEL_PATH", "/app/models/sam3.pt")
SAM3_VERSION = os.environ.get("SAM3_VERSION", "sam3")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    log.info(f"Loading SAM3 predictor (version={SAM3_VERSION}) from {MODEL_PATH} ...")
    from sam3 import build_sam3_predictor
    compile_model = os.environ.get("SAM3_COMPILE", "").lower() in ("1", "true")
    _predictor = build_sam3_predictor(
        checkpoint_path=MODEL_PATH,
        version=SAM3_VERSION,
        compile=compile_model,
        warm_up=compile_model,
        use_fa3=False,  # FA3 requires Hopper (H100); disable for Jetson / consumer GPUs
    )
    log.info("SAM3 predictor loaded.")
    return _predictor


@app.on_event("startup")
async def startup_event():
    try:
        get_predictor()
    except RuntimeError as e:
        log.warning(f"Could not pre-load model: {e}")


# ── Helpers ──────────────────────────────────────────────────

def mask_to_b64(mask: np.ndarray) -> str:
    m = (mask.squeeze().astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def xywh_to_xyxy(box: list) -> list:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def parse_outputs(outputs: dict, include_masks: bool, img_w: int = 1, img_h: int = 1) -> list:
    """Convert a propagate_in_video outputs dict into a list of detection dicts."""
    detections = []
    log.info("propagate outputs keys: %s", list(outputs.keys()))
    binary_masks = outputs.get("out_binary_masks")  # BxHxW
    boxes_xywh   = outputs.get("out_boxes_xywh")    # Nx4, normalized [0,1]
    obj_ids      = outputs.get("out_obj_ids", [])
    obj_scores   = outputs.get("out_scores", outputs.get("out_obj_scores", []))

    if binary_masks is None:
        return detections

    for i, obj_id in enumerate(obj_ids):
        if i >= len(binary_masks):
            break
        mask = binary_masks[i]
        if not np.any(mask):
            continue
        score = float(obj_scores[i]) if i < len(obj_scores) else 1.0
        det: dict = {"obj_id": int(obj_id), "score": score}
        if boxes_xywh is not None and i < len(boxes_xywh):
            # Predictor returns normalized [0,1] — convert to pixels
            nx, ny, nw, nh = [float(v) for v in boxes_xywh[i]]
            x1 = round(nx * img_w)
            y1 = round(ny * img_h)
            x2 = round((nx + nw) * img_w)
            y2 = round((ny + nh) * img_h)
            det["bbox_xyxy"] = [x1, y1, x2, y2]
            det["bbox_xywh"] = [x1, y1, x2 - x1, y2 - y1]
        else:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                mh, mw = mask.shape[-2], mask.shape[-1]
                x1 = round(int(xs.min()) / mw * img_w)
                y1 = round(int(ys.min()) / mh * img_h)
                x2 = round(int(xs.max()) / mw * img_w)
                y2 = round(int(ys.max()) / mh * img_h)
                det["bbox_xyxy"] = [x1, y1, x2, y2]
                det["bbox_xywh"] = [x1, y1, x2 - x1, y2 - y1]
        if include_masks:
            det["mask_b64"] = mask_to_b64(mask)
        detections.append(det)

    return detections


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name":                torch.cuda.get_device_name(0),
            "memory_total_mb":     round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
        }
    return {
        "status":       "ok",
        "device":       DEVICE,
        "model_loaded": _predictor is not None,
        "model_path":   MODEL_PATH,
        "gpu":          gpu_info,
        "timestamp":    time.time(),
    }


@app.get("/model/info")
def model_info():
    return {
        "model_path": MODEL_PATH,
        "device":     DEVICE,
        "backend":    "sam3 (Meta)",
        "version":    SAM3_VERSION,
    }


@app.post("/predict/image")
async def predict_image(
    file:          UploadFile    = File(...),
    text_prompt:   Optional[str] = Form(None, description="Text concept e.g. 'person', 'red car'"),
    points:        Optional[str] = Form(None, description="JSON [[x,y], ...] point prompt"),
    point_labels:  Optional[str] = Form(None, description="JSON [1,0,...] — 1=fg 0=bg"),
    boxes:         Optional[str] = Form(None, description="JSON [[x1,y1,x2,y2], ...]"),
    include_masks: bool          = Form(True, description="Return base64 mask PNGs"),
):
    """
    Segment objects in an image using SAM3.
    Provide exactly one of: text_prompt, points, or boxes.
    """
    import json

    if not text_prompt and not points and not boxes:
        raise HTTPException(status_code=400, detail="Provide at least one of: text_prompt, points, or boxes.")

    img_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_w, img_h = pil_img.size

    # Treat the single image as a one-frame "video" in a temp directory
    frame_dir = tempfile.mkdtemp()
    try:
        pil_img.save(os.path.join(frame_dir, "00000.jpg"), format="JPEG")

        predictor = get_predictor()
        t0 = time.time()

        resp = predictor.handle_request({"type": "start_session", "resource_path": frame_dir})
        session_id = resp["session_id"]

        try:
            prompt_req: dict = {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "obj_id": 1}

            if text_prompt:
                prompt_req["text"] = text_prompt
                prompt_type = "text"
            elif points:
                pts  = json.loads(points)
                lbls = json.loads(point_labels) if point_labels else [1] * len(pts)
                prompt_req["points"]       = pts
                prompt_req["point_labels"] = lbls
                prompt_type = "points"
            else:
                prompt_req["bounding_boxes"] = json.loads(boxes)
                prompt_type = "boxes"

            predictor.handle_request(prompt_req)

            detections = []
            for response in predictor.handle_stream_request({
                "type":               "propagate_in_video",
                "session_id":         session_id,
                "output_prob_thresh": 0.5,
            }):
                detections.extend(parse_outputs(response.get("outputs", {}), include_masks, img_w, img_h))

        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return JSONResponse(content={
        "detections":       detections,
        "inference_time_s": round(time.time() - t0, 3),
        "image_size":       [img_w, img_h],
        "prompt_type":      prompt_type,
        "prompt_value":     text_prompt or points or boxes,
        "device":           DEVICE,
    })


@app.post("/predict/video")
async def predict_video(
    file:          UploadFile    = File(...),
    text_prompt:   Optional[str] = Form(None,  description="Text concept to track"),
    points:        Optional[str] = Form(None,  description="JSON [[x,y], ...] points on frame 0"),
    point_labels:  Optional[str] = Form(None,  description="JSON [1,0,...] — 1=fg 0=bg"),
    boxes:         Optional[str] = Form(None,  description="JSON [[x1,y1,x2,y2]] box on frame 0"),
    max_frames:    int           = Form(300,   description="Max frames to process"),
    include_masks: bool          = Form(False, description="Include masks in response"),
):
    """
    Track objects across video frames using SAM3.
    Prompt is applied to frame 0 and propagated forward.
    Provide at least one of: text_prompt, points, or boxes.
    """
    import json

    if not text_prompt and not points and not boxes:
        raise HTTPException(status_code=400, detail="Provide at least one of: text_prompt, points, or boxes.")

    video_bytes = await file.read()
    frame_dir = tempfile.mkdtemp()
    total_frames = 0
    fps = 0.0

    try:
        # Write video and extract frames
        video_path = os.path.join(frame_dir, "input.mp4")
        Path(video_path).write_bytes(video_bytes)

        frames_dir = os.path.join(frame_dir, "frames")
        os.makedirs(frames_dir)

        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        idx = 0
        while idx < min(total_frames, max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)
            idx += 1
        cap.release()
        extracted = idx

        predictor = get_predictor()
        t0 = time.time()

        resp = predictor.handle_request({"type": "start_session", "resource_path": frames_dir})
        session_id = resp["session_id"]

        try:
            prompt_req: dict = {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "obj_id": 1}

            if text_prompt:
                prompt_req["text"] = text_prompt
            elif points:
                pts  = json.loads(points)
                lbls = json.loads(point_labels) if point_labels else [1] * len(pts)
                prompt_req["points"]       = pts
                prompt_req["point_labels"] = lbls
            else:
                prompt_req["bounding_boxes"] = json.loads(boxes)

            predictor.handle_request(prompt_req)

            frame_results = []
            for response in predictor.handle_stream_request({
                "type":               "propagate_in_video",
                "session_id":         session_id,
                "output_prob_thresh": 0.5,
            }):
                frame_results.append({
                    "frame_idx":  response["frame_index"],
                    "detections": parse_outputs(response.get("outputs", {}), include_masks, vid_w, vid_h),
                })

        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return JSONResponse(content={
        "frames_processed":   len(frame_results),
        "total_video_frames": total_frames,
        "fps":                fps,
        "inference_time_s":   round(time.time() - t0, 3),
        "device":             DEVICE,
        "frames":             frame_results,
    })
