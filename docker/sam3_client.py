"""
sam3_client.py — Example client for the SAM3 REST server

Works from any machine on the same network as the Jetson.
Install: pip install requests Pillow
"""

import json
import base64
import requests
from pathlib import Path
from PIL import Image
import io

# ── Change this to your Jetson's IP ─────────────────────────
JETSON_IP = "192.168.1.100"   # ← update this
BASE_URL  = f"http://{JETSON_IP}:8000"


def check_health():
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    info = r.json()
    print(f"Status : {info['status']}")
    print(f"Device : {info['device']}")
    print(f"Model  : {'loaded' if info['model_loaded'] else 'NOT loaded'}")
    if info.get("gpu"):
        gpu = info["gpu"]
        print(f"GPU    : {gpu['name']}  {gpu['memory_allocated_mb']}/{gpu['memory_total_mb']} MB used")
    return info


def predict_with_text(image_path: str, text_prompt: str, include_masks: bool = False):
    """Find all instances of a concept in an image."""
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            data={
                "text_prompt": text_prompt,
                "include_masks": str(include_masks).lower(),
            },
        )
    r.raise_for_status()
    return r.json()


def predict_with_points(image_path: str, points: list, labels: list = None):
    """Click-style point prompt (SAM2 compatible)."""
    if labels is None:
        labels = [1] * len(points)
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            data={
                "points": json.dumps(points),
                "point_labels": json.dumps(labels),
                "include_masks": "true",
            },
        )
    r.raise_for_status()
    return r.json()


def predict_with_box(image_path: str, box: list):
    """Bounding-box prompt: [x1, y1, x2, y2]."""
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            data={
                "boxes": json.dumps([box]),
                "include_masks": "true",
            },
        )
    r.raise_for_status()
    return r.json()


def predict_video(video_path: str, text_prompt: str, max_frames: int = 100):
    """Track a concept across video frames."""
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/video",
            files={"file": (Path(video_path).name, f, "video/mp4")},
            data={
                "text_prompt": text_prompt,
                "max_frames": str(max_frames),
                "include_masks": "false",
            },
            timeout=300,   # video can take a while
        )
    r.raise_for_status()
    return r.json()


def decode_mask(mask_b64: str) -> Image.Image:
    """Decode a base64 mask PNG into a PIL Image."""
    mask_bytes = base64.b64decode(mask_b64)
    return Image.open(io.BytesIO(mask_bytes))


def print_detections(result: dict):
    """Pretty-print detection results."""
    dets = result.get("detections", [])
    print(f"\n  Inference time : {result.get('inference_time_s', '?')}s")
    print(f"  Device         : {result.get('device', '?')}")
    print(f"  Detections     : {len(dets)}")
    print()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [round(v, 1) for v in d["bbox_xyxy"]]
        score = round(d["score"], 3)
        has_mask = "mask_b64" in d
        print(f"  [{i}]  bbox=[{x1},{y1},{x2},{y2}]  score={score}  mask={'yes' if has_mask else 'no'}")


# ── Demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("SAM3 Client Demo")
    print("=" * 50)

    # 1. Health check
    print("\n[1] Health Check")
    check_health()

    # 2. Text prompt
    # Uncomment and set your image path:
    # print("\n[2] Text Prompt — 'person'")
    # result = predict_with_text("test.jpg", "person", include_masks=False)
    # print_detections(result)

    # 3. Point prompt
    # print("\n[3] Point Prompt — click at (320, 240)")
    # result = predict_with_points("test.jpg", points=[[320, 240]], labels=[1])
    # print_detections(result)

    # 4. Box prompt
    # print("\n[4] Box Prompt")
    # result = predict_with_box("test.jpg", box=[50, 50, 400, 400])
    # print_detections(result)
    # if result["detections"] and "mask_b64" in result["detections"][0]:
    #     mask = decode_mask(result["detections"][0]["mask_b64"])
    #     mask.save("mask_output.png")
    #     print("  Mask saved to mask_output.png")

    # 5. Video tracking
    # print("\n[5] Video Tracking — 'car'")
    # result = predict_video("test.mp4", "car", max_frames=50)
    # print(f"  Frames processed: {result['frames_processed']}")
    # print(f"  Total time: {result['inference_time_s']}s")
