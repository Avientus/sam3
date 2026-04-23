import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results, plot_bbox

IMAGE_PATH = "/home/simon/Downloads/crowd.jpg"
PROMPT = "woman with white hair"
OUTPUT_PATH = "/home/simon/sam3/output_people.png"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(start):
    cuda_sync()
    return time.perf_counter() - start


# ── Model load ────────────────────────────────────────────────────────────────
t = time.perf_counter()
model = build_sam3_image_model(enable_segmentation=False)
processor = Sam3Processor(model, confidence_threshold=0.8)
print(f"Model load:       {elapsed(t):.2f}s")

# ── Image load + preprocessing ────────────────────────────────────────────────
t = time.perf_counter()
image = Image.open(IMAGE_PATH).convert("RGB")
inference_state = processor.set_image(image)
print(f"Image encode:     {elapsed(t):.2f}s")

# ── Text encode + detection ───────────────────────────────────────────────────
t = time.perf_counter()
output = processor.set_text_prompt(state=inference_state, prompt=PROMPT)
print(f"Text + detection: {elapsed(t):.2f}s")

# ── Results ───────────────────────────────────────────────────────────────────
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"\nFound {len(scores)} detection(s)")
for i, score in enumerate(scores):
    print(f"  [{i}] score={score.item():.3f}  box={boxes[i].cpu().tolist()}")

# ── Visualisation ─────────────────────────────────────────────────────────────
t = time.perf_counter()
if output["masks"] is not None:
    plot_results(image, output)
else:
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(image))
    w, h = image.size
    for i, (box, score) in enumerate(zip(boxes.cpu(), scores.cpu())):
        plot_bbox(h, w, box, text=f"(id={i}, score={score:.2f})", box_format="XYXY", relative_coords=False)
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
print(f"\nVisualization:    {elapsed(t):.2f}s")
print(f"Saved to {OUTPUT_PATH}")
