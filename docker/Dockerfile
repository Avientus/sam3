# ============================================================
# SAM3 on Jetson Orin — Docker Image
# Tested on: JetPack 6.2.1 / L4T R36.4.3
#
# Base image: dustynv/pytorch:2.8-r36.4-cu128-24.04
#   - Ubuntu 24.04, CUDA 12.8, PyTorch 2.8, Python 3.10
#   - Compatible with ALL L4T R36.4.x boards (r36.4.0 → r36.4.3)
#   - Note: dusty-nv only publishes r36.4.0 tagged images but
#     they run on r36.4.3 (JetPack 6.2.1) without issues
#
# Exposes a FastAPI REST server on port 8000
# ============================================================

FROM dustynv/pytorch:2.8-r36.4-cu128-24.04

LABEL maintainer="you"
LABEL description="SAM3 inference server for Jetson Orin"

# ── System deps ─────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA env (avoids CUDA_HOME errors on Jetson) ────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ── Pin numpy to avoid version conflicts ────────────────────
RUN pip3 install --no-cache-dir numpy==1.26.4

# ── Ultralytics (includes SAM3 integration) ─────────────────
RUN pip3 install --no-cache-dir "ultralytics>=8.3.237"

# ── Fix CLIP package conflict for SAM3 text prompts ─────────
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 uninstall -y clip 2>/dev/null || true && \
    pip3 install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# ── FastAPI inference server dependencies ───────────────────
RUN pip3 install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    python-multipart==0.0.9 \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.8.0

# ── App directory ───────────────────────────────────────────
WORKDIR /app

# Copy server code
COPY server.py /app/server.py

# Model weights directory — mount your sam3.pt here at runtime
# or COPY it in if you have it locally:
#   COPY sam3.pt /app/models/sam3.pt
RUN mkdir -p /app/models /app/outputs

# ── Healthcheck ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
