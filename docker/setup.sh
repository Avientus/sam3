#!/usr/bin/env bash
# ============================================================
# setup.sh — Build and launch the SAM3 Docker container
# Tested on: JetPack 6.2.1 / L4T R36.4.3
# Run this ON the Jetson Orin
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Check we're on a Jetson ───────────────────────────────
if ! command -v tegrastats &> /dev/null; then
    warn "tegrastats not found — are you running this on a Jetson?"
fi

# ── 2. Check Docker is installed ────────────────────────────
command -v docker &> /dev/null || error "Docker not found. Install it first:\n  sudo apt-get install docker.io"

# ── 3. Check NVIDIA container runtime ───────────────────────
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    warn "NVIDIA Docker runtime may not be configured."
    warn "Install it with:"
    warn "  sudo apt-get install -y nvidia-container-toolkit"
    warn "  sudo nvidia-ctk runtime configure --runtime=docker"
    warn "  sudo systemctl restart docker"
fi

# ── 4. Create directories ────────────────────────────────────
mkdir -p models outputs
info "Created ./models and ./outputs directories."

# ── 5. Check for SAM3 weights ────────────────────────────────
if [ ! -f "models/sam3.pt" ]; then
    warn "models/sam3.pt not found!"
    echo ""
    echo "  You must download it from Hugging Face (requires access request):"
    echo "  https://huggingface.co/models?search=sam3"
    echo ""
    echo "  Once downloaded, place it at: $(pwd)/models/sam3.pt"
    echo ""
    read -p "  Continue building without weights? (y/N): " cont
    [[ "$cont" =~ ^[Yy]$ ]] || exit 0
fi

# ── 6. Confirm base image for this system ────────────────────
# JetPack 6.2.1 = L4T R36.4.3
# dusty-nv publishes r36.4.0 images which are compatible with ALL R36.4.x
# New-style tag: dustynv/pytorch:2.8-r36.4-cu128-24.04
#   (Ubuntu 24.04, CUDA 12.8, PyTorch 2.8)
info "Base image: dustynv/pytorch:2.7-r36.4.0-cu128-24.04"
info "Compatible with your JetPack 6.2.1 / L4T R36.4.3"

# ── 7. Expand swap (critical for Orin boards) ────────────────
SWAP_FILE="/swapfile_sam3"
if [ ! -f "$SWAP_FILE" ]; then
    info "Adding 8 GB swap space for model loading..."
    sudo fallocate -l 8G "$SWAP_FILE"
    sudo chmod 600 "$SWAP_FILE"
    sudo mkswap "$SWAP_FILE"
    sudo swapon "$SWAP_FILE"
    # Make permanent across reboots
    echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
    info "Swap enabled."
else
    info "Swap file already exists."
fi

# ── 8. Build the Docker image ────────────────────────────────
info "Building SAM3 Docker image (this may take 10-20 min first time)..."
info "Base: dustynv/pytorch:2.7-r36.4.0-cu128-24.04 (~6 GB pull on first run)"
docker compose build

# ── 9. Start the container ───────────────────────────────────
info "Starting SAM3 server..."
docker compose up -d

# ── 10. Wait for health ──────────────────────────────────────
info "Waiting for server to come up (model loading takes ~60s)..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        info "✅ SAM3 server is UP!"
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

# ── 11. Show access info ─────────────────────────────────────
JETSON_IP=$(hostname -I | awk '{print $1}')
echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SAM3 is running!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo "  From this Jetson:      http://localhost:8000"
echo "  From any device:       http://${JETSON_IP}:8000"
echo "  API docs (Swagger UI): http://${JETSON_IP}:8000/docs"
echo "  Health check:          http://${JETSON_IP}:8000/health"
echo ""
echo "  Logs:  docker compose logs -f"
echo "  Stop:  docker compose down"
echo ""
