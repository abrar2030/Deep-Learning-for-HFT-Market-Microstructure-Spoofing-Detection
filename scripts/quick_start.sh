#!/usr/bin/env bash
# Quick Start Script for TEN-GNN Production System
# Run from the project root: bash scripts/quick_start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo " TEN-GNN Quick Start"
echo "=========================================="
echo ""

# ── Dependency checks ──────────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "❌  $1 is not installed. $2"
        exit 1
    fi
    echo "✓  $1 found"
}

check_cmd docker       "Please install Docker first: https://docs.docker.com/get-docker/"
check_cmd docker-compose "Please install docker-compose: https://docs.docker.com/compose/install/"
check_cmd curl         "Please install curl."

# ── .env setup ─────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from template..."
    cp infrastructure/.env.example .env
    echo "⚠️   Please edit .env and set secure passwords before deploying to production."
    echo "    Run: nano .env"
    read -rp "Press Enter to continue..."
fi

# ── Runtime directories ────────────────────────────────────────────────────
echo ""
echo "Creating runtime directories..."
mkdir -p pretrained_models logs data alerts

# ── Pre-trained model check ────────────────────────────────────────────────
if [ ! -f pretrained_models/ten_model_synthetic.pth ]; then
    echo ""
    echo "⚠️   Pre-trained model not found at pretrained_models/ten_model_synthetic.pth"
    echo "    Options:"
    echo "      1. Train from scratch: python scripts/train_pretrained_model.py"
    echo "      2. Download pre-trained weights (if available)"
    echo ""
    read -rp "Continue without model? Services will fail to start. (y/N) " -n 1
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ── Profile selection ──────────────────────────────────────────────────────
echo ""
echo "Select deployment profile:"
echo "  1) CPU only       : recommended for development/testing"
echo "  2) GPU            : requires NVIDIA GPU + nvidia-container-toolkit"
echo "  3) Full stack     : CPU + Kafka streaming + Prometheus/Grafana monitoring"
echo ""
read -rp "Enter choice [1-3]: " profile_choice

case $profile_choice in
    1) PROFILES="--profile cpu" ;;
    2)
        echo ""
        echo "Verifying NVIDIA GPU access..."
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 \
             nvidia-smi &>/dev/null 2>&1; then
            echo "❌  NVIDIA GPU or nvidia-container-toolkit not available."
            echo "    See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
        echo "✓  GPU detected"
        PROFILES="--profile gpu"
        ;;
    3) PROFILES="--profile cpu --profile streaming --profile monitoring" ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# ── Build ──────────────────────────────────────────────────────────────────
echo ""
echo "Building Docker images (this may take a few minutes)..."
docker-compose build

# ── Start ──────────────────────────────────────────────────────────────────
echo ""
echo "Starting services ($PROFILES)..."
# shellcheck disable=SC2086
docker-compose $PROFILES up -d

# ── Health check ───────────────────────────────────────────────────────────
echo ""
echo "Waiting for API to become healthy..."
MAX_WAIT=60
for i in $(seq 1 $MAX_WAIT); do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        echo "✓  API is healthy!"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "⚠️   API did not respond within ${MAX_WAIT}s."
        echo "    Check logs: docker-compose logs -f ten-gnn-cpu"
    else
        printf "\r   Waiting... (%d/%d)" "$i" "$MAX_WAIT"
        sleep 1
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Deployment Complete"
echo "=========================================="
echo ""
docker-compose ps
echo ""
echo "Access points:"
echo "  API:          http://localhost:8000"
echo "  API docs:     http://localhost:8000/docs"
echo "  Health:       http://localhost:8000/health"
if [[ $PROFILES == *"monitoring"* ]]; then
    echo "  Prometheus:   http://localhost:9090"
    echo "  Grafana:      http://localhost:3000  (admin / see .env)"
fi
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop all:     docker-compose down"
echo "  Full teardown (incl. volumes): docker-compose down -v"
