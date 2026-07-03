#!/usr/bin/env bash
# =============================================================================
# Run script for Deep Learning for HFT Market Microstructure Spoofing
# Detection (TEN / TEN-GNN)
#
# Usage:
#   bash scripts/run.sh <command> [options]
#
# Commands:
#   test         Run the unit test suite
#   train        Train on synthetic LOB data (full defaults; pass CLI flags)
#   train-real   Train on real LOB data: run.sh train-real /path/to/data.csv
#   benchmark    Run inference latency / throughput benchmarks
#   robustness   Run adversarial robustness testing
#   figures      Generate paper figures
#   api          Launch the FastAPI detection service (requires redis)
#   demo         Quick smoke run: tiny model, 1 epoch (~2 min, CPU)
#   all          test -> train
#
# Extra arguments are forwarded to the underlying Python entry point:
#   bash scripts/run.sh train --num_epochs 20 --model_type TEN-GNN
#
# Real-data schema (CSV / Parquet): timestamp, best_bid, best_ask,
# bid_volume, ask_volume [, label, bid_price_2, ask_price_2, ...].
# Pre-windowed NPZ: sequences (N,T,F), labels (N,) [, time_deltas (N,T,1)].
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$ROOT_DIR/code"

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python3}"

usage() {
    awk 'NR>1 && /^# ={10,}/{c++; if(c==2) exit} NR>1{sub(/^# ?/,""); print}' "$0"
}

cmd="${1:-}"
shift || true

case "$cmd" in
    test)
        echo "[run.sh] Running unit tests..."
        cd "$CODE_DIR"
        "$PYTHON" -m pytest tests/ -v "$@"
        ;;

    train)
        echo "[run.sh] Training on synthetic LOB data..."
        cd "$CODE_DIR"
        "$PYTHON" train/train.py "$@"
        ;;

    train-real)
        data_path="${1:-}"
        if [[ -z "$data_path" ]]; then
            echo "Usage: bash scripts/run.sh train-real /path/to/data.{csv,parquet,npz} [flags]" >&2
            exit 1
        fi
        shift
        echo "[run.sh] Training on real LOB data: $data_path"
        cd "$CODE_DIR"
        "$PYTHON" train/train.py --data_path "$data_path" "$@"
        ;;

    benchmark)
        echo "[run.sh] Running performance benchmarks..."
        cd "$CODE_DIR"
        "$PYTHON" benchmarks/performance_benchmark.py "$@"
        ;;

    robustness)
        echo "[run.sh] Running adversarial robustness tests..."
        cd "$CODE_DIR"
        "$PYTHON" adversarial/robustness_testing.py "$@"
        ;;

    figures)
        echo "[run.sh] Generating paper figures..."
        cd "$ROOT_DIR"
        "$PYTHON" scripts/generate_figures.py "$@"
        ;;

    api)
        echo "[run.sh] Launching detection API (requires redis on localhost)..."
        cd "$ROOT_DIR"
        "$PYTHON" -m uvicorn infrastructure.api.server:app --host 0.0.0.0 --port 8000 "$@"
        ;;

    demo)
        echo "[run.sh] Demo: tiny TEN model, 1 epoch on synthetic data..."
        cd "$CODE_DIR"
        "$PYTHON" train/train.py \
            --num_samples 100 \
            --window_size 20 \
            --num_epochs 1 \
            --batch_size 8 \
            --d_model 32 \
            --num_layers 1 \
            --num_heads 2 \
            --d_ff 64 \
            --device cpu \
            --checkpoint_dir /tmp/hft_demo_ckpt \
            "$@"
        ;;

    all)
        bash "$0" test
        bash "$0" train
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        usage
        exit 1
        ;;
esac
