#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=== beat-lens Spectrum Visualizer & Mood Classifier Demo ==="
echo "[1] Running test suite (DSP 440Hz peak check & feature vector)..."
pytest

echo "[2] Starting real-time 48-band spectrum visualizer with synthwave loop..."
echo "Press Ctrl+C to exit."
sleep 1
beat-lens --demo
