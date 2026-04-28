#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/sai/Documents/IVC Project}"
DEVICE="${2:-cuda}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -u src/extract_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --mode full \
  --device "$DEVICE" \
  --checkpoint-every 10

python src/summarize_i3d_features.py \
  --project-root "$PROJECT_ROOT"

python src/run_feature_loader_sanity.py \
  --project-root "$PROJECT_ROOT" \
  --split train \
  --batch-size 8 \
  --target-segments 32
