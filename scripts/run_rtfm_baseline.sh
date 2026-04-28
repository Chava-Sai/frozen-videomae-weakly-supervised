#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-64}"
TARGET_SEGMENTS="${5:-32}"

cd "$PROJECT_ROOT"
mkdir -p outputs outputs/rtfm_baseline

python -u src/train_rtfm_baseline.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/ucf_crime/manifests/ucf_violence_features_i3d.csv \
  --output-dir outputs/rtfm_baseline \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --target-segments "$TARGET_SEGMENTS" \
  --balanced-sampler \
  --device "$DEVICE" | tee outputs/rtfm_baseline/train.log
