#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-64}"
TARGET_SEGMENTS="${5:-32}"
PSEUDO_TOPK="${6:-4}"

cd "$PROJECT_ROOT"
mkdir -p outputs outputs/rtfm_classifier

python -u src/train_rtfm_classifier.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/ucf_crime/manifests/ucf_violence_features_i3d.csv \
  --init-ckpt outputs/rtfm_baseline/checkpoints/best.pt \
  --output-dir outputs/rtfm_classifier \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --target-segments "$TARGET_SEGMENTS" \
  --pseudo-topk "$PSEUDO_TOPK" \
  --balanced-sampler \
  --checkpoint-metric val_macro_f1 \
  --device "$DEVICE" | tee outputs/rtfm_classifier/train.log
