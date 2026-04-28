#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"
STAGE1_EPOCHS="${3:-30}"
STAGE2_EPOCHS="${4:-30}"
STAGE3_EPOCHS="${5:-40}"
BATCH_SIZE="${6:-64}"
TARGET_SEGMENTS="${7:-32}"
PSEUDO_TOPK="${8:-4}"

cd "$PROJECT_ROOT"
mkdir -p outputs outputs/rtfm_progressive

python -u src/train_rtfm_progressive.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/ucf_crime/manifests/ucf_violence_features_i3d.csv \
  --master-manifest data/ucf_crime/manifests/ucf_violence_master.csv \
  --temporal-root data/ucf_crime/annotations/temporal_segments \
  --output-dir outputs/rtfm_progressive \
  --stage1-epochs "$STAGE1_EPOCHS" \
  --stage2-epochs "$STAGE2_EPOCHS" \
  --stage3-epochs "$STAGE3_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --target-segments "$TARGET_SEGMENTS" \
  --pseudo-topk "$PSEUDO_TOPK" \
  --balanced-sampler \
  --checkpoint-stage stage3 \
  --checkpoint-metric val_macro_f1 \
  --device "$DEVICE" | tee outputs/rtfm_progressive/train.log
