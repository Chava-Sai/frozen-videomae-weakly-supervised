#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p outputs outputs/step9_step7_calibration

python -u src/sweep_step7_temporal_calibration.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/ucf_crime/manifests/ucf_violence_features_i3d.csv \
  --master-manifest data/ucf_crime/manifests/ucf_violence_master.csv \
  --temporal-root data/ucf_crime/annotations/temporal_segments \
  --checkpoint outputs/rtfm_trn_boundary/checkpoints/best.pt \
  --output-dir outputs/step9_step7_calibration \
  --device "$DEVICE" | tee outputs/step9_step7_calibration/sweep.log
