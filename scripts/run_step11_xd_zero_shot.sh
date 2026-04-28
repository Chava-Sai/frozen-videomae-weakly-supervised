#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <project_root> [device]"
  echo "Example: $0 /projectnb/cs585/students/saichava/IVC_Project cuda"
  exit 1
fi

PROJECT_ROOT="$1"
DEVICE="${2:-cuda}"

cd "$PROJECT_ROOT"

mkdir -p outputs/xd_violence_zero_shot
mkdir -p data/xd_violence/manifests

echo "=== Step-11: prepare XD manifest ==="
python src/prepare_xd_violence_manifest.py \
  --project-root "$PROJECT_ROOT" \
  --raw-root data/xd_violence/raw_videos \
  --output-master data/xd_violence/manifests/xd_violence_master.csv \
  --output-eval data/xd_violence/manifests/xd_violence_zero_shot_eval.csv \
  --output-report data/xd_violence/manifests/xd_violence_manifest_report.json

echo "=== Step-11: extract XD I3D features (same settings family as UCF) ==="
python src/extract_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --manifest data/xd_violence/manifests/xd_violence_master.csv \
  --output-root data/xd_violence/features/i3d_kinetics400_16f \
  --feature-manifest data/xd_violence/manifests/xd_violence_features_i3d.csv \
  --sanity-report data/xd_violence/manifests/xd_i3d_feature_sanity_report.md \
  --mode full \
  --segment-length 16 \
  --batch-size 8 \
  --resize-shorter-side 256 \
  --crop-size 224 \
  --device "$DEVICE"

echo "=== Step-11: summarize XD features ==="
python src/summarize_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --video-manifest data/xd_violence/manifests/xd_violence_master.csv \
  --feature-manifest data/xd_violence/manifests/xd_violence_features_i3d.csv \
  --feature-root data/xd_violence/features/i3d_kinetics400_16f \
  --output-report data/xd_violence/manifests/xd_i3d_feature_summary.md \
  --output-json data/xd_violence/manifests/xd_i3d_feature_summary.json

echo "=== Step-11: zero-shot inference on XD ==="
python src/eval_xd_zero_shot.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/xd_violence/manifests/xd_violence_features_i3d.csv \
  --master-manifest data/xd_violence/manifests/xd_violence_master.csv \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --split test \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --output-json outputs/xd_violence_zero_shot/results_summary.json \
  --device "$DEVICE"

echo "=== Step-11: advisor report ==="
python src/print_step11_xd_report.py \
  --project-root "$PROJECT_ROOT" \
  --results outputs/xd_violence_zero_shot/results_summary.json \
  | tee outputs/xd_violence_zero_shot/step11_report.txt

echo "Step-11 complete"
echo "- results: $PROJECT_ROOT/outputs/xd_violence_zero_shot/results_summary.json"
echo "- report:  $PROJECT_ROOT/outputs/xd_violence_zero_shot/step11_report.txt"

