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

mkdir -p outputs/shanghaitech_robustness
mkdir -p data/shanghaitech/manifests

LABEL_ARGS=()
if [[ -f data/shanghaitech/annotations/video_labels.csv ]]; then
  LABEL_ARGS+=(
    --label-csv data/shanghaitech/annotations/video_labels.csv
    --label-key video_id
    --label-value binary_label
  )
fi

echo "=== Step-13: prepare ShanghaiTech manifest ==="
python src/prepare_shanghaitech_manifest.py \
  --project-root "$PROJECT_ROOT" \
  --raw-root data/shanghaitech/raw_videos \
  --output-master data/shanghaitech/manifests/shanghaitech_master.csv \
  --output-eval data/shanghaitech/manifests/shanghaitech_eval.csv \
  --output-report data/shanghaitech/manifests/shanghaitech_manifest_report.json \
  --eval-split test \
  "${LABEL_ARGS[@]}"

echo "=== Step-13: extract ShanghaiTech I3D features (same settings as UCF/XD/RWF) ==="
python src/extract_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --manifest data/shanghaitech/manifests/shanghaitech_master.csv \
  --output-root data/shanghaitech/features/i3d_kinetics400_16f \
  --feature-manifest data/shanghaitech/manifests/shanghaitech_features_i3d.csv \
  --sanity-report data/shanghaitech/manifests/shanghaitech_i3d_feature_sanity_report.md \
  --mode full \
  --segment-length 16 \
  --batch-size 8 \
  --resize-shorter-side 256 \
  --crop-size 224 \
  --device "$DEVICE"

echo "=== Step-13: summarize ShanghaiTech features ==="
python src/summarize_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --video-manifest data/shanghaitech/manifests/shanghaitech_master.csv \
  --feature-manifest data/shanghaitech/manifests/shanghaitech_features_i3d.csv \
  --feature-root data/shanghaitech/features/i3d_kinetics400_16f \
  --output-report data/shanghaitech/manifests/shanghaitech_i3d_feature_summary.md \
  --output-json data/shanghaitech/manifests/shanghaitech_i3d_feature_summary.json

echo "=== Step-13: ShanghaiTech robustness eval (binary only, zero-shot) ==="
python src/eval_shanghaitech_robustness.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/shanghaitech/manifests/shanghaitech_features_i3d.csv \
  --master-manifest data/shanghaitech/manifests/shanghaitech_master.csv \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --split test \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --output-json outputs/shanghaitech_robustness/results_summary.json \
  --device "$DEVICE"

echo "=== Step-13: advisor report ==="
python src/print_step13_shanghaitech_report.py \
  --project-root "$PROJECT_ROOT" \
  --results outputs/shanghaitech_robustness/results_summary.json \
  | tee outputs/shanghaitech_robustness/step13_report.txt

echo "Step-13 complete"
echo "- results: $PROJECT_ROOT/outputs/shanghaitech_robustness/results_summary.json"
echo "- report:  $PROJECT_ROOT/outputs/shanghaitech_robustness/step13_report.txt"
