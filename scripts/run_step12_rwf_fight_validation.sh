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

mkdir -p outputs/rwf_2000_fight_validation
mkdir -p data/rwf_2000/manifests

echo "=== Step-12: prepare RWF manifest ==="
python src/prepare_rwf_2000_manifest.py \
  --project-root "$PROJECT_ROOT" \
  --raw-root data/rwf_2000/raw_videos \
  --output-master data/rwf_2000/manifests/rwf_2000_master.csv \
  --output-eval data/rwf_2000/manifests/rwf_2000_eval.csv \
  --output-report data/rwf_2000/manifests/rwf_2000_manifest_report.json \
  --eval-split val

echo "=== Step-12: extract RWF I3D features (same settings as UCF/XD) ==="
python src/extract_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --manifest data/rwf_2000/manifests/rwf_2000_master.csv \
  --output-root data/rwf_2000/features/i3d_kinetics400_16f \
  --feature-manifest data/rwf_2000/manifests/rwf_2000_features_i3d.csv \
  --sanity-report data/rwf_2000/manifests/rwf_i3d_feature_sanity_report.md \
  --mode full \
  --segment-length 16 \
  --batch-size 8 \
  --resize-shorter-side 256 \
  --crop-size 224 \
  --device "$DEVICE"

echo "=== Step-12: summarize RWF features ==="
python src/summarize_i3d_features.py \
  --project-root "$PROJECT_ROOT" \
  --video-manifest data/rwf_2000/manifests/rwf_2000_master.csv \
  --feature-manifest data/rwf_2000/manifests/rwf_2000_features_i3d.csv \
  --feature-root data/rwf_2000/features/i3d_kinetics400_16f \
  --output-report data/rwf_2000/manifests/rwf_i3d_feature_summary.md \
  --output-json data/rwf_2000/manifests/rwf_i3d_feature_summary.json

echo "=== Step-12: RWF fight validation (zero-shot, no retraining) ==="
python src/eval_rwf_fight_validation.py \
  --project-root "$PROJECT_ROOT" \
  --feature-manifest data/rwf_2000/manifests/rwf_2000_features_i3d.csv \
  --master-manifest data/rwf_2000/manifests/rwf_2000_master.csv \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --split val \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --output-json outputs/rwf_2000_fight_validation/results_summary.json \
  --device "$DEVICE"

echo "=== Step-12: advisor report ==="
python src/print_step12_rwf_report.py \
  --project-root "$PROJECT_ROOT" \
  --results outputs/rwf_2000_fight_validation/results_summary.json \
  | tee outputs/rwf_2000_fight_validation/step12_report.txt

echo "Step-12 complete"
echo "- results: $PROJECT_ROOT/outputs/rwf_2000_fight_validation/results_summary.json"
echo "- report:  $PROJECT_ROOT/outputs/rwf_2000_fight_validation/step12_report.txt"

