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
mkdir -p outputs/step14_interpretability/step14c

python src/step14c_feature_space_tsne.py \
  --project-root "$PROJECT_ROOT" \
  --step14a-error-table outputs/step14_interpretability/step14a/step14a_error_table.csv \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --xd-feature-manifest data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv \
  --threshold 0.55 \
  --infer-window 32 \
  --infer-stride 16 \
  --topk-ratio 0.125 \
  --per-dataset-samples 80 \
  --tsne-perplexity 30 \
  --tsne-iter 1500 \
  --out-dir outputs/step14_interpretability/step14c \
  --device "$DEVICE"

echo "Step-14C complete"
echo "- report:  $PROJECT_ROOT/outputs/step14_interpretability/step14c/step14c_report.txt"
echo "- summary: $PROJECT_ROOT/outputs/step14_interpretability/step14c/step14c_summary.json"
echo "- plots:   $PROJECT_ROOT/outputs/step14_interpretability/step14c/plots"
