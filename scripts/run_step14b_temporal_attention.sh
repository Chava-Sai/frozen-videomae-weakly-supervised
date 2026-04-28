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
mkdir -p outputs/step14_interpretability/step14b

python src/step14b_temporal_attention.py \
  --project-root "$PROJECT_ROOT" \
  --step14a-cases outputs/step14_interpretability/step14a/step14a_case_studies.json \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --xd-feature-manifest data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --attention-layer -1 \
  --max-cases 10 \
  --out-dir outputs/step14_interpretability/step14b \
  --device "$DEVICE"

echo "Step-14B complete"
echo "- report:  $PROJECT_ROOT/outputs/step14_interpretability/step14b/step14b_report.txt"
echo "- summary: $PROJECT_ROOT/outputs/step14_interpretability/step14b/step14b_summary.json"
echo "- plots:   $PROJECT_ROOT/outputs/step14_interpretability/step14b/plots"
echo "- tensors: $PROJECT_ROOT/outputs/step14_interpretability/step14b/attention_tensors"
