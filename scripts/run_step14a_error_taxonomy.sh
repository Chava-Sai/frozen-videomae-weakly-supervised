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
mkdir -p outputs/step14_interpretability/step14a

python src/step14a_error_taxonomy.py \
  --project-root "$PROJECT_ROOT" \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --ucf-split test \
  --xd-split test \
  --rwf-split val \
  --sh-split test \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --out-dir outputs/step14_interpretability/step14a \
  --device "$DEVICE"

echo "Step-14A complete"
echo "- report:  $PROJECT_ROOT/outputs/step14_interpretability/step14a/step14a_report.txt"
echo "- table:   $PROJECT_ROOT/outputs/step14_interpretability/step14a/step14a_error_table.csv"
echo "- cases:   $PROJECT_ROOT/outputs/step14_interpretability/step14a/step14a_case_studies.json"

