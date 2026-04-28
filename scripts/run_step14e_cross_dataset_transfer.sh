#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <project_root>"
  echo "Example: $0 /projectnb/cs585/students/saichava/IVC_Project"
  exit 1
fi

PROJECT_ROOT="$1"

cd "$PROJECT_ROOT"
mkdir -p outputs/step14_interpretability/step14e

python src/step14e_cross_dataset_transfer_summary.py \
  --project-root "$PROJECT_ROOT" \
  --ucf-json outputs/step10_ablations/k1/eval_fixed.json \
  --xd-json outputs/xd_violence_zero_shot/results_summary.json \
  --rwf-json outputs/rwf_2000_fight_validation/results_summary.json \
  --sh-json outputs/shanghaitech_robustness/results_summary.json \
  --step14a-json outputs/step14_interpretability/step14a/step14a_taxonomy_counts.json \
  --step14b-json outputs/step14_interpretability/step14b/step14b_summary.json \
  --step14c-json outputs/step14_interpretability/step14c/step14c_summary.json \
  --step14d-json outputs/step14_interpretability/step14d/step14d_summary.json \
  --out-dir outputs/step14_interpretability/step14e

echo "Step-14E complete"
echo "- report:   $PROJECT_ROOT/outputs/step14_interpretability/step14e/step14e_report.txt"
echo "- summary:  $PROJECT_ROOT/outputs/step14_interpretability/step14e/step14e_summary.json"
echo "- table_md: $PROJECT_ROOT/outputs/step14_interpretability/step14e/step14e_cross_dataset_table.md"
echo "- table_csv:$PROJECT_ROOT/outputs/step14_interpretability/step14e/step14e_cross_dataset_table.csv"
