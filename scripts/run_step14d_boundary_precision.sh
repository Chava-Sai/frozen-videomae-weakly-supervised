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
mkdir -p outputs/step14_interpretability/step14d

XD_ARGS=()
if compgen -G "data/xd_violence/annotations/temporal_segments/*.json" > /dev/null; then
  XD_ARGS+=(--include-xd)
  echo "Step-14D: XD temporal annotations found, including XD in boundary analysis."
else
  echo "Step-14D: XD temporal annotations not found, running UCF-only boundary analysis."
fi

python src/step14d_boundary_precision.py \
  --project-root "$PROJECT_ROOT" \
  --model-kind step7_boundary \
  --checkpoint outputs/step10_ablations/k1/train/checkpoints/best.pt \
  --ucf-split test \
  --xd-split test \
  --threshold 0.55 \
  --smooth-window 1 \
  --min-event-len 5 \
  --merge-gap 0 \
  --boundary-radius 2 \
  --boundary-refine \
  --tiou-thresholds 0.3,0.5,0.7 \
  --boundary-peak-tol 2 \
  --boundary-profile-window 8 \
  --qual-videos 5 \
  --out-dir outputs/step14_interpretability/step14d \
  --device "$DEVICE" \
  "${XD_ARGS[@]}"

echo "Step-14D complete"
echo "- report:  $PROJECT_ROOT/outputs/step14_interpretability/step14d/step14d_report.txt"
echo "- summary: $PROJECT_ROOT/outputs/step14_interpretability/step14d/step14d_summary.json"
echo "- plots:   $PROJECT_ROOT/outputs/step14_interpretability/step14d/plots"
echo "- qual:    $PROJECT_ROOT/outputs/step14_interpretability/step14d/qualitative"
