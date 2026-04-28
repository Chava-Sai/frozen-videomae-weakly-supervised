#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"
K_LIST="${3:-1,3,5,10}"

# Fixed Step-9 decoding settings for fair ablations
THRESHOLD="${4:-0.55}"
SMOOTH_WINDOW="${5:-1}"
MIN_EVENT_LEN="${6:-5}"
MERGE_GAP="${7:-0}"
BOUNDARY_RADIUS="${8:-2}"

cd "$PROJECT_ROOT"
mkdir -p outputs/step10_ablations

COMMON_ARGS=(
  --project-root "$PROJECT_ROOT"
  --threshold "$THRESHOLD"
  --smooth-window "$SMOOTH_WINDOW"
  --min-event-len "$MIN_EVENT_LEN"
  --merge-gap "$MERGE_GAP"
  --boundary-radius "$BOUNDARY_RADIUS"
  --device "$DEVICE"
)

echo "=== Step-10 eval: Step5 (TRN off / boundary off) ==="
python -u src/evaluate_ablation_checkpoint.py \
  "${COMMON_ARGS[@]}" \
  --model-kind step5_classifier \
  --checkpoint outputs/rtfm_classifier/checkpoints/best.pt \
  --no-boundary-refine \
  --output-json outputs/step10_ablations/eval_step5_fixed.json | tee outputs/step10_ablations/eval_step5_fixed.log

echo "=== Step-10 eval: Step6 (TRN on / boundary off) ==="
python -u src/evaluate_ablation_checkpoint.py \
  "${COMMON_ARGS[@]}" \
  --model-kind step6_trn \
  --checkpoint outputs/rtfm_trn/checkpoints/best.pt \
  --no-boundary-refine \
  --output-json outputs/step10_ablations/eval_step6_fixed.json | tee outputs/step10_ablations/eval_step6_fixed.log

echo "=== Step-10 eval: Step7 boundary ON ==="
python -u src/evaluate_ablation_checkpoint.py \
  "${COMMON_ARGS[@]}" \
  --model-kind step7_boundary \
  --checkpoint outputs/rtfm_trn_boundary/checkpoints/best.pt \
  --boundary-refine \
  --output-json outputs/step10_ablations/eval_step7_boundary_on_fixed.json | tee outputs/step10_ablations/eval_step7_boundary_on_fixed.log

echo "=== Step-10 eval: Step7 boundary OFF (decode ablation) ==="
python -u src/evaluate_ablation_checkpoint.py \
  "${COMMON_ARGS[@]}" \
  --model-kind step7_boundary \
  --checkpoint outputs/rtfm_trn_boundary/checkpoints/best.pt \
  --no-boundary-refine \
  --output-json outputs/step10_ablations/eval_step7_boundary_off_fixed.json | tee outputs/step10_ablations/eval_step7_boundary_off_fixed.log

echo "=== Step-10 eval: Step8 progressive ==="
python -u src/evaluate_ablation_checkpoint.py \
  "${COMMON_ARGS[@]}" \
  --model-kind step8_progressive \
  --checkpoint outputs/rtfm_progressive/checkpoints/best.pt \
  --boundary-refine \
  --output-json outputs/step10_ablations/eval_step8_progressive_fixed.json | tee outputs/step10_ablations/eval_step8_progressive_fixed.log

IFS=',' read -r -a K_VALUES <<< "$K_LIST"
for K in "${K_VALUES[@]}"; do
  K_TRIMMED="$(echo "$K" | xargs)"
  CKPT="outputs/step10_ablations/k${K_TRIMMED}/train/checkpoints/best.pt"
  OUT_JSON="outputs/step10_ablations/k${K_TRIMMED}/eval_fixed.json"
  OUT_LOG="outputs/step10_ablations/k${K_TRIMMED}/eval_fixed.log"

  if [[ -f "$CKPT" ]]; then
    echo "=== Step-10 eval: k=${K_TRIMMED} ==="
    mkdir -p "outputs/step10_ablations/k${K_TRIMMED}"
    python -u src/evaluate_ablation_checkpoint.py \
      "${COMMON_ARGS[@]}" \
      --model-kind step7_boundary \
      --checkpoint "$CKPT" \
      --boundary-refine \
      --output-json "$OUT_JSON" | tee "$OUT_LOG"
  else
    echo "Skipping k=${K_TRIMMED} (missing checkpoint: $CKPT)"
  fi

done

python -u src/print_step10_ablation_report.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir outputs/step10_ablations \
  --k-values "$K_LIST" | tee outputs/step10_ablations/step10_report.txt
