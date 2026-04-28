#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
DEVICE="${2:-cuda}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-64}"
TARGET_SEGMENTS="${5:-32}"
K_LIST="${6:-1,3,5,10}"

# Fixed Step-9 decoding settings for fair ablations
THRESHOLD="${7:-0.55}"
SMOOTH_WINDOW="${8:-1}"
MIN_EVENT_LEN="${9:-5}"
MERGE_GAP="${10:-0}"
BOUNDARY_RADIUS="${11:-2}"

cd "$PROJECT_ROOT"
mkdir -p outputs/step10_ablations

IFS=',' read -r -a K_VALUES <<< "$K_LIST"

for K in "${K_VALUES[@]}"; do
  K_TRIMMED="$(echo "$K" | xargs)"
  OUT_BASE="outputs/step10_ablations/k${K_TRIMMED}"
  TRAIN_OUT="${OUT_BASE}/train"

  mkdir -p "$OUT_BASE"

  echo "\n=== Step-10 k-sweep: training k=${K_TRIMMED} ==="
  python -u src/train_rtfm_trn_boundary.py \
    --project-root "$PROJECT_ROOT" \
    --feature-manifest data/ucf_crime/manifests/ucf_violence_features_i3d.csv \
    --master-manifest data/ucf_crime/manifests/ucf_violence_master.csv \
    --temporal-root data/ucf_crime/annotations/temporal_segments \
    --init-ckpt outputs/rtfm_trn/checkpoints/best.pt \
    --output-dir "$TRAIN_OUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --target-segments "$TARGET_SEGMENTS" \
    --pseudo-topk "$K_TRIMMED" \
    --balanced-sampler \
    --checkpoint-metric val_macro_f1 \
    --device "$DEVICE" | tee "${OUT_BASE}/train.log"

  echo "=== Step-10 k-sweep: fixed-decoding eval k=${K_TRIMMED} ==="
  python -u src/evaluate_ablation_checkpoint.py \
    --project-root "$PROJECT_ROOT" \
    --model-kind step7_boundary \
    --checkpoint "${TRAIN_OUT}/checkpoints/best.pt" \
    --output-json "${OUT_BASE}/eval_fixed.json" \
    --threshold "$THRESHOLD" \
    --smooth-window "$SMOOTH_WINDOW" \
    --min-event-len "$MIN_EVENT_LEN" \
    --merge-gap "$MERGE_GAP" \
    --boundary-radius "$BOUNDARY_RADIUS" \
    --boundary-refine \
    --device "$DEVICE" | tee "${OUT_BASE}/eval.log"

done

python -u src/print_step10_ablation_report.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir outputs/step10_ablations \
  --k-values "$K_LIST" | tee outputs/step10_ablations/step10_report.txt
