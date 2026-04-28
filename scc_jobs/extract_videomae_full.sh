#!/bin/bash -l
#$ -P cs585
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l h_vmem=48G
#$ -N videomae_full
#$ -j y
#$ -o /projectnb/cs585/students/saichava/IVC_Project/logs/videomae_full.log
#$ -m bea
#$ -M saichava@bu.edu

# ── environment ──────────────────────────────────────────────────────────────
module load python3/3.10.12
module load cuda/11.8

PROJECT=/projectnb/cs585/students/saichava/IVC_Project
cd $PROJECT

export HF_HOME=/scratch/saichava/hf_cache
mkdir -p $HF_HOME
mkdir -p $PROJECT/logs

pip install --quiet --user "transformers>=4.36.0" "timm>=0.9.0" 2>/dev/null

# ── full run (all 1300 videos) ────────────────────────────────────────────────
echo "=========================================="
echo "VideoMAE FULL — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

python src/extract_videomae_features.py \
    --manifest data/ucf_crime/manifests/ucf_crime_manifest.csv \
    --video-root data/ucf_crime/videos \
    --out-dir data/ucf_crime/features/videomae_kinetics400_16f \
    --out-manifest data/ucf_crime/manifests/ucf_violence_features_videomae.csv \
    --hf-cache /scratch/saichava/hf_cache \
    --batch-size 4 \
    --mode full

echo "=========================================="
echo "FULL EXTRACTION DONE — $(date)"
echo "Features at: data/ucf_crime/features/videomae_kinetics400_16f/"
echo "Manifest at: data/ucf_crime/manifests/ucf_violence_features_videomae.csv"
echo "=========================================="

# Quick sanity check
echo "--- Feature file count ---"
ls data/ucf_crime/features/videomae_kinetics400_16f/*.npz | wc -l

echo "--- Manifest row count (excl header) ---"
tail -n +2 data/ucf_crime/manifests/ucf_violence_features_videomae.csv | wc -l

echo "--- Sample shape check (first file) ---"
python - <<'PYEOF'
import numpy as np, glob, os
files = sorted(glob.glob("data/ucf_crime/features/videomae_kinetics400_16f/*.npz"))
if files:
    d = np.load(files[0])
    print(f"File: {os.path.basename(files[0])}")
    print(f"  features shape : {d['features'].shape}")   # expected (T, 768)
    print(f"  segment_starts : {d['segment_start_frames'].shape}")
    print(f"  feature_dim    : {d['features'].shape[-1]}")  # must be 768
else:
    print("ERROR: no .npz files found!")
PYEOF
