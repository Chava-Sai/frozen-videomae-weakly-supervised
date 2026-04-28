#!/bin/bash -l
#$ -P cs585
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l h_vmem=32G
#$ -N videomae_pilot
#$ -j y
#$ -o /projectnb/cs585/students/saichava/IVC_Project/logs/videomae_pilot.log
#$ -m bea
#$ -M saichava@bu.edu

# ── environment ──────────────────────────────────────────────────────────────
module load python3/3.10.12
module load cuda/11.8

PROJECT=/projectnb/cs585/students/saichava/IVC_Project
cd $PROJECT

# Hugging Face cache → scratch (avoids quota issues on home)
export HF_HOME=/scratch/saichava/hf_cache
mkdir -p $HF_HOME
mkdir -p $PROJECT/logs

# install / upgrade transformers once per job (cached after first run)
pip install --quiet --user "transformers>=4.36.0" "timm>=0.9.0" 2>/dev/null

# ── pilot run (13 videos) ─────────────────────────────────────────────────────
echo "=========================================="
echo "VideoMAE PILOT — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

python src/extract_videomae_features.py \
    --manifest data/ucf_crime/manifests/ucf_crime_manifest.csv \
    --video-root data/ucf_crime/videos \
    --out-dir data/ucf_crime/features/videomae_kinetics400_16f \
    --out-manifest data/ucf_crime/manifests/ucf_violence_features_videomae.csv \
    --hf-cache /scratch/saichava/hf_cache \
    --batch-size 4 \
    --mode pilot

echo "=========================================="
echo "PILOT DONE — $(date)"
echo "=========================================="
