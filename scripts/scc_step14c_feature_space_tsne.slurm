#!/bin/bash -l
#SBATCH -J ivc_step14c
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/step14_interpretability/step14c/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="/projectnb/cs585/students/saichava/IVC_Project"
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p outputs/step14_interpretability/step14c
bash scripts/run_step14c_feature_space_tsne.sh "$PROJECT_ROOT" cuda
