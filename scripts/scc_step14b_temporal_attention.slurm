#!/bin/bash -l
#SBATCH -J ivc_step14b
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/step14_interpretability/step14b/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="/projectnb/cs585/students/saichava/IVC_Project"
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p outputs/step14_interpretability/step14b
bash scripts/run_step14b_temporal_attention.sh "$PROJECT_ROOT" cuda
