#!/bin/bash
#SBATCH -J step14d_boundary
#SBATCH -p cs585
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/step14_interpretability/step14d/slurm-%j.out

set -euo pipefail

cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate
mkdir -p outputs/step14_interpretability/step14d
bash scripts/run_step14d_boundary_precision.sh /projectnb/cs585/students/saichava/IVC_Project cuda
