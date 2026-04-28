#!/bin/bash
#SBATCH -J step14e_transfer
#SBATCH -p cs585
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/step14_interpretability/step14e/slurm-%j.out

set -euo pipefail

cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate
mkdir -p outputs/step14_interpretability/step14e
bash scripts/run_step14e_cross_dataset_transfer.sh /projectnb/cs585/students/saichava/IVC_Project
