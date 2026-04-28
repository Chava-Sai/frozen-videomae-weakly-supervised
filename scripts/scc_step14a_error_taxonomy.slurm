#!/bin/bash
#SBATCH -J step14a_tax
#SBATCH -p cs585
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/step14_interpretability/step14a/slurm-%j.out

set -euo pipefail

cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate
bash scripts/run_step14a_error_taxonomy.sh /projectnb/cs585/students/saichava/IVC_Project cuda

