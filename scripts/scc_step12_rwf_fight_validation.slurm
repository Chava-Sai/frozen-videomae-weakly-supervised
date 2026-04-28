#!/bin/bash
#SBATCH -J step12_rwf
#SBATCH -p cs585
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/rwf_2000_fight_validation/slurm-%j.out

set -euo pipefail

cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate
bash scripts/run_step12_rwf_fight_validation.sh /projectnb/cs585/students/saichava/IVC_Project cuda

