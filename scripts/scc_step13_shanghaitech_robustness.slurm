#!/bin/bash
#SBATCH -J step13_shanghai
#SBATCH -p cs585
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /projectnb/cs585/students/saichava/IVC_Project/outputs/shanghaitech_robustness/slurm-%j.out

set -euo pipefail

cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate
bash scripts/run_step13_shanghaitech_robustness.sh /projectnb/cs585/students/saichava/IVC_Project cuda

