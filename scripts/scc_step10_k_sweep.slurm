#!/bin/bash -l
#$ -P cs585
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=05:00:00
#$ -N step10_k_sweep
#$ -j y
#$ -o outputs/step10_k_sweep.$JOB_ID.log

set -euo pipefail
cd /projectnb/cs585/students/saichava/IVC_Project
source .venv/bin/activate

bash scripts/run_step10_k_sweep.sh /projectnb/cs585/students/saichava/IVC_Project cuda 30 64 32 1,3,5,10 0.55 1 5 0 2
