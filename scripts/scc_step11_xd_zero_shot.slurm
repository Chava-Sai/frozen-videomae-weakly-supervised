#!/bin/bash -l
#$ -P cs585
#$ -l h_rt=48:00:00
#$ -N step11_xd_zero_shot
#$ -pe omp 4
#$ -l gpu=1
#$ -l gpus=1
#$ -j y
#$ -o step11_xd_zero_shot.$JOB_ID.log

source /projectnb/cs585/students/saichava/IVC_Project/.venv/bin/activate
cd /projectnb/cs585/students/saichava/IVC_Project

bash scripts/run_step11_xd_zero_shot.sh /projectnb/cs585/students/saichava/IVC_Project cuda

