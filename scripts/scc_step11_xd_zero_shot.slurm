#!/bin/bash -l
#$ -l h_rt=48:00:00
#$ -N step11_xd_zero_shot
#$ -pe omp 4
#$ -l gpu=1
#$ -l gpus=1
#$ -j y
#$ -o step11_xd_zero_shot.$JOB_ID.log

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PROJECT_ROOT/.venv/bin/activate"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

bash scripts/run_step11_xd_zero_shot.sh ${PROJECT_ROOT} cuda

