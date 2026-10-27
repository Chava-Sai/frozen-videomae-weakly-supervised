#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=03:00:00
#$ -N step10_eval
#$ -j y
#$ -o outputs/step10_eval.$JOB_ID.log

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

bash scripts/run_step10_eval_and_report.sh ${PROJECT_ROOT} cuda 1,3,5,10 0.55 1 5 0 2
