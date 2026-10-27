#!/bin/bash
#SBATCH -J step14a_tax
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o outputs/step14_interpretability/step14a/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
bash scripts/run_step14a_error_taxonomy.sh ${PROJECT_ROOT} cuda

