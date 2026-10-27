#!/bin/bash
#SBATCH -J step14e_transfer
#SBATCH -p gpu
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o outputs/step14_interpretability/step14e/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
mkdir -p outputs/step14_interpretability/step14e
bash scripts/run_step14e_cross_dataset_transfer.sh ${PROJECT_ROOT}
