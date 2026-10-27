#!/bin/bash
#SBATCH -J step14d_boundary
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o outputs/step14_interpretability/step14d/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
mkdir -p outputs/step14_interpretability/step14d
bash scripts/run_step14d_boundary_precision.sh ${PROJECT_ROOT} cuda
