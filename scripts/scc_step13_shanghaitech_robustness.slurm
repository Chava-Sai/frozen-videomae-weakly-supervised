#!/bin/bash
#SBATCH -J step13_shanghai
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o outputs/shanghaitech_robustness/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
bash scripts/run_step13_shanghaitech_robustness.sh ${PROJECT_ROOT} cuda

