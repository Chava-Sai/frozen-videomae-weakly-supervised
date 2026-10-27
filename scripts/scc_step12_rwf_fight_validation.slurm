#!/bin/bash
#SBATCH -J step12_rwf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o outputs/rwf_2000_fight_validation/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
bash scripts/run_step12_rwf_fight_validation.sh ${PROJECT_ROOT} cuda

