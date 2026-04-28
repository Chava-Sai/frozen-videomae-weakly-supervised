#!/bin/bash
#SBATCH --job-name=ivc_rtfm_prog
#SBATCH --output=outputs/ivc_rtfm_prog_%j.log
#SBATCH --error=outputs/ivc_rtfm_prog_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

set -euo pipefail

PROJECT_ROOT="/projectnb/cs585/students/saichava/IVC_Project"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas tqdm opencv-python-headless torch torchvision pytorchvideo iopath fvcore av scikit-learn matplotlib

bash scripts/run_rtfm_progressive.sh "$PROJECT_ROOT" cuda 30 30 40 64 32 4

python src/print_rtfm_progressive_report.py \
  --project-root "$PROJECT_ROOT" \
  --results outputs/rtfm_progressive/results_summary.json \
  | tee outputs/rtfm_progressive/step8_report.txt
