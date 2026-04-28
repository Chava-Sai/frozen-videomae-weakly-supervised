#!/bin/bash
#SBATCH --job-name=ivc_step9_calib
#SBATCH --output=outputs/ivc_step9_calib_%j.log
#SBATCH --error=outputs/ivc_step9_calib_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/projectnb/cs585/students/saichava/IVC_Project"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas tqdm opencv-python-headless torch torchvision pytorchvideo iopath fvcore av scikit-learn matplotlib

bash scripts/run_step9_calibration.sh "$PROJECT_ROOT" cuda

python src/print_step9_calibration_report.py \
  --project-root "$PROJECT_ROOT" \
  --results outputs/step9_step7_calibration/results_summary.json \
  | tee outputs/step9_step7_calibration/step9_report.txt
