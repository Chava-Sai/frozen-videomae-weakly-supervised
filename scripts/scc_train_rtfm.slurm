#!/bin/bash
#SBATCH --job-name=ivc_rtfm
#SBATCH --output=outputs/ivc_rtfm_%j.log
#SBATCH --error=outputs/ivc_rtfm_%j.err
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

bash scripts/run_rtfm_baseline.sh "$PROJECT_ROOT" cuda 30 64 32
