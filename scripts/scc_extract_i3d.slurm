#!/bin/bash
#SBATCH --job-name=ivc_i3d_full
#SBATCH --output=outputs/ivc_i3d_full_%j.log
#SBATCH --error=outputs/ivc_i3d_full_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/path/to/IVC Project"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas tqdm opencv-python-headless torch torchvision pytorchvideo iopath fvcore av

bash scripts/run_full_i3d_extraction.sh "$PROJECT_ROOT" cuda
