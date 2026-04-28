# Frozen VideoMAE-B Representations for Weakly-Supervised Violence Detection and Temporal Localization

**BU CS585 — Image and Video Computing (Spring 2026)**  
Srinivasa Sai Chava · Yuxiang Liu · Shristhy Gupta · Vishakha Kumaresan · Anagha P Krishna · Boston University

---

### Presentation: https://canva.link/bqh75pyseq0y55p

---

## Overview

This repository contains the full pipeline for weakly-supervised surveillance violence detection using frozen VideoMAE-B features.  
We replace the commonly-used I3D backbone with a frozen ViT-B (VideoMAE-B, Kinetics-400) and build an RTFM-style stack with an event classifier head, TRN temporal refinement, and a boundary confidence head — all trained on UCF-Crime with weak (video-level) labels only.

---

## Results

### UCF-Crime Violence Subset (3-seed mean ± std)

| Metric | Score |
|--------|-------|
| AUC | **0.922 ± 0.010** |
| AP | **0.887 ± 0.014** |
| Macro-F1 | 0.255 ± 0.044 |
| Weighted-F1 | 0.719 ± 0.027 |
| mAP@0.3 | 0.034 ± 0.012 |
| mAP@0.5 | 0.012 ± 0.008 |
| mAP@0.7 | 0.006 ± 0.005 |

### Cross-Dataset Zero-Shot Transfer (seed-42 checkpoint)

| Dataset | AUC | AP | Class Metric | Notes |
|---------|-----|----|--------------|-------|
| UCF-Crime (ref.) | 0.927 | 0.868 | Macro-F1 0.303 | In-domain reference |
| XD-Violence | 0.836 | 0.885 | Overlap Macro-F1 0.361 | Transfer ratio 0.901 — strong binary transfer |
| RWF-2000 | 0.862 | 0.819 | Fight F1 0.246 | Ranking transfers; conservative fight recall |
| ShanghaiTech | 0.371 | 0.217 | binary only (FN=43/44) | Negative — domain shift |

---

## Repository Structure

```
.
├── src/                             # Core Python source code
│   ├── train_videomae_full.py       # Main training script (VideoMAE-B features)
│   ├── train_rtfm_baseline.py       # Step 4 — RTFM baseline
│   ├── train_rtfm_classifier.py     # Step 5 — + event classifier head
│   ├── train_rtfm_trn.py            # Step 6 — + TRN temporal refinement
│   ├── train_rtfm_trn_boundary.py   # Step 7 — + boundary head
│   ├── train_rtfm_progressive.py    # Step 8 — progressive training variant
│   ├── feature_dataset.py           # Dataset loader for cached features
│   ├── extract_videomae_features.py # VideoMAE-B feature extraction
│   ├── extract_i3d_features.py      # I3D feature extraction (baseline)
│   ├── eval_xd_zero_shot.py         # XD-Violence zero-shot evaluation
│   ├── eval_rwf_fight_validation.py # RWF-2000 fight validation
│   ├── eval_shanghaitech_robustness.py  # ShanghaiTech robustness check
│   ├── evaluate_ablation_checkpoint.py  # Checkpoint evaluator for ablations
│   ├── step14a_error_taxonomy.py    # Interpretability: error taxonomy
│   ├── step14b_temporal_attention.py    # Interpretability: TRN attention
│   ├── step14c_feature_space_tsne.py    # Interpretability: t-SNE
│   ├── step14d_boundary_precision.py    # Interpretability: boundary analysis
│   ├── step14e_cross_dataset_transfer_summary.py  # Transfer summary
│   ├── prepare_ucf_violence_manifest.py # UCF-Crime manifest builder
│   ├── prepare_xd_violence_manifest.py  # XD-Violence manifest builder
│   ├── prepare_rwf_2000_manifest.py     # RWF-2000 manifest builder
│   ├── prepare_shanghaitech_manifest.py # ShanghaiTech manifest builder
│   ├── plot_videomae_results.py     # Result plotting scripts
│   ├── plot_videomae_paper.py       # Paper figure generation
│   └── ...                          # Additional reporting utilities
│
├── scripts/                         # Shell scripts (local + SCC/SLURM)
│   ├── scc_train_rtfm.slurm         # SLURM: train RTFM baseline
│   ├── scc_train_rtfm_trn_boundary.slurm  # SLURM: train full model
│   ├── scc_step11_xd_zero_shot.slurm      # SLURM: XD-Violence eval
│   ├── scc_step12_rwf_fight_validation.slurm  # SLURM: RWF eval
│   ├── scc_step13_shanghaitech_robustness.slurm  # SLURM: ShanghaiTech eval
│   ├── scc_step14*.slurm            # SLURM: interpretability steps
│   ├── run_rtfm_trn_boundary.sh     # Local training runner
│   ├── run_step11_xd_zero_shot.sh   # Local XD eval runner
│   └── ...                          # Additional runners
│
├── scc_jobs/                        # SCC feature extraction jobs
│   ├── extract_videomae_full.sh     # Full VideoMAE-B extraction
│   └── extract_videomae_pilot.sh    # Pilot extraction (sanity check)
│
├── outputs_scc/                     # Results from SCC runs (JSONs, figures)
│   ├── videomae_rtfm/               # Per-seed training results
│   │   ├── seed_42/                 # results_summary.json, train_curves.png
│   │   ├── seed_123/
│   │   └── seed_456/
│   ├── step10_ablations/            # k-sweep and ablation results
│   ├── step14_interpretability/     # Interpretability analysis outputs
│   ├── xd_violence_zero_shot/       # XD-Violence results JSON
│   ├── rwf_2000_fight_validation/   # RWF-2000 results JSON
│   └── shanghaitech_robustness/     # ShanghaiTech results JSON
│
├── paper/                           # ACL-format paper (LaTeX)
│   ├── main_acl.tex                 # Final paper source
│   ├── references.bib               # Bibliography
│   └── figures/                     # All paper figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Chava-Sai/frozen-videomae-weakly-supervised.git
cd frozen-videomae-weakly-supervised
```

### 2. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

> **GPU required.** All training and feature extraction were run on a single NVIDIA A100 (40 GB) via BU SCC. A GPU with ≥16 GB VRAM is recommended.

---

## Data Preparation

### UCF-Crime

1. Download UCF-Crime from the [official page](https://www.crcv.ucf.edu/projects/real-world/).
2. Place videos under `data/ucf_crime/raw_videos/`.
3. Build the manifest:

```bash
python src/prepare_ucf_violence_manifest.py \
    --video-root data/ucf_crime/raw_videos \
    --out-dir    data/ucf_crime/manifests
```

### XD-Violence

```bash
python src/prepare_xd_violence_manifest.py \
    --video-root data/xd_violence/videos \
    --out-dir    data/xd_violence/manifests
```

### RWF-2000

```bash
python src/prepare_rwf_2000_manifest.py \
    --video-root data/rwf_2000/videos \
    --out-dir    data/rwf_2000/manifests
```

### ShanghaiTech

```bash
python src/prepare_shanghaitech_manifest.py \
    --video-root data/shanghaitech/videos \
    --out-dir    data/shanghaitech/manifests
```

---

## Feature Extraction

We use **frozen VideoMAE-B** (ViT-B, Kinetics-400 pre-trained) from HuggingFace:  
[`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)

Features are extracted as **16-frame non-overlapping segments**, producing tensors of shape `(T, 768)`.

```bash
# UCF-Crime (full extraction — ~1300 videos)
python src/extract_videomae_features.py \
    --manifest  data/ucf_crime/manifests/train_manifest.csv \
    --video-root data/ucf_crime/raw_videos \
    --out-dir   data/ucf_crime/features/videomae_b_16f \
    --segment-len 16 \
    --batch-size 8 \
    --device cuda
```

> On SCC, use `scc_jobs/extract_videomae_full.sh` with the provided SLURM script.

---

## Training

Train across 3 seeds for reproducibility:

```bash
for SEED in 42 123 456; do
    python src/train_videomae_full.py \
        --seed       $SEED \
        --feature-dir data/ucf_crime/features/videomae_b_16f \
        --manifest-dir data/ucf_crime/manifests \
        --output-dir  outputs/videomae_rtfm/seed_${SEED} \
        --epochs 40 \
        --batch-size 64 \
        --lr 1e-4 \
        --device cuda
done
```

**Key training settings:**

| Hyperparameter | Value |
|----------------|-------|
| Feature backbone | VideoMAE-B (frozen, D=768) |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 1e-4 (cosine anneal to 1e-6) |
| MIL top-k ratio | 12.5% of sequence (`k = ⌊0.125 × T⌋`) |
| Pseudo-label threshold | 0.3 |
| Loss weights λ₁/λ₂/λ₃ | 0.5 / 0.3 / 0.1 |
| Seeds | 42, 123, 456 |

> On SCC: `sbatch scripts/scc_train_rtfm_trn_boundary.slurm`

---

## Evaluation

### UCF-Crime (in-domain)

```bash
python src/evaluate_ablation_checkpoint.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --feature-dir data/ucf_crime/features/videomae_b_16f \
    --manifest-dir data/ucf_crime/manifests \
    --threshold 0.55 \
    --output-dir outputs/eval_seed42
```

### XD-Violence (zero-shot)

```bash
python src/eval_xd_zero_shot.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --feature-dir data/xd_violence/features/videomae_b_16f \
    --manifest-dir data/xd_violence/manifests \
    --output-dir outputs/xd_zero_shot
```

### RWF-2000 (fight validation)

```bash
python src/eval_rwf_fight_validation.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --feature-dir data/rwf_2000/features/videomae_b_16f \
    --manifest-dir data/rwf_2000/manifests \
    --output-dir outputs/rwf_fight_val
```

### ShanghaiTech (robustness)

```bash
python src/eval_shanghaitech_robustness.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --feature-dir data/shanghaitech/features/videomae_b_16f \
    --manifest-dir data/shanghaitech/manifests \
    --output-dir outputs/shanghaitech_robustness
```

**Locked decode settings used across all evaluations:**

| Parameter | Value |
|-----------|-------|
| Threshold | 0.55 |
| Smooth window | 1 |
| Min event length | 5 |
| Merge gap | 0 |
| Boundary radius | 2 |
| Boundary refine | enabled |

---

## Interpretability Analysis

All four interpretability analyses from the paper (Steps 14A–14D):

```bash
# 14A — Error taxonomy
python src/step14a_error_taxonomy.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --output-dir outputs/interpretability/14a

# 14B — Temporal attention (TRN)
python src/step14b_temporal_attention.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --output-dir outputs/interpretability/14b

# 14C — Feature-space t-SNE
python src/step14c_feature_space_tsne.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --output-dir outputs/interpretability/14c

# 14D — Boundary precision analysis
python src/step14d_boundary_precision.py \
    --checkpoint outputs/videomae_rtfm/seed_42/checkpoints/best.pt \
    --output-dir outputs/interpretability/14d
```

---

## Pre-trained Checkpoints

Checkpoints are not stored in this repository due to file size.

| Checkpoint | Seed | UCF AUC | UCF AP |
|------------|------|---------|--------|
| seed_42_best.pt | 42 | 0.927 | 0.868 |
| seed_123_best.pt | 123 | 0.909 | 0.897 |
| seed_456_best.pt | 456 | 0.931 | 0.896 |

> Contact the author or check the course submission for checkpoint files.

---

## Ablation Summary

| Configuration | AUC | AP | mAP@0.3 | mAP@0.5 |
|--------------|-----|-----|---------|---------|
| RTFM baseline (Step 5) | 0.919 | 0.841 | 0.042 | 0.013 |
| + TRN (Step 6) | 0.928 | 0.872 | 0.022 | 0.004 |
| + TRN + Boundary (Step 7) | 0.928 | 0.870 | 0.025 | 0.009 |
| **Final (k=1, Step 10)** | **0.927** | **0.868** | **0.034** | **0.012** |

---

## Paper

The full ACL-format paper is in `paper/main_acl.tex`.

**Title:** *Frozen VideoMAE-B Representations for Weakly-Supervised Violence Detection and Temporal Localization*  
**Author:** Srinivasa Sai Chava, Boston University  
**Course:** CS585 Image and Video Computing, Spring 2026

To compile:
```bash
cd paper
pdflatex main_acl.tex
bibtex main_acl
pdflatex main_acl.tex
pdflatex main_acl.tex
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{chava2026frozenvideomaevad,
  title   = {Frozen {VideoMAE-B} Representations for Weakly-Supervised
             Violence Detection and Temporal Localization},
  author  = {Chava, Srinivasa Sai},
  year    = {2026},
  note    = {BU CS585 Course Project},
  url     = {https://github.com/Chava-Sai/frozen-videomae-weakly-supervised}
}
```

---

## Acknowledgements

- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) — backbone feature extractor
- [RTFM](https://github.com/tianyu0207/RTFM) — weakly-supervised MIL framework
- [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) — primary training dataset
- BU Research Computing — SCC cluster (A100 GPUs)
