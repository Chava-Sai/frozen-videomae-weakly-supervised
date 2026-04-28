"""
draw_system_design.py
Generates a dark-theme, publication-quality system-design PNG for the IVC paper.
Style modelled on the professional system-architecture diagrams used in ML papers.

Output: outputs_scc/videomae_rtfm/figures/system_design.png  (dpi=200, ~4400×2600 px)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib.patheffects as pe
import numpy as np, os

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f0f0f"
SEC_BG  = "#191919"

# Section accent colours
AC1 = "#3b82f6"   # blue   — ① Feature Extraction
AC2 = "#8b5cf6"   # violet — ② Model Architecture
AC3 = "#10b981"   # emerald— ③ Training & Evaluation

# Box themes: (fill, border)
BLUE   = ("#0d2240", "#2563eb")
INDIGO = ("#151040", "#6366f1")
VIOLET = ("#1a0e3f", "#7c3aed")
GREEN  = ("#052e1a", "#059669")
TEAL   = ("#02262e", "#0891b2")
AMBER  = ("#271500", "#d97706")
ORANGE = ("#2e1200", "#ea580c")
RED    = ("#2a0c0c", "#dc2626")
GRAY   = ("#1e1e1e", "#6b7280")
CYAN   = ("#012830", "#22d3ee")

W = "#ffffff"
LG = "#cbd5e1"    # light grey subtitle
DG = "#64748b"    # dark  grey annotation

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────────────────────────────────────
FW, FH = 24, 14          # inches
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect("equal"); ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, theme, title, subs=(), ft=8.8, fs=7.2, zb=4):
    """Rounded rectangle with bold title + italic subtitle lines."""
    fill, bord = theme
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fill, edgecolor=bord,
        linewidth=1.6, zorder=zb))
    n = len(subs)
    top = y + h/2 + (n * 0.085) * 0.5
    ax.text(x + w/2, top, title,
            color=W, fontsize=ft, fontweight="bold",
            ha="center", va="center", zorder=zb+1,
            fontfamily="DejaVu Sans")
    for i, s in enumerate(subs):
        ax.text(x + w/2, top - 0.17*(i+1), s,
                color=LG, fontsize=fs,
                ha="center", va="center", zorder=zb+1)


def cylinder(x, y, w, h, theme, title, subs=()):
    """Data-store cylinder."""
    fill, bord = theme
    ew = w; eh = h * 0.22
    # Body
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h - eh/2,
        boxstyle="square,pad=0",
        facecolor=fill, edgecolor=bord, linewidth=1.6, zorder=4))
    # Top ellipse
    ax.add_patch(Ellipse((x+w/2, y+h-eh/2), ew, eh,
        facecolor=bord, edgecolor=bord, linewidth=1.2, zorder=5))
    # Bottom ellipse
    ax.add_patch(Ellipse((x+w/2, y), ew, eh,
        facecolor=fill, edgecolor=bord, linewidth=1.2, zorder=4))
    mid = y + (h - eh/2)/2
    ax.text(x+w/2, mid + len(subs)*0.07, title,
            color=W, fontsize=8.5, fontweight="bold",
            ha="center", va="center", zorder=6)
    for i, s in enumerate(subs):
        ax.text(x+w/2, mid - 0.16*(i+1) + len(subs)*0.07, s,
                color=LG, fontsize=7,
                ha="center", va="center", zorder=6)


def section(x, y, w, h, color, num, title):
    """Section container with circled number."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0",
        facecolor=SEC_BG, edgecolor=color,
        linewidth=2.2, zorder=1))
    # Badge circle
    cx2, cy2 = x + 0.38, y + h - 0.34
    ax.add_patch(plt.Circle((cx2, cy2), 0.24, color=color, zorder=3))
    ax.text(cx2, cy2, num, color=W, fontsize=9.5, fontweight="bold",
            ha="center", va="center", zorder=4)
    ax.text(cx2 + 0.38, cy2, f"  {title}",
            color=color, fontsize=9.5, fontweight="bold",
            ha="left", va="center", zorder=4)


def arrow(x1, y1, x2, y2, color=W, lw=1.4, label="", rad=0.0, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=11),
        zorder=6)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label,
                color=color, fontsize=6.8, ha="center", va="bottom",
                zorder=7,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.8, pad=1))


def dashed(x1, y1, x2, y2, color=DG, lw=1.2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            linestyle="dashed",
            connectionstyle="arc3,rad=0",
            mutation_scale=10),
        zorder=5)
    if label:
        ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label,
                color=color, fontsize=6.5, ha="left", va="center",
                style="italic", zorder=6)


def hline(x1, x2, y, color=DG, lw=1.0):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=2)


def footnote(x, y, icon, text, color=DG):
    ax.text(x, y, icon + "  " + text, color=color,
            fontsize=7.2, ha="left", va="center", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# ① FEATURE EXTRACTION  (top, full width)
# ─────────────────────────────────────────────────────────────────────────────
S1X, S1Y, S1W, S1H = 0.25, 8.55, 23.5, 5.15
section(S1X, S1Y, S1W, S1H, AC1, "①", "FEATURE EXTRACTION PIPELINE — One-Time Preprocessing")

# Dataset box
rbox(0.5, 10.2, 2.5, 2.1, BLUE, "UCF-Crime Dataset",
     ("Sultani et al. CVPR 2018", "1,300 videos total", "6 violence categories", "Video-level weak labels"))

# Frame sampler
rbox(3.4, 10.2, 2.2, 2.1, INDIGO, "Frame Sampler",
     ("16-frame segments", "Non-overlapping", "Tail segment dropped", "T segments per video"))

arrow(3.0, 11.25, 3.4, 11.25, AC1)   # Dataset → Sampler

# Split arrow Y-junction
ax.plot([5.6, 6.1], [11.25, 11.25], color=AC1, lw=1.6, zorder=6)
ax.plot([6.1, 6.1], [11.25, 9.2],   color=AC1, lw=1.6, zorder=6)   # down to I3D row
ax.plot([6.1, 6.1], [11.25, 12.4],  color=AC1, lw=1.6, zorder=6)   # up to VideoMAE row

# ── I3D branch (lower) ───────────────────────────────────────────────────────
# Small label
ax.text(6.3, 10.65, "Step-7 Baseline", color=ORANGE[1],
        fontsize=7.5, style="italic", va="center", zorder=5)

rbox(6.5, 8.9, 3.0, 2.0, ORANGE, "I3D Backbone",
     ("Frozen · Kinetics-400", "Two-stream RGB + Flow", "Feature dim D = 2048", "Standard VAD backbone"))

arrow(6.1, 9.2, 6.5, 9.75, ORANGE[1])

rbox(9.85, 8.9, 2.2, 2.0, AMBER, "I3D Features",
     ("D = 2048 per segment", "f_t ∈ ℝ²⁰⁴⁸", "1,300 .npz files", "Step-7 cache"))

arrow(9.5, 9.9, 9.85, 9.9, ORANGE[1])

# ── VideoMAE-B branch (upper) ─────────────────────────────────────────────────
ax.text(6.3, 12.55, "Our Contribution ★", color=BLUE[1],
        fontsize=7.5, fontweight="bold", style="italic", va="center", zorder=5)

rbox(6.5, 11.6, 3.4, 2.0, BLUE, "VideoMAE-B Backbone",
     ("Frozen ViT-B · Kinetics-400", "MCG-NJU checkpoint", "Masked autoencoder pre-training", "BU SCC · NVIDIA A100"))

arrow(6.1, 12.4, 6.5, 12.6, AC1)

rbox(10.25, 11.6, 2.2, 2.0, TEAL, "VideoMAE-B Features",
     ("D = 768 per segment", "f_t ∈ ℝ⁷⁶⁸", "1,300 .npz files", "494.6 s · 0.38 s/vid"))

arrow(9.9, 12.6, 10.25, 12.6, AC1)

# ── Merge into shared cache ───────────────────────────────────────────────────
ax.plot([12.45, 13.0], [12.6, 12.6], color=AC1, lw=1.6, zorder=6)
ax.plot([12.05, 13.0], [9.9,  9.9],  color=ORANGE[1], lw=1.6, zorder=6)
ax.plot([13.0,  13.0], [9.9, 12.6],  color=DG, lw=1.2, ls="--", zorder=5)
ax.plot([13.0,  13.3], [11.25, 11.25], color=AC1, lw=1.6, zorder=6)
arrow(13.3, 11.25, 13.7, 11.25, AC1)

# Feature cache cylinder
cylinder(13.7, 9.45, 2.6, 3.3, TEAL,
         "Feature Cache",
         (".npz · 1,300 files per backbone",
          "Cached once · reused every run",
          "Resume-safe · skip existing"))

# ── Comparison callout ────────────────────────────────────────────────────────
rbox(16.7, 10.3, 4.0, 2.2, GRAY, "Feature Comparison",
     ("I3D:      D=2048 · optical flow · 2017",
      "VideoMAE: D=768  · masked AE  · 2022",
      "Same architecture — only backbone swaps",
      "Dimension ↓ 63%  →  less overfitting"))

arrow(16.3, 11.25, 16.7, 11.4, DG)

# ─────────────────────────────────────────────────────────────────────────────
# ② MODEL ARCHITECTURE  (bottom-left)
# ─────────────────────────────────────────────────────────────────────────────
S2X, S2Y, S2W, S2H = 0.25, 0.25, 11.4, 8.05
section(S2X, S2Y, S2W, S2H, AC2, "②", "MODEL ARCHITECTURE — 1.9M Trainable / 86M Frozen")

# Input from cache
rbox(0.5, 6.8, 1.9, 1.5, VIOLET, "Input",
     ("Feature f_t ∈ ℝ⁷⁶⁸", "per segment t", "VideoMAE-B cache"))

dashed(14.0, 9.45, 1.4, 7.55, color=AC2, label="")
arrow(2.4, 7.55, 2.75, 7.55, AC2)

# MIL Anomaly Scorer
rbox(2.75, 6.7, 2.6, 1.7, VIOLET, "MIL Anomaly Scorer",
     ("s_t = σ(W₃ Dropout(ReLU(W₂ ReLU(W₁f_t))))",
      "Video score: s_vid = max_t s_t",
      "ℒ_MIL  ranking loss · pairs anom/norm",
      "1.1M params"))

arrow(5.35, 7.55, 5.7, 7.55, AC2)

# Event Classifier
rbox(5.7, 6.7, 2.5, 1.7, VIOLET, "Event Classifier",
     ("7-class softmax (6 violence + normal)",
      "Top-k pseudo-positive segments",
      "Conf. threshold τ=0.30 filter",
      "0.3M params"))

arrow(8.2, 7.55, 8.2, 6.35, AC2)  # down

# TRN
rbox(5.7, 4.55, 2.5, 1.7, INDIGO, "Temporal Refinement (TRN)",
     ("Transformer encoder over {f_t + PE_t}",
      "Multi-head self-attention",
      "Refined scores s̃_t and h̃_t",
      "0.4M params"))

arrow(8.2, 4.55, 8.2, 3.7, AC2)   # down

# Boundary Head
rbox(5.7, 2.4, 2.5, 1.7, INDIGO, "Boundary Head",
     ("b_t = σ(W_b[h̃_t ; h̃_{t+1} ; |s̃_t − s̃_{t+1}|])",
      "Boundary consistency loss ℒ_bnd",
      "Event start / end confidence",
      "0.1M params"))

arrow(8.2, 5.7, 8.2, 4.55, AC2, label="") # TRN ← Classifier down
# Actually connect classifier → TRN with left arrow
arrow(5.7, 5.45, 2.75+2.6/2, 5.45, AC2, rad=0)   # back left to MIL?
# Actually the flow is sequential: let me use a cleaner layout

# Loss objective box
rbox(0.5, 4.45, 5.0, 2.1, GRAY, "Full Training Objective",
     ("ℒ = ℒ_MIL + λ₁ℒ_cls + λ₂ℒ_bnd + λ₃ℒ_smooth",
      "ℒ_smooth = Σ(s̃_t − s̃_{t−1})²",
      "λ₁=0.5 · λ₂=0.3 · λ₃=0.1",
      "AdamW · lr=1e-4 · dropout=0.5"))

arrow(2.75+2.6/2, 6.7, 2.75+2.6/2, 6.55, AC2)  # scorer → loss (implicit)

# Training recipe (4 pills, bottom)
pill_y = 0.55; pill_h = 1.6; pill_w = 2.55; gap = 0.18

recipes = [
    (VIOLET, "① Inv-Freq Weights",
     ("w_c = N/(C·n_c)", "Normal: w≈0.06", "Shooting: w≈1.83", "Counters 10:1 gap")),
    (VIOLET, "② Conf. Threshold",
     ("τ = 0.30", "Filter noisy pseudo-", "positive segments", "in early epochs")),
    (VIOLET, "③ Cosine Anneal LR",
     ("η: 1e-4 → 1e-6", "T_max = 40 epochs", "No warm restarts", "No step-decay spikes")),
    (VIOLET, "④ 3-Seed Training",
     ("Seeds: 42 · 123 · 456", "40 epochs each", "Report mean ± std", "Reproducibility")),
]
for i, (th, ti, su) in enumerate(recipes):
    px = 0.5 + i*(pill_w + gap)
    rbox(px, pill_y, pill_w, pill_h, th, ti, su, ft=7.8, fs=6.5)

# Section label at bottom
ax.text(S2X + S2W/2, 0.15, "Total trainable: 1.9M  ·  VideoMAE-B frozen: 86M  ·  Dropout 0.5 · WD 1e-4  ·  Batch 32  ·  PyTorch 2.1",
        color=DG, fontsize=7, ha="center", va="center", zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# ③ TRAINING & EVALUATION  (bottom-right)
# ─────────────────────────────────────────────────────────────────────────────
S3X, S3Y, S3W, S3H = 11.9, 0.25, 11.85, 8.05
section(S3X, S3Y, S3W, S3H, AC3, "③", "INFERENCE & EVALUATION — UCF-Crime Violence Subset")

# Input bridge from ② model
rbox(12.1, 6.75, 2.2, 1.55, GREEN, "Trained Model",
     ("Best checkpoint (seed 123)", "AUC = 93.4% peak", "40-epoch training"))

dashed(8.2, 2.4+1.7/2, 12.1, 7.2, color=AC3)

# Inference pipeline
rbox(14.6, 6.75, 2.5, 1.55, GREEN, "Inference Pipeline",
     ("Forward pass per segment", "s̃_t · p_t · b_t outputs", "Temporal smoothing"))

arrow(14.3, 7.52, 14.6, 7.52, AC3)

# Post-processing
rbox(17.4, 6.75, 2.5, 1.55, TEAL, "Post-Processing",
     ("Threshold candidate segs.", "Merge adjacent positives", "Boundary peak refinement"))

arrow(17.1, 7.52, 17.4, 7.52, AC3)

rbox(20.2, 6.75, 1.65, 1.55, TEAL, "Event Tuples",
     ("(t_start, t_end,", "class, confidence)", "per video"))

arrow(19.9, 7.52, 20.2, 7.52, AC3)

# Metrics row
ax.text(12.1, 6.4, "─────  Evaluation Metrics  ─────", color=AC3,
        fontsize=8, ha="left", va="center", fontweight="bold", zorder=4)

metric_boxes = [
    (GREEN,  "Binary Detection",
     ("AUC  (ROC curve)", "AP   (PR  curve)", "Video-level scores")),
    (TEAL,   "Multi-Class F1",
     ("Macro-F1  (6 classes)", "Weighted-F1", "Per-category breakdown")),
    (CYAN,   "Temporal Localization",
     ("mAP @ tIoU 0.3 / 0.5 / 0.7", "Segment-level detection", "Boundary precision")),
]
mw = 3.6; mg = 0.15
for i, (th, ti, su) in enumerate(metric_boxes):
    mx = 12.1 + i*(mw + mg)
    rbox(mx, 4.55, mw, 1.6, th, ti, su, ft=8, fs=6.8)
    arrow(mx + mw/2, 6.75, mx + mw/2, 6.17, AC3)

# Results comparison table
ax.text(12.1, 4.2, "─────  Results: I3D Baseline  vs  VideoMAE-B (3 Seeds, Mean ± Std)  ─────",
        color=AC3, fontsize=8, ha="left", va="center", fontweight="bold", zorder=4)

# Header row
rbox(12.1, 3.35, 2.0, 0.7, GRAY, "Metric", (), ft=8)
rbox(14.2, 3.35, 3.2, 0.7, (RED[0], RED[1]), "[I3D] I3D Baseline", (), ft=8)
rbox(17.5, 3.35, 3.8, 0.7, (BLUE[0], BLUE[1]), "[VM] VideoMAE-B (mean +/-std)", (), ft=7.5)
rbox(21.4, 3.35, 2.2, 0.7, (GREEN[0], GREEN[1]), "[+] Gain", (), ft=8)

rows = [
    ("Binary AUC",    "87.4%",         "92.2% ± 1.3%",       "+4.8 pp"),
    ("Binary AP",     "82.1%",         "88.7% ± 1.8%",       "+8%"),
    ("Macro-F1",      "19.8%",         "25.5% ± 2.7%",       "+28%"),
    ("mAP @ IoU 0.3", "0.009",         "0.058 ± 0.012",      "× 6.4 ↑↑"),
    ("mAP @ IoU 0.5", "0.004",         "0.038 ± 0.008",      "× 9.4 ↑↑"),
    ("mAP @ IoU 0.7", "0.001",         "0.022 ± 0.010",      "× 21.7 ↑↑↑"),
]
rh = 0.42
for i, (met, i3d, vm, gn) in enumerate(rows):
    ry = 3.35 - (i+1)*(rh + 0.03)
    i3d_fill = "#1a0808" if "mAP" in met else "#180808"
    vm_fill  = "#081a10"
    gn_fill  = "#061a0a"
    ax.add_patch(FancyBboxPatch((12.1, ry), 2.0, rh, boxstyle="square,pad=0",
        facecolor="#121212", edgecolor="#2a2a2a", lw=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((14.2, ry), 3.2, rh, boxstyle="square,pad=0",
        facecolor=i3d_fill, edgecolor="#3a1010", lw=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((17.5, ry), 3.8, rh, boxstyle="square,pad=0",
        facecolor=vm_fill, edgecolor="#0a2f18", lw=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((21.4, ry), 2.2, rh, boxstyle="square,pad=0",
        facecolor=gn_fill, edgecolor="#0a2510", lw=0.8, zorder=3))
    ax.text(13.1, ry+rh/2, met, color=LG, fontsize=7.5, ha="center", va="center", zorder=5, fontweight="bold")
    ax.text(15.8, ry+rh/2, i3d, color="#f87171", fontsize=7.5, ha="center", va="center", zorder=5)
    ax.text(19.4, ry+rh/2, vm,  color="#60a5fa", fontsize=8, ha="center", va="center", zorder=5, fontweight="bold")
    ax.text(22.5, ry+rh/2, gn,  color="#34d399", fontsize=8, ha="center", va="center", zorder=5, fontweight="bold")

# Publication outputs row
pub_y = 0.55; pub_h = 1.4
pub_items = [
    (GREEN,  "6 Figures (matplotlib)",
     ("Training Dynamics · Dumbbell", "mAP Curve · Reproducibility", "Per-Class · Radar Chart")),
    (TEAL,   "ACM MM Workshop 2026",
     ("Deadline: July 16, 2026", "6-page ACM SIGCONF", "Current results sufficient")),
    (CYAN,   "arXiv Preprint",
     ("Submit immediately", "Establishes priority", "Public timestamp")),
    (GREEN,  "CVPR 2027 (Target)",
     ("Deadline: Nov 2026", "Add cross-dataset eval", "VideoMAE-Large backbone")),
]
pw = 2.75; pg = 0.17
for i, (th, ti, su) in enumerate(pub_items):
    px = 12.1 + i*(pw + pg)
    rbox(px, pub_y, pw, pub_h, th, ti, su, ft=7.8, fs=6.5)

ax.text(S3X + S3W/2, 0.15,
        "Embeddings: VideoMAE-B (768-dim) · Kinetics-400 pre-train  ·  Multi-seed: 42 / 123 / 456  ·  40 epochs · AdamW · BU SCC A100",
        color=DG, fontsize=7, ha="center", va="center", zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SECTION ARROWS
# ─────────────────────────────────────────────────────────────────────────────
# ① → ②  (cache → model input)
ax.annotate("", xy=(1.4, 8.55), xytext=(1.4, 9.45),
    arrowprops=dict(arrowstyle="->", color=AC2, lw=2,
                    connectionstyle="arc3,rad=0", mutation_scale=12), zorder=7)
ax.text(1.85, 9.0, "features", color=AC2, fontsize=7, va="center",
        style="italic", zorder=7)

# ① → ③  (cache → evaluation)
ax.annotate("", xy=(13.2, 8.55), xytext=(13.2, 9.45),
    arrowprops=dict(arrowstyle="->", color=AC3, lw=2,
                    connectionstyle="arc3,rad=0", mutation_scale=12), zorder=7)
ax.text(13.65, 9.0, "test features", color=AC3, fontsize=7, va="center",
        style="italic", zorder=7)

# ② → ③  model weights
ax.annotate("", xy=(11.9+0.2, 7.1), xytext=(11.4, 7.1),
    arrowprops=dict(arrowstyle="->", color=DG, lw=1.5, linestyle="dashed",
                    connectionstyle="arc3,rad=0", mutation_scale=10), zorder=7)
ax.text(11.65, 7.3, "weights", color=DG, fontsize=6.5, ha="center",
        style="italic", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(FW/2, FH - 0.32,
        "System Design: Weakly-Supervised Violence Detection with VideoMAE-B",
        color=W, fontsize=13.5, fontweight="bold", ha="center", va="center",
        zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER STRIP
# ─────────────────────────────────────────────────────────────────────────────
# (already inside each section — nothing more needed)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_dir = "outputs_scc/videomae_rtfm/figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "system_design.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"✓  {out_path}")
print(f"   Size: {FW}×{FH} in @ 200 dpi  →  {int(FW*200)}×{int(FH*200)} px")
