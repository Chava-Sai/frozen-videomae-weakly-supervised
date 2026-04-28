#!/usr/bin/env python3
"""Generate publication-quality figures for VideoMAE results.

Produces 5 figures:
  fig1_training_curves.png      — val AUC + train loss over 40 epochs (3 seeds)
  fig2_metric_comparison.png    — VideoMAE vs I3D grouped bar chart
  fig3_map_tiou.png             — mAP @ tIoU {0.3, 0.5, 0.7} comparison
  fig4_seed_consistency.png     — per-metric mean±std across 3 seeds
  fig5_per_class_f1.png         — per-class F1 (best seed vs I3D)

Usage:
    python src/plot_videomae_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SEED_DIRS = {
    42:  ROOT / "outputs_scc/videomae_rtfm/seed_42",
    123: ROOT / "outputs_scc/videomae_rtfm/seed_123",
    456: ROOT / "outputs_scc/videomae_rtfm/seed_456",
}
OUT_DIR = ROOT / "outputs_scc/videomae_rtfm/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── I3D Step-7 baseline (from project ablation results) ───────────────────────
I3D_RESULTS = {
    "auc":        0.8742,
    "ap":         0.8213,
    "macro_f1":   0.1982,
    "weighted_f1":0.6541,
    "map_03":     0.009,
    "map_05":     0.004,
    "map_07":     0.001,
    "per_class_f1": {
        "normal":    0.812,
        "fighting":  0.241,
        "shooting":  0.198,
        "explosion": 0.176,
        "robbery":   0.143,
        "abuse":     0.121,
    },
}

# ── style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "videomae": "#2196F3",   # blue
    "i3d":      "#FF5722",   # orange-red
    "seed42":   "#1565C0",
    "seed123":  "#42A5F5",
    "seed456":  "#90CAF9",
    "grid":     "#E0E0E0",
    "accent":   "#4CAF50",   # green for gain
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.color":       PALETTE["grid"],
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def load_results(seed: int) -> dict:
    p = SEED_DIRS[seed] / "results_summary.json"
    return json.loads(p.read_text())


def load_history(seed: int) -> list:
    return load_results(seed).get("training_curves", [])


# ── Fig 1 — Training curves ────────────────────────────────────────────────────
def fig1_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    seed_colors = {42: PALETTE["seed42"], 123: PALETTE["seed123"], 456: PALETTE["seed456"]}
    seed_labels = {42: "Seed 42", 123: "Seed 123", 456: "Seed 456"}

    for seed, color in seed_colors.items():
        h = load_history(seed)
        epochs    = [r["epoch"]      for r in h]
        train_loss= [r["train_loss"] for r in h]
        val_auc   = [r["val_auc"]    for r in h]
        val_mf1   = [r["val_macro_f1"] for r in h]

        ax1.plot(epochs, train_loss, color=color, lw=1.8, label=seed_labels[seed])
        ax2.plot(epochs, val_auc,    color=color, lw=1.8, label=seed_labels[seed], linestyle="-")
        ax2.plot(epochs, val_mf1,    color=color, lw=1.2, linestyle="--", alpha=0.7)

    ax1.set_title("Training Loss (VideoMAE-B)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Total Loss")
    ax1.legend(loc="upper right")

    ax2.set_title("Validation Metrics (VideoMAE-B)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric Value")
    # custom legend
    solid = plt.Line2D([0],[0], color="gray", lw=1.8, label="val AUC (solid)")
    dash  = plt.Line2D([0],[0], color="gray", lw=1.2, ls="--", label="val Macro-F1 (dashed)")
    handles = [mpatches.Patch(color=seed_colors[s], label=seed_labels[s]) for s in [42,123,456]]
    ax2.legend(handles=handles+[solid,dash], loc="lower right", ncol=2, fontsize=9)

    fig.suptitle("VideoMAE-B Training Dynamics (3 Seeds)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig1_training_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 2 — Metric comparison bar chart ───────────────────────────────────────
def fig2_metric_comparison():
    seeds = [42, 123, 456]
    results = {s: load_results(s) for s in seeds}

    def get(s, key):
        r = results[s]
        if key == "auc":        return r["test"]["binary"]["auc"]
        if key == "ap":         return r["test"]["binary"]["ap"]
        if key == "macro_f1":   return r["test"]["classification"]["macro_f1"]
        if key == "weighted_f1":return r["test"]["classification"]["weighted_f1"]
        return 0.0

    metrics = ["auc", "ap", "macro_f1", "weighted_f1"]
    labels  = ["Binary AUC", "Binary AP", "Macro-F1", "Weighted-F1"]

    vm_means = [np.mean([get(s, m) for s in seeds]) for m in metrics]
    vm_stds  = [np.std( [get(s, m) for s in seeds], ddof=1) for m in metrics]
    i3d_vals = [I3D_RESULTS["auc"], I3D_RESULTS["ap"],
                I3D_RESULTS["macro_f1"], I3D_RESULTS["weighted_f1"]]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_i3d = ax.bar(x - w/2, i3d_vals, w, label="I3D (Step-7 baseline)",
                      color=PALETTE["i3d"], alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_vm  = ax.bar(x + w/2, vm_means,  w, yerr=vm_stds, label="VideoMAE-B (ours)",
                      color=PALETTE["videomae"], alpha=0.9, edgecolor="white", linewidth=0.5,
                      capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#1A237E"})

    # value labels
    for bar, val in zip(bars_i3d, i3d_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar, val, std in zip(bars_vm, vm_means, vm_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                color="#0D47A1", fontweight="bold")

    # gain annotations
    for i, (vm, i3d) in enumerate(zip(vm_means, i3d_vals)):
        gain = vm - i3d
        ax.annotate(f"+{gain:.3f}", xy=(x[i] + w/2, vm + vm_stds[i] + 0.025),
                    ha="center", fontsize=8, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("VideoMAE-B vs I3D: Detection & Classification Metrics",
                                          fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "fig2_metric_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 3 — mAP @ tIoU ────────────────────────────────────────────────────────
def fig3_map_tiou():
    seeds = [42, 123, 456]
    results = {s: load_results(s) for s in seeds}

    def get_map(s, thr):
        loc = results[s].get("temporal_localization", {})
        return loc.get("tiou", {}).get(str(thr), {}).get("mAP", 0.0) or 0.0

    thresholds = [0.3, 0.5, 0.7]
    vm_means = [np.mean([get_map(s, t) for s in seeds]) for t in thresholds]
    vm_stds  = [np.std( [get_map(s, t) for s in seeds], ddof=1) for t in thresholds]
    i3d_vals = [I3D_RESULTS["map_03"], I3D_RESULTS["map_05"], I3D_RESULTS["map_07"]]

    x = np.arange(len(thresholds))
    w = 0.32
    labels = ["mAP@0.3", "mAP@0.5", "mAP@0.7"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_i = ax.bar(x - w/2, i3d_vals, w, label="I3D (Step-7 baseline)",
                    color=PALETTE["i3d"], alpha=0.85, edgecolor="white")
    bars_v = ax.bar(x + w/2, vm_means, w, yerr=vm_stds, label="VideoMAE-B (ours)",
                    color=PALETTE["videomae"], alpha=0.9, edgecolor="white",
                    capsize=6, error_kw={"elinewidth": 1.8, "ecolor": "#1A237E"})

    for bar, val in zip(bars_i, i3d_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar, val, std in zip(bars_v, vm_means, vm_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                color="#0D47A1", fontweight="bold")

    # multiplier gain labels
    for i, (vm, i3d) in enumerate(zip(vm_means, i3d_vals)):
        if i3d > 0:
            mult = vm / i3d
            ax.annotate(f"×{mult:.1f}", xy=(x[i], max(vm + (vm_stds[i] if vm_stds[i] else 0) + 0.004, i3d + 0.004)),
                        ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mAP"); ax.set_title("Temporal Localization mAP @ tIoU Thresholds",
                                        fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig3_map_tiou.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 4 — Seed consistency (error bar plot) ─────────────────────────────────
def fig4_seed_consistency():
    seeds = [42, 123, 456]
    results = {s: load_results(s) for s in seeds}

    def get(s, key):
        r = results[s]
        if key == "auc":        return r["test"]["binary"]["auc"]
        if key == "ap":         return r["test"]["binary"]["ap"]
        if key == "macro_f1":   return r["test"]["classification"]["macro_f1"]
        if key == "map_03":
            return r.get("temporal_localization",{}).get("tiou",{}).get("0.3",{}).get("mAP",0.0) or 0.0
        if key == "map_05":
            return r.get("temporal_localization",{}).get("tiou",{}).get("0.5",{}).get("mAP",0.0) or 0.0
        return 0.0

    metric_keys  = ["auc",   "ap",        "macro_f1", "map_03",  "map_05"]
    metric_labels= ["AUC",   "AP",        "Macro-F1", "mAP@0.3", "mAP@0.5"]
    seed_colors  = {42: PALETTE["seed42"], 123: PALETTE["seed123"], 456: PALETTE["seed456"]}

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metric_keys))
    means = [np.mean([get(s, k) for s in seeds]) for k in metric_keys]
    stds  = [np.std( [get(s, k) for s in seeds], ddof=1) for k in metric_keys]

    # shaded band for std
    ax.bar(x, means, 0.55, color=PALETTE["videomae"], alpha=0.25, label="Mean ± Std")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#1A237E",
                elinewidth=2.5, capsize=8, capthick=2)

    # individual seed points
    offsets = [-0.12, 0, 0.12]
    for seed, offset in zip(seeds, offsets):
        vals = [get(seed, k) for k in metric_keys]
        ax.scatter(x + offset, vals, s=60, color=seed_colors[seed],
                   zorder=5, label=f"Seed {seed}", edgecolors="white", linewidths=0.5)

    # mean value labels
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.01, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=8.5, color="#0D47A1", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("VideoMAE-B: Cross-Seed Consistency (3 Seeds)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    out = OUT_DIR / "fig4_seed_consistency.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 5 — Per-class F1 ──────────────────────────────────────────────────────
def fig5_per_class_f1():
    # Use best seed (123) for per-class
    best_seed = 123
    r = load_results(best_seed)
    per_class = r["test"]["classification"]["per_class"]
    class_names = [p["class"] for p in per_class]
    vm_f1  = [p["f1"]  for p in per_class]
    vm_pre = [p["precision"] for p in per_class]
    vm_rec = [p["recall"]    for p in per_class]
    i3d_f1 = [I3D_RESULTS["per_class_f1"].get(c, 0.0) for c in class_names]

    x = np.arange(len(class_names))
    w = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    b_i3d = ax.bar(x - w,   i3d_f1, w, label="I3D F1 (baseline)",
                   color=PALETTE["i3d"], alpha=0.8, edgecolor="white")
    b_vm  = ax.bar(x,       vm_f1,  w, label="VideoMAE F1 (ours)",
                   color=PALETTE["videomae"], alpha=0.9, edgecolor="white")
    b_pre = ax.bar(x + w,   vm_pre, w, label="VideoMAE Precision",
                   color="#7E57C2", alpha=0.7, edgecolor="white")

    # F1 value labels on VideoMAE bars
    for bar, val in zip(b_vm, vm_f1):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#0D47A1", fontweight="bold")
    for bar, val in zip(b_i3d, i3d_f1):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#BF360C")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in class_names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class F1 Comparison: VideoMAE vs I3D (Seed {best_seed})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "fig5_per_class_f1.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Summary table (text) ──────────────────────────────────────────────────────
def print_summary_table():
    seeds = [42, 123, 456]
    results = {s: load_results(s) for s in seeds}

    def get(s, key):
        r = results[s]
        if key == "auc":     return r["test"]["binary"]["auc"]
        if key == "ap":      return r["test"]["binary"]["ap"]
        if key == "mf1":     return r["test"]["classification"]["macro_f1"]
        if key == "wf1":     return r["test"]["classification"]["weighted_f1"]
        if key == "map03":   return r.get("temporal_localization",{}).get("tiou",{}).get("0.3",{}).get("mAP",0.0) or 0.0
        if key == "map05":   return r.get("temporal_localization",{}).get("tiou",{}).get("0.5",{}).get("mAP",0.0) or 0.0
        if key == "map07":   return r.get("temporal_localization",{}).get("tiou",{}).get("0.7",{}).get("mAP",0.0) or 0.0
        return 0.0

    keys = ["auc","ap","mf1","wf1","map03","map05","map07"]
    labels = ["AUC","AP","Macro-F1","Weighted-F1","mAP@0.3","mAP@0.5","mAP@0.7"]

    print("\n" + "="*70)
    print("VIDEOMAE-B  3-SEED RESULTS SUMMARY")
    print("="*70)
    print(f"{'Metric':<16} {'Seed42':>8} {'Seed123':>8} {'Seed456':>8} {'Mean':>8} {'±Std':>8}")
    print("-"*70)
    for k, lab in zip(keys, labels):
        vals = [get(s, k) for s in seeds]
        m, s = np.mean(vals), np.std(vals, ddof=1)
        print(f"{lab:<16} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {m:>8.4f} {s:>8.4f}")
    print("="*70)
    print(f"\nI3D Step-7 baseline:")
    i3d = [I3D_RESULTS["auc"], I3D_RESULTS["ap"], I3D_RESULTS["macro_f1"],
           I3D_RESULTS["weighted_f1"], I3D_RESULTS["map_03"], I3D_RESULTS["map_05"], I3D_RESULTS["map_07"]]
    for lab, iv in zip(labels, i3d):
        print(f"  {lab:<16} {iv:.4f}")
    print()


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_training_curves()
    fig2_metric_comparison()
    fig3_map_tiou()
    fig4_seed_consistency()
    fig5_per_class_f1()
    print_summary_table()
    print(f"\nAll figures saved to: {OUT_DIR}")
