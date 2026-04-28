#!/usr/bin/env python3
"""Clean, collision-free publication figures for VideoMAE results.

Run:  python src/plot_videomae_paper.py
"""

from __future__ import annotations
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
SEED_DIRS = {
    42:  ROOT / "outputs_scc/videomae_rtfm/seed_42",
    123: ROOT / "outputs_scc/videomae_rtfm/seed_123",
    456: ROOT / "outputs_scc/videomae_rtfm/seed_456",
}
OUT_DIR = ROOT / "outputs_scc/videomae_rtfm/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456]

I3D = {
    "auc": 0.8742, "ap": 0.8213, "macro_f1": 0.1982, "weighted_f1": 0.6541,
    "map_03": 0.009, "map_05": 0.004, "map_07": 0.001,
    "per_class": {
        "Normal":0.812,"Fighting":0.241,"Shooting":0.198,
        "Explosion":0.176,"Robbery":0.143,"Abuse":0.121,
    },
}

# ── palette ───────────────────────────────────────────────────────────────────
C_VM   = "#1565C0"
C_I3D  = "#C62828"
C_GAIN = "#2E7D32"
C_BG   = "#F7F9FC"
C_GRID = "#DDE3EA"
SEED_C  = ["#E65100", "#6A1B9A", "#00695C"]   # orange, purple, teal — clearly distinct
SEED_LS = ["--",      "-.",      ":"]            # dashed, dash-dot, dotted

def base_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         12,
        "axes.facecolor":    C_BG,
        "figure.facecolor":  "white",
        "axes.edgecolor":    "#B0BEC5",
        "axes.linewidth":    1.0,
        "axes.grid":         True,
        "grid.color":        C_GRID,
        "grid.linewidth":    0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "legend.framealpha": 0.95,
        "legend.edgecolor":  "#B0BEC5",
        "figure.dpi":        150,
    })

def load(seed):
    return json.loads((SEED_DIRS[seed] / "results_summary.json").read_text())

def gm(res, key):
    if key == "auc":     return res["test"]["binary"]["auc"]
    if key == "ap":      return res["test"]["binary"]["ap"]
    if key == "macro_f1":return res["test"]["classification"]["macro_f1"]
    if key == "wf1":     return res["test"]["classification"]["weighted_f1"]
    t = {"map03":"0.3","map05":"0.5","map07":"0.7"}.get(key,"0.3")
    v = res.get("temporal_localization",{}).get("tiou",{}).get(t,{})
    return (v.get("mAP") or 0.0) if isinstance(v, dict) else 0.0

ALL = {s: load(s) for s in SEEDS}


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Training curves  (3 panels, shaded band + mean line)
# ══════════════════════════════════════════════════════════════════════════════
def fig1():
    base_style()
    curves = {}
    for s in SEEDS:
        h = ALL[s]["training_curves"]
        curves[s] = {
            "ep":   [r["epoch"] for r in h],
            "loss": [r["train_loss"] for r in h],
            "auc":  [r["val_auc"] for r in h],
            "mf1":  [r["val_macro_f1"] for r in h],
        }
    ep = np.array(curves[42]["ep"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    panels = [
        ("loss", "Training Loss",    "Loss",     C_I3D,  True),
        ("auc",  "Val AUC",          "AUC",      C_VM,   False),
        ("mf1",  "Val Macro-F1",     "Macro-F1", C_GAIN, False),
    ]

    for ax, (key, title, ylabel, color, minimize) in zip(axes, panels):
        mat  = np.array([curves[s][key] for s in SEEDS])
        mean = mat.mean(0)
        std  = mat.std(0, ddof=1)

        # shaded std band
        ax.fill_between(ep, mean-std, mean+std, alpha=0.15, color=color)

        # individual seed lines — different color AND different line style
        for i, s in enumerate(SEEDS):
            ax.plot(ep, curves[s][key], lw=1.6, alpha=0.75,
                    color=SEED_C[i], linestyle=SEED_LS[i], label=f"Seed {s}")

        # mean line (thick)
        ax.plot(ep, mean, lw=2.8, color=color, label="Mean", zorder=5)

        # best-point marker — no overlapping annotation
        best_i = int(np.argmin(mean) if minimize else np.argmax(mean))
        ax.scatter(ep[best_i], mean[best_i], s=90, color=color,
                   zorder=6, edgecolors="white", linewidths=2)

        # annotate in top corner, not on data
        corner_x = ep[-1] * 0.55
        corner_y = ax.get_ylim()[1] if not ax.get_ylim()[1] else mean.max()*1.05
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(1, int(ep[-1]))

    # single shared legend below all panels
    handles = [
        plt.Line2D([0],[0], color=SEED_C[0], lw=1.8, ls=SEED_LS[0], label="Seed 42"),
        plt.Line2D([0],[0], color=SEED_C[1], lw=1.8, ls=SEED_LS[1], label="Seed 123"),
        plt.Line2D([0],[0], color=SEED_C[2], lw=1.8, ls=SEED_LS[2], label="Seed 456"),
        mpatches.Patch(color="#888", alpha=0.3, label="±1 std"),
        plt.Line2D([0],[0], color="#333", lw=2.8, label="Mean"),
        plt.scatter([],[], s=90, color="#333", edgecolors="white", linewidths=2, label="Best epoch"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.08), fontsize=10.5,
               framealpha=0.95, edgecolor="#B0BEC5")
    fig.suptitle("VideoMAE-B Training Dynamics  (3 Seeds, 40 Epochs)",
                 fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout(w_pad=3)
    p = OUT_DIR / "fig1_training_dynamics.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Dumbbell chart  (fixed text columns, no collisions)
# ══════════════════════════════════════════════════════════════════════════════
def fig2():
    base_style()
    mk = ["auc","ap","macro_f1","wf1","map03","map05","map07"]
    ml = ["Binary AUC","Binary AP","Macro-F1","Weighted-F1",
          "mAP @ 0.3","mAP @ 0.5","mAP @ 0.7"]
    i3d = [I3D["auc"],I3D["ap"],I3D["macro_f1"],I3D["weighted_f1"],
           I3D["map_03"],I3D["map_05"],I3D["map_07"]]
    vm_all   = [[gm(ALL[s],k) for s in SEEDS] for k in mk]
    vm_means = [np.mean(v) for v in vm_all]
    vm_stds  = [np.std(v,ddof=1) for v in vm_all]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    y = np.arange(len(mk)) * 1.4   # spread rows apart

    # background alternating bands
    for i in range(0, len(mk), 2):
        ax.axhspan(y[i]-0.55, y[i]+0.55, color="#EEF2FA", zorder=0, lw=0)

    for i, (yl, iv, vm, vsd, lab) in enumerate(
            zip(y, i3d, vm_means, vm_stds, ml)):

        # connecting line
        ax.plot([iv, vm], [yl, yl], lw=2.0, color="#90A4AE", zorder=1, solid_capstyle="round")

        # I3D dot
        ax.scatter(iv, yl, s=200, color=C_I3D, zorder=4,
                   edgecolors="white", linewidths=2)
        # VideoMAE dot
        ax.scatter(vm, yl, s=240, color=C_VM, zorder=4,
                   edgecolors="white", linewidths=2)
        # std cap
        ax.errorbar(vm, yl, xerr=vsd, fmt="none",
                    ecolor=C_VM, elinewidth=2, capsize=6, capthick=2, zorder=5)

        # ── FIXED TEXT COLUMNS (no collision) ─────────────────────────────
        # col A: I3D value  at x = -0.10
        ax.text(-0.10, yl, f"{iv:.3f}", va="center", ha="center",
                fontsize=10.5, color=C_I3D, fontweight="bold")
        # col B: VideoMAE value  at x = 1.05
        ax.text(1.05, yl, f"{vm:.3f}", va="center", ha="center",
                fontsize=10.5, color=C_VM, fontweight="bold")
        # col C: gain  at x = 1.20
        gain = vm - iv
        pct  = gain/iv*100 if iv > 0 else 0
        sign = "+" if gain >= 0 else ""
        ax.text(1.20, yl, f"{sign}{gain:.3f}\n({sign}{pct:.0f}%)",
                va="center", ha="center", fontsize=9.5,
                color=C_GAIN, fontweight="bold", linespacing=1.4)

    # column headers
    for xh, lh, col in [(-0.10,"I3D",C_I3D),(1.05,"VideoMAE",C_VM),(1.20,"Gain",C_GAIN)]:
        ax.text(xh, y[-1]+1.1, lh, va="center", ha="center",
                fontsize=11, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.2))

    ax.set_yticks(y)
    ax.set_yticklabels(ml, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.22, 1.36)
    ax.set_ylim(y[0]-0.9, y[-1]+1.6)
    ax.set_xlabel("Score", fontsize=12)
    ax.axvline(0, color="#CFD8DC", lw=0.8, ls=":")
    ax.axvline(1, color="#CFD8DC", lw=0.8, ls=":")
    ax.set_title("I3D Baseline  →  VideoMAE-B: Per-Metric Improvement",
                 fontsize=15, fontweight="bold", pad=16)

    legend_h = [
        plt.scatter([],[],s=200,color=C_I3D,edgecolors="white",lw=2,label="I3D (Step-7 baseline)"),
        plt.scatter([],[],s=240,color=C_VM, edgecolors="white",lw=2,label="VideoMAE-B  mean ± std"),
    ]
    ax.legend(handles=legend_h, loc="lower right", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    p = OUT_DIR / "fig2_dumbbell_comparison.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — mAP vs tIoU line chart
# ══════════════════════════════════════════════════════════════════════════════
def fig3():
    base_style()
    thrs     = [0.3, 0.5, 0.7]
    thr_keys = ["map03","map05","map07"]
    i3d_line = [I3D["map_03"],I3D["map_05"],I3D["map_07"]]
    vm_all   = [[gm(ALL[s],k) for s in SEEDS] for k in thr_keys]
    vm_means = [np.mean(v) for v in vm_all]
    vm_stds  = [np.std(v,ddof=1) for v in vm_all]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    # I3D dashed line
    ax.plot(thrs, i3d_line, "o--", color=C_I3D, lw=2.2,
            ms=10, mec="white", mew=2, label="I3D (Step-7 baseline)", zorder=3)

    # VideoMAE ribbon + line
    vm_lo = [m-s for m,s in zip(vm_means,vm_stds)]
    vm_hi = [m+s for m,s in zip(vm_means,vm_stds)]
    ax.fill_between(thrs, vm_lo, vm_hi, alpha=0.18, color=C_VM)
    ax.plot(thrs, vm_means, "o-", color=C_VM, lw=2.8,
            ms=11, mec="white", mew=2, label="VideoMAE-B  mean ± std", zorder=4)

    # per-seed thin dots only (no lines — avoids clutter)
    for i, s in enumerate(SEEDS):
        vals = [gm(ALL[s],k) for k in thr_keys]
        ax.plot(thrs, vals, ".", ms=8, color=SEED_C[i], alpha=0.6,
                zorder=5, label=f"Seed {s}")

    # gain labels — placed ABOVE each VideoMAE point, away from I3D line
    for t, iv, vm, vsd in zip(thrs, i3d_line, vm_means, vm_stds):
        mult = vm/iv if iv > 0 else 0
        y_pos = vm + vsd + 0.004
        ax.annotate(
            f"×{mult:.1f} vs I3D",
            xy=(t, vm), xytext=(t, y_pos + 0.008),
            ha="center", va="bottom", fontsize=11, color=C_GAIN, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_GAIN, lw=1.2,
                            mutation_scale=10),
        )

    # I3D value labels below each point
    for t, iv in zip(thrs, i3d_line):
        ax.text(t, iv - 0.003, f"{iv:.3f}", ha="center", va="top",
                fontsize=10, color=C_I3D, fontweight="bold")

    # VideoMAE value labels — offset slightly to avoid ribbon
    for t, vm in zip(thrs, vm_means):
        ax.text(t + 0.018, vm, f"{vm:.3f}", ha="left", va="center",
                fontsize=10, color=C_VM, fontweight="bold")

    ax.set_xticks(thrs)
    ax.set_xticklabels(["tIoU = 0.3","tIoU = 0.5","tIoU = 0.7"], fontsize=12)
    ax.set_ylabel("mAP", fontsize=13)
    ax.set_xlim(0.22, 0.82)
    ax.set_ylim(-0.005, ax.get_ylim()[1] * 1.35)
    ax.set_title("Temporal Localization  mAP @ tIoU Thresholds",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="upper right", fontsize=10.5)

    fig.tight_layout()
    p = OUT_DIR / "fig3_localization_curve.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Reproducibility  (clean horizontal, text in fixed right column)
# ══════════════════════════════════════════════════════════════════════════════
def fig4():
    base_style()
    mk = ["auc","ap","macro_f1","wf1","map03","map05"]
    ml = ["Binary AUC","Binary AP","Macro-F1","Weighted-F1","mAP@0.3","mAP@0.5"]
    vm_all   = [[gm(ALL[s],k) for s in SEEDS] for k in mk]
    vm_means = [np.mean(v) for v in vm_all]
    vm_stds  = [np.std(v,ddof=1) for v in vm_all]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    y = np.arange(len(mk))

    # background bands
    for i in range(0,len(mk),2):
        ax.axhspan(i-0.45, i+0.45, color="#EEF2FA", zorder=0, lw=0)

    # light bar to show magnitude
    ax.barh(y, vm_means, height=0.5, color=C_VM, alpha=0.12, zorder=1)

    # std error bar
    ax.errorbar(vm_means, y, xerr=vm_stds, fmt="none",
                ecolor=C_VM, elinewidth=2.5, capsize=10, capthick=2.2, zorder=3)

    # mean diamond
    ax.scatter(vm_means, y, marker="D", s=130, color=C_VM,
               zorder=5, edgecolors="white", linewidths=2, label="Mean")

    # seed dots — spaced vertically so they NEVER overlap the diamond
    offsets = [-0.20, 0.20, 0.0]  # two above/below, one at center but pushed right
    x_nudge = [0, 0, 0.01]
    for i, (s, off, xn) in enumerate(zip(SEEDS, offsets, x_nudge)):
        vals = [gm(ALL[s],k) for k in mk]
        ax.scatter([v+xn for v in vals], y+off, s=75, color=SEED_C[i], zorder=4,
                   edgecolors="white", linewidths=1.2, label=f"Seed {s}",
                   marker="o")

    # ── FIXED text column at x = 1.03 (no collision with dots) ───────────
    for xi, (m, s, lab) in enumerate(zip(vm_means, vm_stds, ml)):
        ax.text(1.03, xi,
                f"{m:.4f} ± {s:.4f}",
                va="center", ha="left", fontsize=10.5,
                color=C_VM, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(ml, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim(0, 1.28)
    ax.set_title("Cross-Seed Reproducibility  —  VideoMAE-B (3 Seeds)",
                 fontsize=15, fontweight="bold", pad=14)

    # vertical reference line at 1.03 column
    ax.axvline(1.0, color="#B0BEC5", lw=0.8, ls=":")

    ax.legend(loc="lower right", fontsize=10.5, ncol=2)
    ax.invert_yaxis()
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)

    fig.tight_layout()
    p = OUT_DIR / "fig4_reproducibility.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Per-class F1  (horizontal, sorted, delta in separate column)
# ══════════════════════════════════════════════════════════════════════════════
def fig5():
    base_style()
    best = 123
    pc      = ALL[best]["test"]["classification"]["per_class"]
    classes = [p["class"].capitalize() for p in pc]
    vm_f1   = [p["f1"] for p in pc]
    vm_pre  = [p["precision"] for p in pc]
    vm_rec  = [p["recall"] for p in pc]
    i3d_f1  = [I3D["per_class"].get(c, 0.0) for c in classes]
    deltas  = [v-i for v,i in zip(vm_f1, i3d_f1)]

    # sort descending by VideoMAE F1
    order = sorted(range(len(classes)), key=lambda x: vm_f1[x], reverse=True)
    classes = [classes[o] for o in order]
    vm_f1   = [vm_f1[o]   for o in order]
    vm_pre  = [vm_pre[o]  for o in order]
    vm_rec  = [vm_rec[o]  for o in order]
    i3d_f1  = [i3d_f1[o]  for o in order]
    deltas  = [deltas[o]  for o in order]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    y = np.arange(len(classes))
    h = 0.26

    # alternating bands
    for i in range(0, len(classes), 2):
        ax.axhspan(i-0.5, i+0.5, color="#EEF2FA", zorder=0, lw=0)

    # bars: I3D, VideoMAE F1, VideoMAE Precision
    ax.barh(y+h,  i3d_f1, h*1.9, color=C_I3D,    alpha=0.80,
            label="I3D F1",              edgecolor="white", linewidth=0.6)
    ax.barh(y,    vm_f1,  h*1.9, color=C_VM,     alpha=0.90,
            label="VideoMAE F1",         edgecolor="white", linewidth=0.6)
    ax.barh(y-h,  vm_pre, h*1.9, color="#5C6BC0", alpha=0.70,
            label="VideoMAE Precision",  edgecolor="white", linewidth=0.6)

    # ── values in a FIXED column at x=0.88 (no bar overlap) ──────────────
    for i, (vf, ip, d) in enumerate(zip(vm_f1, i3d_f1, deltas)):
        # VideoMAE F1
        ax.text(0.88, y[i]+h,  f"{ip:.2f}", va="center", ha="left",
                fontsize=10, color=C_I3D, fontweight="bold")
        ax.text(0.88, y[i],    f"{vf:.2f}", va="center", ha="left",
                fontsize=10, color=C_VM, fontweight="bold")

        # delta in a separate column at x=1.02
        sign = "+" if d >= 0 else ""
        col  = C_GAIN if d >= 0 else C_I3D
        ax.text(1.02, y[i], f"{sign}{d:.2f}",
                va="center", ha="left", fontsize=10,
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=col, lw=0.9, alpha=0.9))

    # column headers
    ax.text(0.88, len(classes)-0.5, "Score", ha="left",
            fontsize=10, color="#455A64", style="italic")
    ax.text(1.02, len(classes)-0.5, "Δ F1",  ha="left",
            fontsize=10, color="#455A64", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim(0, 1.18)
    ax.axvline(0.85, color="#B0BEC5", lw=0.8, ls=":")
    ax.axvline(1.0,  color="#B0BEC5", lw=0.8, ls=":")
    ax.set_title(f"Per-Class Performance  —  VideoMAE-B vs I3D  (Best Seed {best})",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=10.5)
    ax.invert_yaxis()
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)

    fig.tight_layout()
    p = OUT_DIR / "fig5_per_class.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Radar / Spider  (large, spaced labels)
# ══════════════════════════════════════════════════════════════════════════════
def fig6():
    base_style()
    mk = ["auc","ap","macro_f1","wf1","map03","map05"]
    ml = ["AUC","AP","Macro-F1","Wtd-F1","mAP@0.3","mAP@0.5"]
    i3d_v = [I3D["auc"],I3D["ap"],I3D["macro_f1"],I3D["weighted_f1"],
             I3D["map_03"],I3D["map_05"]]
    vm_means = [np.mean([gm(ALL[s],k) for s in SEEDS]) for k in mk]
    vm_stds  = [np.std( [gm(ALL[s],k) for s in SEEDS], ddof=1) for k in mk]

    N      = len(mk)
    angles = [n/N * 2*math.pi for n in range(N)] + [0]
    i3d_c  = i3d_v + [i3d_v[0]]
    vm_c   = vm_means + [vm_means[0]]
    lo_c   = [m-s for m,s in zip(vm_means,vm_stds)] + [vm_means[0]-vm_stds[0]]
    hi_c   = [m+s for m,s in zip(vm_means,vm_stds)] + [vm_means[0]+vm_stds[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C_BG)

    # styling
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"],
                       fontsize=9, color="#90A4AE")
    ax.set_rlabel_position(15)
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
    ax.tick_params(axis="both", which="both", length=0)

    # I3D
    ax.plot(angles, i3d_c, "o--", color=C_I3D, lw=2, ms=7,
            mec="white", mew=1.5, label="I3D baseline")
    ax.fill(angles, i3d_c, alpha=0.10, color=C_I3D)

    # VideoMAE ribbon + line
    ax.fill_between(angles, lo_c, hi_c, alpha=0.18, color=C_VM)
    ax.plot(angles, vm_c, "o-", color=C_VM, lw=2.8, ms=9,
            mec="white", mew=2, label="VideoMAE-B mean")
    ax.fill(angles, vm_c, alpha=0.12, color=C_VM)

    # axis labels — pushed outward with padding to avoid overlap
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])   # clear default
    label_pad = 1.18         # push outside the chart
    for ang, lab, vm, iv in zip(angles[:-1], ml, vm_means, i3d_v):
        x = math.sin(ang) * label_pad
        y = math.cos(ang) * label_pad
        ax.text(ang, label_pad, f"{lab}\n{vm:.3f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=C_VM,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=C_VM, lw=1.0, alpha=0.95))

    ax.set_title("All-Metrics Radar  —  VideoMAE-B vs I3D",
                 fontsize=14, fontweight="bold", pad=85)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.18, -0.12), fontsize=11)

    p = OUT_DIR / "fig6_radar_overview.png"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓  {p.name}")


if __name__ == "__main__":
    print("\nGenerating paper figures...\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"\n✓  All 6 figures → {OUT_DIR}\n")
