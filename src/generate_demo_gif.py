"""
generate_demo_gif.py
Generates an animated GIF showing anomaly score detection in real-time.
Like the baseball GIF — shows the model "thinking" frame by frame.

Run: python src/generate_demo_gif.py
Output: outputs_scc/videomae_rtfm/figures/anomaly_demo.gif
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

OUT_DIR = Path("outputs_scc/videomae_rtfm/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulated anomaly scores for a "Shooting" video ──────────────────────────
# Realistic pattern: normal → build-up → violence peak → return to normal
np.random.seed(42)
N = 48  # number of segments

def make_score_curve():
    base  = np.random.uniform(0.05, 0.18, N)        # normal baseline
    peak  = np.zeros(N)
    # violence region: segments 22–35
    for i in range(22, 36):
        t = (i - 22) / 13
        peak[i] = 0.62 + 0.30 * np.sin(np.pi * t) + np.random.normal(0, 0.03)
    # build-up: segments 18–22
    for i in range(18, 22):
        t = (i - 18) / 4
        peak[i] = 0.15 + 0.50 * t + np.random.normal(0, 0.02)
    # cool-down: segments 35–40
    for i in range(35, 40):
        t = (i - 35) / 5
        peak[i] = 0.70 * (1 - t) + np.random.normal(0, 0.02)
    scores = np.clip(base + peak, 0.02, 0.97)
    return scores

scores  = make_score_curve()
THRESH  = 0.40
GT_ANOM = np.array([1 if 22 <= i <= 35 else 0 for i in range(N)])

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0f172a"
NORMAL  = "#1e3a5f"
ANOM    = "#7f1d1d"
BORDER  = "#334155"
BLUE    = "#3b82f6"
RED     = "#ef4444"
GREEN   = "#10b981"
AMBER   = "#f59e0b"
WHITE   = "#f8fafc"
GRAY    = "#94a3b8"

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.10,
                    hspace=0.55)

ax_frames = fig.add_axes([0.06, 0.70, 0.91, 0.18])   # frame strip
ax_bar    = fig.add_axes([0.06, 0.38, 0.91, 0.26])   # score bars
ax_curve  = fig.add_axes([0.06, 0.10, 0.91, 0.22])   # score curve

for ax in [ax_frames, ax_bar, ax_curve]:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

# ── Static elements ───────────────────────────────────────────────────────────
# Title
fig.text(0.5, 0.96, "VideoMAE-B Anomaly Detection — Live Score Timeline",
         ha="center", va="top", fontsize=15, fontweight="bold",
         color=WHITE, fontfamily="DejaVu Sans")
fig.text(0.5, 0.91, "Shooting video · 48 segments · threshold τ = 0.40",
         ha="center", va="top", fontsize=11, color=GRAY)

# Frame strip labels
ax_frames.set_xlim(-0.5, N - 0.5)
ax_frames.set_ylim(0, 1)
ax_frames.axis("off")
ax_frames.text(-0.3, 0.5, "Video\nFrames", ha="right", va="center",
               fontsize=9, color=GRAY)

# Draw all frame slots (dim)
frame_rects = []
for i in range(N):
    color = ANOM if GT_ANOM[i] else NORMAL
    r = mpatches.FancyBboxPatch((i - 0.42, 0.08), 0.84, 0.84,
                                 boxstyle="round,pad=0.04",
                                 facecolor="#1e293b", edgecolor=BORDER,
                                 linewidth=0.8, zorder=1)
    ax_frames.add_patch(r)
    frame_rects.append(r)

# Scan line on frame strip
scan_line = ax_frames.axvline(0, color=AMBER, lw=2, zorder=5, alpha=0.9)

# Bar chart
ax_bar.set_xlim(-0.5, N - 0.5)
ax_bar.set_ylim(0, 1.05)
ax_bar.set_ylabel("Anomaly Score", color=WHITE, fontsize=10)
ax_bar.tick_params(colors=GRAY, labelsize=8)
ax_bar.set_xticks(np.arange(0, N, 6))
ax_bar.set_xticklabels([f"t={i}" for i in np.arange(0, N, 6)], color=GRAY)
ax_bar.yaxis.label.set_color(WHITE)
ax_bar.tick_params(axis="y", colors=GRAY)
# Threshold line
ax_bar.axhline(THRESH, color=AMBER, lw=1.5, ls="--", alpha=0.8, zorder=3)
ax_bar.text(N - 0.5, THRESH + 0.02, f"τ={THRESH}", ha="right",
            color=AMBER, fontsize=9)
# GT shading
ax_bar.axvspan(21.5, 35.5, alpha=0.08, color=RED, zorder=0)

bars = ax_bar.bar(range(N), [0]*N, width=0.7, color=BLUE, zorder=2)

# Score curve
ax_curve.set_xlim(-0.5, N - 0.5)
ax_curve.set_ylim(0, 1.05)
ax_curve.set_xlabel("Segment Index", color=WHITE, fontsize=10)
ax_curve.set_ylabel("Score", color=WHITE, fontsize=10)
ax_curve.tick_params(colors=GRAY, labelsize=8)
ax_curve.xaxis.label.set_color(WHITE)
ax_curve.yaxis.label.set_color(WHITE)
ax_curve.axhline(THRESH, color=AMBER, lw=1.2, ls="--", alpha=0.7)
ax_curve.axvspan(21.5, 35.5, alpha=0.08, color=RED, zorder=0)

curve_line, = ax_curve.plot([], [], color=GREEN, lw=2.2, zorder=4)
curve_dot,  = ax_curve.plot([], [], "o", color=AMBER, ms=7, zorder=5)

# Legend
leg_items = [
    mpatches.Patch(facecolor=GREEN,   label="Score curve"),
    mpatches.Patch(facecolor=AMBER,   alpha=0.7, label=f"Threshold τ={THRESH}"),
    mpatches.Patch(facecolor=RED,     alpha=0.3, label="Ground-truth violence"),
    mpatches.Patch(facecolor="#1e40af", label="Normal score"),
    mpatches.Patch(facecolor=RED,     label="Anomaly score"),
]
ax_curve.legend(handles=leg_items, loc="upper left", fontsize=8,
                facecolor="#1e293b", edgecolor=BORDER,
                labelcolor=WHITE, ncol=5)

# Detection banner (hidden initially)
banner_bg = fig.add_axes([0.30, 0.81, 0.40, 0.07])
banner_bg.set_facecolor("#7f1d1d")
banner_bg.axis("off")
banner_txt = banner_bg.text(0.5, 0.5, "** VIOLENCE DETECTED **",
                             ha="center", va="center",
                             fontsize=13, fontweight="bold",
                             color="#fca5a5")
banner_bg.set_visible(False)

# Segment counter text
seg_text = ax_bar.text(0.01, 0.95, "Segment: 0 / 48",
                        transform=ax_bar.transAxes,
                        ha="left", va="top", fontsize=10,
                        color=WHITE, fontweight="bold")

score_text = ax_bar.text(0.99, 0.95, "Score: 0.00",
                          transform=ax_bar.transAxes,
                          ha="right", va="top", fontsize=10,
                          color=WHITE, fontweight="bold")

# ── Animation ─────────────────────────────────────────────────────────────────
violence_active = False

def update(frame):
    global violence_active
    i = frame  # current segment index

    # Update frame strip — light up processed frames
    for j in range(i + 1):
        s = scores[j]
        col = RED if s >= THRESH else BLUE
        frame_rects[j].set_facecolor(col)
        frame_rects[j].set_edgecolor(col)
        frame_rects[j].set_alpha(0.85)

    # Scan line
    scan_line.set_xdata([i, i])

    # Update bars
    for j, bar in enumerate(bars):
        if j <= i:
            s = scores[j]
            bar.set_height(s)
            bar.set_color(RED if s >= THRESH else "#1e40af")
        else:
            bar.set_height(0)

    # Update curve
    xs = list(range(i + 1))
    ys = scores[:i + 1].tolist()
    curve_line.set_data(xs, ys)
    curve_dot.set_data([i], [scores[i]])

    # Texts
    seg_text.set_text(f"Segment: {i+1} / {N}")
    s = scores[i]
    score_text.set_text(f"Score: {s:.3f}")
    score_text.set_color(RED if s >= THRESH else GREEN)

    # Violence banner
    currently_anomaly = s >= THRESH
    if currently_anomaly and not violence_active:
        violence_active = True
        banner_bg.set_visible(True)
    elif not currently_anomaly and violence_active:
        violence_active = False
        banner_bg.set_visible(False)

    return list(bars) + [scan_line, curve_line, curve_dot,
                        seg_text, score_text] + list(frame_rects)

# Pause at end: repeat last frame 20 times
total_frames = list(range(N)) + [N - 1] * 20

ani = FuncAnimation(fig, update, frames=total_frames,
                    interval=120, blit=False, repeat=True)

out = OUT_DIR / "anomaly_demo.gif"
writer = PillowWriter(fps=8)
ani.save(str(out), writer=writer, dpi=110)
plt.close(fig)
print(f"✓  {out}")
print(f"   Size: {out.stat().st_size / 1024:.0f} KB")
print(f"   Frames: {len(total_frames)}  ·  Duration: {len(total_frames)/8:.1f}s")
