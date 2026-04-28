"""
generate_workflow.py
Generates project_workflow.drawio — complete end-to-end IVC project diagram.
Run:  python3 generate_workflow.py
Then: open project_workflow.drawio in https://app.diagrams.net (File → Import)
"""

import textwrap, xml.etree.ElementTree as ET

# ── helpers ────────────────────────────────────────────────────────────────────

def cell(uid, value, style, x, y, w, h, parent="1"):
    c = ET.Element("mxCell", id=str(uid), value=value, style=style,
                   vertex="1", parent=str(parent))
    g = ET.SubElement(c, "mxGeometry")
    g.set("x", str(x)); g.set("y", str(y))
    g.set("width", str(w)); g.set("height", str(h))
    g.set("as", "geometry")
    return c

def edge(uid, src, tgt, label="", style=None, parent="1"):
    default = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
               "jettySize=auto;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
               "entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
               "strokeWidth=2.5;strokeColor=#555555;fontSize=11;")
    c = ET.Element("mxCell", id=str(uid), value=label,
                   style=style or default,
                   edge="1", source=str(src), target=str(tgt), parent=str(parent))
    g = ET.SubElement(c, "mxGeometry", relative="1")
    g.set("as", "geometry")
    return c

def hedge(uid, src, tgt, label="", color="#555555", parent="1"):
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
             f"jettySize=auto;exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
             f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
             f"strokeWidth=1.8;strokeColor={color};endArrow=block;endFill=1;"
             f"fontSize=10;")
    c = ET.Element("mxCell", id=str(uid), value=label, style=style,
                   edge="1", source=str(src), target=str(tgt), parent=str(parent))
    g = ET.SubElement(c, "mxGeometry", relative="1")
    g.set("as", "geometry")
    return c

# Phase container (big rounded rect with title in top-left)
def phase_box(uid, title_html, x, y, w, h, fill, stroke, font="#ffffff"):
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor={stroke};strokeWidth=3;fontSize=14;fontStyle=1;"
             f"verticalAlign=top;align=left;spacingTop=8;spacingLeft=14;"
             f"fontColor={font};")
    return cell(uid, title_html, style, x, y, w, h)

# Regular content node
def node(uid, html, x, y, w, h, fill, stroke, font_size=11, bold=False):
    fs = "fontStyle=1;" if bold else ""
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor={stroke};fontSize={font_size};{fs}"
             f"verticalAlign=middle;align=center;")
    return cell(uid, html, style, x, y, w, h)

# Annotation / label (no border)
def label(uid, html, x, y, w, h, color="#555555", font_size=10):
    style = (f"text;html=1;strokeColor=none;fillColor=none;"
             f"align=center;verticalAlign=middle;fontSize={font_size};"
             f"fontColor={color};whiteSpace=wrap;")
    return cell(uid, html, style, x, y, w, h)

# Divider line
def divider(uid, x, y, w, h, color="#cccccc"):
    style = (f"line;strokeColor={color};strokeWidth=2;fillColor=none;")
    return cell(uid, "", style, x, y, w, h)

# ── layout constants ───────────────────────────────────────────────────────────
LM   = 80          # left margin
CW   = 1640        # canvas width
MX   = LM + CW    # right edge = 1720
cx   = LM + CW//2  # centre x = 900

IDs = iter(range(2, 9999))
cells = []
def add(*cs): cells.extend(cs)

# ── TITLE ─────────────────────────────────────────────────────────────────────
TITLE_ID = next(IDs)
add(cell(TITLE_ID,
    "<div style='font-size:22px;font-weight:bold;color:#1a237e;'>"
    "IVC Project — Complete End-to-End Workflow</div>"
    "<div style='font-size:13px;color:#555;margin-top:3px;'>"
    "Boston University CS &nbsp;·&nbsp; "
    "Weakly-Supervised Violence Detection in Surveillance Video &nbsp;·&nbsp; "
    "Srinivasa Sai Chava</div>",
    "text;html=1;strokeColor=none;fillColor=none;align=center;"
    "verticalAlign=middle;fontSize=14;",
    LM, 15, CW, 65))

# ── PHASE 1 — PROBLEM DEFINITION ──────────────────────────────────────────────
P1Y = 95; P1H = 125
P1_ID = next(IDs)
add(phase_box(P1_ID,
    "📌  PHASE 1 — Problem Definition",
    LM, P1Y, CW, P1H, fill="#dde1ff", stroke="#3730a3", font="#1e1b4b"))

# Three content boxes inside
p1n = [next(IDs) for _ in range(3)]
add(node(p1n[0],
    "<b>Research Question</b><br/>Can we detect, classify &amp; temporally<br/>"
    "localize violence events using only<br/>video-level weak labels?",
    LM+20, P1Y+48, 500, 65, "#eef2ff", "#4338ca"))
add(node(p1n[1],
    "<b>Core Approach</b><br/>Multiple Instance Learning (MIL) +<br/>"
    "Temporal Refinement + Boundary prediction<br/>with frozen backbone features",
    LM+570, P1Y+48, 500, 65, "#eef2ff", "#4338ca"))
add(node(p1n[2],
    "<b>Key Innovation</b><br/>Replace I3D (2017) backbone with<br/>"
    "VideoMAE-B (2022) masked autoencoder<br/>— same architecture, better features",
    LM+1120, P1Y+48, 500, 65, "#eef2ff", "#4338ca"))

# ── ARROW P1 → P2 ─────────────────────────────────────────────────────────────
ARR1 = next(IDs)

# ── PHASE 2 — DATASET ─────────────────────────────────────────────────────────
P2Y = P1Y + P1H + 30; P2H = 140
P2_ID = next(IDs)
add(phase_box(P2_ID,
    "📁  PHASE 2 — Dataset: UCF-Crime Violence Subset",
    LM, P2Y, CW, P2H, fill="#dcfce7", stroke="#15803d", font="#14532d"))

p2n = [next(IDs) for _ in range(4)]
add(node(p2n[0],
    "<b>UCF-Crime Dataset</b><br/>Sultani et al. CVPR 2018<br/>"
    "Surveillance camera footage",
    LM+20, P2Y+50, 370, 70, "#f0fdf4", "#16a34a"))
add(node(p2n[1],
    "<b>1,300 Videos</b><br/>~950 training · ~350 test<br/>"
    "Violence subset only",
    LM+420, P2Y+50, 340, 70, "#f0fdf4", "#16a34a"))
add(node(p2n[2],
    "<b>6 Categories</b><br/>Abuse · Explosion · Fighting<br/>"
    "Robbery · Shooting · Normal",
    LM+790, P2Y+50, 370, 70, "#f0fdf4", "#16a34a"))
add(node(p2n[3],
    "<b>Weak Supervision Only</b><br/>Video-level binary labels +<br/>"
    "category label (no frame annotations)",
    LM+1190, P2Y+50, 430, 70, "#f0fdf4", "#16a34a"))

add(edge(ARR1, P1_ID, P2_ID))
ARR2 = next(IDs)

# ── PHASE 3 — FEATURE EXTRACTION (two columns) ────────────────────────────────
P3Y = P2Y + P2H + 30; P3H = 310
P3_ID = next(IDs)
add(phase_box(P3_ID,
    "⚙️  PHASE 3 — Feature Extraction  (two pipelines, same 1,300 videos)",
    LM, P3Y, CW, P3H, fill="#f5f5f5", stroke="#737373", font="#404040"))

# LEFT: I3D Baseline ───────────────────────────────────────────────
I3D_BG = next(IDs)
add(phase_box(I3D_BG, "🔴  I3D Baseline (Step-7)",
    LM+15, P3Y+48, 775, P3H-60, fill="#fff7ed", stroke="#c2410c", font="#7c2d12"))

i3d_nodes = [next(IDs) for _ in range(4)]
add(node(i3d_nodes[0],
    "<b>Frozen I3D</b><br/>Pre-trained Kinetics-400<br/>Two-stream (RGB + Flow)",
    LM+30, P3Y+105, 350, 70, "#ffedd5", "#ea580c"))
add(node(i3d_nodes[1],
    "<b>16-frame segments</b><br/>Non-overlapping · tail drop<br/>Video → {v₁,…,vT}",
    LM+400, P3Y+105, 370, 70, "#ffedd5", "#ea580c"))
add(node(i3d_nodes[2],
    "<b>Feature dim D = 2048</b><br/>f_t ∈ ℝ²⁰⁴⁸<br/>Higher risk of overfitting",
    LM+30, P3Y+205, 350, 70, "#ffedd5", "#ea580c"))
add(node(i3d_nodes[3],
    "<b>Output: 1,300 .npz files</b><br/>Cached to disk · reused every run<br/>"
    "Step-7 pipeline",
    LM+400, P3Y+205, 370, 70, "#ffedd5", "#ea580c"))
add(hedge(next(IDs), i3d_nodes[0], i3d_nodes[1], color="#ea580c"))
add(hedge(next(IDs), i3d_nodes[2], i3d_nodes[3], color="#ea580c"))

# RIGHT: VideoMAE-B ────────────────────────────────────────────────
VM_BG = next(IDs)
add(phase_box(VM_BG, "🔵  VideoMAE-B  (Our Upgrade)",
    LM+850, P3Y+48, 775, P3H-60, fill="#dbeafe", stroke="#1d4ed8", font="#1e3a8a"))

vm_nodes = [next(IDs) for _ in range(5)]
add(node(vm_nodes[0],
    "<b>Frozen VideoMAE-B</b><br/>MCG-NJU · Pre-trained Kinetics-400<br/>"
    "ViT-B masked autoencoder",
    LM+865, P3Y+105, 350, 70, "#eff6ff", "#2563eb"))
add(node(vm_nodes[1],
    "<b>16-frame segments</b><br/>Non-overlapping · tail drop<br/>Identical to I3D pipeline",
    LM+1235, P3Y+105, 375, 70, "#eff6ff", "#2563eb"))
add(node(vm_nodes[2],
    "<b>Feature dim D = 768</b><br/>f_t ∈ ℝ⁷⁶⁸ · 3× smaller<br/>Reduces overfitting risk",
    LM+865, P3Y+205, 230, 70, "#eff6ff", "#2563eb"))
add(node(vm_nodes[3],
    "<b>BU SCC · NVIDIA A100</b><br/>494.6 s total · 0.38 s/video<br/>"
    "Transformers 5.5.4",
    LM+1115, P3Y+205, 240, 70, "#eff6ff", "#2563eb"))
add(node(vm_nodes[4],
    "<b>Output: 1,300 .npz files</b><br/>Cached · resume-safe<br/>"
    "0 failures",
    LM+1375, P3Y+205, 235, 70, "#eff6ff", "#2563eb"))
add(hedge(next(IDs), vm_nodes[0], vm_nodes[1], color="#2563eb"))
add(hedge(next(IDs), vm_nodes[2], vm_nodes[3], color="#2563eb"))
add(hedge(next(IDs), vm_nodes[3], vm_nodes[4], color="#2563eb"))

add(edge(ARR2, P2_ID, P3_ID))
ARR3 = next(IDs)

# ── PHASE 4 — MODEL ARCHITECTURE ──────────────────────────────────────────────
P4Y = P3Y + P3H + 30; P4H = 220
P4_ID = next(IDs)
add(phase_box(P4_ID,
    "🧠  PHASE 4 — Model Architecture  (Shared · 1.9M trainable params · 86M frozen)",
    LM, P4Y, CW, P4H, fill="#ede9fe", stroke="#7c3aed", font="#4c1d95"))

arch = [next(IDs) for _ in range(4)]
AW = 370; AGAP = 40
xs = [LM+20, LM+20+AW+AGAP, LM+20+2*(AW+AGAP), LM+20+3*(AW+AGAP)]
labels_arch = [
    ("<b>MIL Anomaly Scorer</b><br/>2-layer MLP · σ(W₃ Dropout(ReLU(…)))<br/>"
     "s_t ∈ [0,1] per segment<br/>ℒ_MIL ranking loss<br/><i>1.1M params</i>"),
    ("<b>Event Classifier</b><br/>7-class softmax head<br/>"
     "Top-k pseudo-positive segments<br/>Inv-freq weighted CE loss<br/><i>0.3M params</i>"),
    ("<b>Temporal Refinement (TRN)</b><br/>Transformer encoder<br/>"
     "Multi-head self-attention over {f_t}<br/>Refined scores s̃_t<br/><i>0.4M params</i>"),
    ("<b>Boundary Head</b><br/>b_t = σ(W_b[h̃_t; h̃_{t+1}; |s̃_t − s̃_{t+1}|])<br/>"
     "Boundary consistency loss ℒ_bnd<br/>Predicts event start/end<br/><i>0.1M params</i>"),
]
for i, (aid, lbl) in enumerate(zip(arch, labels_arch)):
    add(node(aid, lbl, xs[i], P4Y+52, AW, 150, "#f5f3ff", "#7c3aed", font_size=11))
    if i < 3:
        add(hedge(next(IDs), arch[i], arch[i+1], color="#7c3aed"))

add(edge(ARR3, P3_ID, P4_ID))
ARR4 = next(IDs)

# ── PHASE 5 — TRAINING RECIPE ─────────────────────────────────────────────────
P5Y = P4Y + P4H + 30; P5H = 190
P5_ID = next(IDs)
add(phase_box(P5_ID,
    "🏋️  PHASE 5 — Training Recipe  (3 Innovations beyond vanilla RTFM)",
    LM, P5Y, CW, P5H, fill="#fef3c7", stroke="#d97706", font="#78350f"))

tr_nodes = [next(IDs) for _ in range(4)]
tw = 375; tgap = 30
txs = [LM+20, LM+20+tw+tgap, LM+20+2*(tw+tgap), LM+20+3*(tw+tgap)]
tr_labels = [
    ("<b>① Inverse-Frequency<br/>Class Weights</b><br/>"
     "w_c = N / (C · n_c)<br/>Normal: w ≈ 0.06<br/>Shooting: w ≈ 1.83<br/>"
     "Counters 10:1 imbalance"),
    ("<b>② Confidence Threshold</b><br/>τ = 0.30<br/>"
     "Pseudo-positive segments<br/>accepted only if s_t ≥ τ<br/>"
     "Reduces noisy label training<br/>in early epochs"),
    ("<b>③ Cosine Annealing LR</b><br/>η: 10⁻⁴ → 10⁻⁶<br/>"
     "T_max = 40 epochs<br/>No warm restarts<br/>"
     "Avoids step-decay oscillations"),
    ("<b>④ Multi-Seed Training</b><br/>Seeds: 42 · 123 · 456<br/>"
     "40 epochs each<br/>AdamW · batch 32<br/>"
     "Dropout 0.5 · WD 10⁻⁴"),
]
for i, (tid, lbl) in enumerate(zip(tr_nodes, tr_labels)):
    add(node(tid, lbl, txs[i], P5Y+48, tw, 130, "#fffbeb", "#d97706", font_size=11))

add(edge(ARR4, P4_ID, P5_ID))
ARR5 = next(IDs)

# ── PHASE 6 — EVALUATION METRICS ──────────────────────────────────────────────
P6Y = P5Y + P5H + 30; P6H = 140
P6_ID = next(IDs)
add(phase_box(P6_ID,
    "📐  PHASE 6 — Evaluation Metrics",
    LM, P6Y, CW, P6H, fill="#ccfbf1", stroke="#0d9488", font="#134e4a"))

ev_nodes = [next(IDs) for _ in range(3)]
ew = 505; egap = 30
exs = [LM+20, LM+20+ew+egap, LM+20+2*(ew+egap)]
ev_labels = [
    "<b>Binary Detection</b><br/>AUC (ROC) · AP (PR curve)<br/>Video-level anomaly scores",
    ("<b>Multi-Class Classification</b><br/>Macro-F1 (6 classes, balanced)<br/>"
     "Weighted-F1 (frequency-weighted)"),
    ("<b>Temporal Localization</b><br/>mAP @ tIoU = 0.3 / 0.5 / 0.7<br/>"
     "Segment-level event detection"),
]
for i, (eid, lbl) in enumerate(zip(ev_nodes, ev_labels)):
    add(node(eid, lbl, exs[i], P6Y+50, ew, 75, "#f0fdfa", "#0d9488", font_size=11))

add(edge(ARR5, P5_ID, P6_ID))
ARR6 = next(IDs)

# ── PHASE 7 — RESULTS ─────────────────────────────────────────────────────────
P7Y = P6Y + P6H + 30; P7H = 280
P7_ID = next(IDs)
add(phase_box(P7_ID,
    "📊  PHASE 7 — Results  (Mean ± Std, 3 Seeds — VideoMAE-B dominates every metric)",
    LM, P7Y, CW, P7H, fill="#f0fdf4", stroke="#15803d", font="#14532d"))

# Column headers
RCW = 490  # result column width
RGP = 25
rx = [LM+20, LM+20+RCW+RGP, LM+20+2*(RCW+RGP)]
add(node(next(IDs), "<b>Metric</b>",
    LM+20, P7Y+50, 200, 40, "#dcfce7", "#15803d", font_size=12, bold=True))
add(node(next(IDs), "<b>🔴  I3D Baseline</b>",
    rx[0]+220, P7Y+50, 270, 40, "#fee2e2", "#b91c1c", font_size=12, bold=True))
add(node(next(IDs), "<b>🔵  VideoMAE-B (mean ± std)</b>",
    rx[1], P7Y+50, RCW, 40, "#dbeafe", "#1d4ed8", font_size=12, bold=True))
add(node(next(IDs), "<b>🟢  Gain</b>",
    rx[2], P7Y+50, RCW, 40, "#dcfce7", "#15803d", font_size=12, bold=True))

# Result rows
result_rows = [
    ("Binary AUC",    "87.4%",         "92.2% ± 1.3%",        "+4.8 pp  ↑"),
    ("Binary AP",     "82.1%",         "88.7% ± 1.8%",        "+6.6 pp  ↑"),
    ("Macro-F1",      "19.8%",         "25.5% ± 2.7%",        "+28%  ↑"),
    ("Weighted-F1",   "65.4%",         "71.9% ± 2.2%",        "+10%  ↑"),
    ("mAP @ IoU 0.3", "0.009",         "0.058 ± 0.012",       "× 6.4  ↑↑"),
    ("mAP @ IoU 0.5", "0.004",         "0.038 ± 0.008",       "× 9.4  ↑↑"),
    ("mAP @ IoU 0.7", "0.001",         "0.022 ± 0.010",       "× 21.7  ↑↑↑"),
]
row_h = 28; row_y0 = P7Y + 100
for i, (metric, i3d_val, vm_val, gain) in enumerate(result_rows):
    ry = row_y0 + i * (row_h + 4)
    i3d_fill = "#fef2f2"; vm_fill = "#eff6ff"; gn_fill = "#f0fdf4"
    add(node(next(IDs), f"<b>{metric}</b>",
        LM+20, ry, 200, row_h, "#e7f6e7", "#15803d", font_size=11))
    add(node(next(IDs), f"<font color='#b91c1c'>{i3d_val}</font>",
        rx[0]+220, ry, 270, row_h, i3d_fill, "#fca5a5", font_size=11))
    add(node(next(IDs), f"<font color='#1d4ed8'><b>{vm_val}</b></font>",
        rx[1], ry, RCW, row_h, vm_fill, "#93c5fd", font_size=11))
    add(node(next(IDs), f"<font color='#15803d'><b>{gain}</b></font>",
        rx[2], ry, RCW, row_h, gn_fill, "#86efac", font_size=11))

add(edge(ARR6, P6_ID, P7_ID))
ARR7 = next(IDs)

# ── PHASE 8 — VISUALIZATIONS ──────────────────────────────────────────────────
P8Y = P7Y + P7H + 30; P8H = 155
P8_ID = next(IDs)
add(phase_box(P8_ID,
    "🎨  PHASE 8 — Publication Figures  (6 matplotlib figures · dpi=200)",
    LM, P8Y, CW, P8H, fill="#dbeafe", stroke="#2563eb", font="#1e3a8a"))

fig_labels = [
    "<b>Fig 1</b><br/>Training Dynamics<br/>(Loss · AUC · F1)",
    "<b>Fig 2</b><br/>Dumbbell Chart<br/>(I3D vs VideoMAE-B)",
    "<b>Fig 3</b><br/>mAP vs tIoU<br/>Localization Curve",
    "<b>Fig 4</b><br/>Cross-Seed<br/>Reproducibility",
    "<b>Fig 5</b><br/>Per-Class F1<br/>Analysis",
    "<b>Fig 6</b><br/>All-Metrics<br/>Radar Chart",
]
fw = 248; fgap = 12
for i, lbl in enumerate(fig_labels):
    add(node(next(IDs), lbl,
        LM+20 + i*(fw+fgap), P8Y+55, fw, 85, "#eff6ff", "#2563eb", font_size=11))

add(edge(ARR7, P7_ID, P8_ID))
ARR8 = next(IDs)

# ── PHASE 9 — PUBLICATION ─────────────────────────────────────────────────────
P9Y = P8Y + P8H + 30; P9H = 155
P9_ID = next(IDs)
add(phase_box(P9_ID,
    "📝  PHASE 9 — Publication Plan",
    LM, P9Y, CW, P9H, fill="#f3e8ff", stroke="#7c3aed", font="#4c1d95"))

pub_labels = [
    ("<b>① arXiv Preprint</b><br/>Post immediately<br/>Establishes priority<br/>"
     "Public timestamp"),
    ("<b>② ACM MM 2026 Workshop</b><br/>Deadline: July 16, 2026<br/>"
     "6-page sigconf format<br/>Current results sufficient"),
    ("<b>③ WACV 2027</b><br/>Deadline: ~Aug 2026<br/>"
     "Full paper · add ablations<br/>+ cross-dataset eval"),
    ("<b>④ CVPR 2027</b><br/>Deadline: ~Nov 2026<br/>"
     "Top-tier · needs XD-Violence<br/>+ VideoMAE-L + ablations"),
]
pw = 380; pgap = 22
for i, lbl in enumerate(pub_labels):
    add(node(next(IDs), lbl,
        LM+20 + i*(pw+pgap), P9Y+55, pw, 85, "#faf5ff", "#9333ea", font_size=11))

add(edge(ARR8, P8_ID, P9_ID))

# ── SIDE ANNOTATIONS ──────────────────────────────────────────────────────────
# "SCC GPU" badge on Phase 3 right column
add(label(next(IDs),
    "<font color='#1d4ed8'><b>BU SCC</b><br/>NVIDIA A100<br/>40 GB</font>",
    MX - 70, P3Y + 120, 80, 60, "#1d4ed8", 10))

# "Training time" annotation
add(label(next(IDs),
    "<font color='#d97706'>40 epochs × 3 seeds<br/>≈ 15 min/run on A100</font>",
    MX - 140, P5Y + 140, 140, 40, "#d97706", 10))

# ── ASSEMBLE XML ──────────────────────────────────────────────────────────────
root_el = ET.Element("mxGraphModel",
    dx="1800", dy="900",
    grid="0", gridSize="10",
    guides="1", tooltips="1",
    connect="1", arrows="1", fold="1",
    page="0", pageScale="1",
    pageWidth="1800", pageHeight=str(P9Y + P9H + 80),
    math="0", shadow="1")

root_node = ET.SubElement(root_el, "root")
ET.SubElement(root_node, "mxCell", id="0")
ET.SubElement(root_node, "mxCell", id="1", parent="0")
for c in cells:
    root_node.append(c)

diagram_el = ET.Element("diagram", name="IVC Project Workflow", id="ivc-wf")
diagram_el.text = ""  # will be overwritten
mxfile_el = ET.Element("mxfile", host="app.diagrams.net",
                        modified="2026-04-17", agent="Claude", version="21.0.0")
mxfile_el.append(diagram_el)

# Serialize mxGraphModel to string and embed as diagram content
import io
ET.indent(root_el, space="  ")
buf = io.StringIO()
ET.ElementTree(root_el).write(buf, encoding="unicode", xml_declaration=False)
diagram_el.text = buf.getvalue()

# Write final file
out_path = "project_workflow.drawio"
with open(out_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<mxfile host="app.diagrams.net" modified="2026-04-17" agent="Claude" version="21.0.0">\n')
    f.write(f'  <diagram name="IVC Project Workflow" id="ivc-wf">')
    f.write(buf.getvalue())
    f.write('</diagram>\n</mxfile>\n')

print(f"✓  Written → {out_path}")
print(f"   Total cells  : {len(cells)}")
print(f"   Canvas height: {P9Y + P9H + 80} px")
print(f"   Phases       : 9")
print()
print("How to open:")
print("  1. Go to https://app.diagrams.net")
print("  2. File → Import from → Device")
print("  3. Select project_workflow.drawio")
