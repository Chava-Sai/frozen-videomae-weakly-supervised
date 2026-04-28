"""
generate_drawio_dark.py
Generates project_workflow_dark.drawio — dark-theme, draw.io-editable version
of the IVC system design (matches the PNG aesthetic).

Import at: https://app.diagrams.net  →  File → Import from → Device
"""

import io, xml.etree.ElementTree as ET

# ── palette (hex without #) ───────────────────────────────────────────────────
BG      = "#111111"
SEC_BG  = "#1a1a1a"
W_TEXT  = "#ffffff"
LG_TEXT = "#9ca3af"

# (fill, stroke)  — all dark-theme
BLUE   = ("#0d2240", "#2563eb")
INDIGO = ("#151040", "#6366f1")
VIOLET = ("#1a0e3f", "#7c3aed")
GREEN  = ("#052e1a", "#059669")
TEAL   = ("#02262e", "#0891b2")
AMBER  = ("#271500", "#d97706")
ORANGE = ("#2e1200", "#ea580c")
RED_T  = ("#2a0c0c", "#dc2626")
GRAY   = ("#1e1e1e", "#6b7280")
CYAN   = ("#012830", "#22d3ee")

# Section accents
AC1 = "#3b82f6"   # ① Feature Extraction — blue
AC2 = "#8b5cf6"   # ② Architecture        — violet
AC3 = "#10b981"   # ③ Evaluation          — emerald

ID = iter(range(2, 99999))
cells = []

def add(c): cells.append(c)

# ── XML cell helpers ──────────────────────────────────────────────────────────

def mk(uid, value, style, x, y, w, h, parent="1"):
    c = ET.Element("mxCell", id=str(uid), value=value,
                   style=style, vertex="1", parent=str(parent))
    g = ET.SubElement(c, "mxGeometry")
    g.set("x", str(int(x))); g.set("y", str(int(y)))
    g.set("width", str(int(w))); g.set("height", str(int(h)))
    g.set("as", "geometry")
    return c

def edge(uid, src, tgt, label="", color="#888888", dashed=False, parent="1"):
    dash = "dashed=1;dashPattern=6 3;" if dashed else ""
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;{dash}"
             f"strokeColor={color};strokeWidth=2;exitX=0.5;exitY=1;"
             f"exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
             f"fontColor={W_TEXT};fontSize=10;")
    c = ET.Element("mxCell", id=str(uid), value=label, style=style,
                   edge="1", source=str(src), target=str(tgt), parent=str(parent))
    g = ET.SubElement(c, "mxGeometry", relative="1")
    g.set("as", "geometry")
    return c

def hedge(uid, src, tgt, label="", color="#888888", parent="1"):
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;"
             f"strokeColor={color};strokeWidth=1.8;"
             f"exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
             f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
             f"endArrow=block;endFill=1;"
             f"fontColor={W_TEXT};fontSize=10;")
    c = ET.Element("mxCell", id=str(uid), value=label, style=style,
                   edge="1", source=str(src), target=str(tgt), parent=str(parent))
    g = ET.SubElement(c, "mxGeometry", relative="1")
    g.set("as", "geometry")
    return c

def node(x, y, w, h, fill, stroke, title, subs="", ft=13, fs=11):
    uid = next(ID)
    html = (f"<b><font color='{W_TEXT}' style='font-size:{ft}px'>{title}</font></b>"
            + (f"<br/><font color='{LG_TEXT}' style='font-size:{fs}px'>{subs}</font>" if subs else ""))
    style = (f"rounded=1;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
             f"fontColor={W_TEXT};fontSize={ft};fontStyle=1;"
             f"verticalAlign=middle;align=center;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def cyl(x, y, w, h, fill, stroke, title, subs=""):
    uid = next(ID)
    html = (f"<b><font color='{W_TEXT}' style='font-size:13px'>{title}</font></b>"
            + (f"<br/><font color='{LG_TEXT}' style='font-size:11px'>{subs}</font>" if subs else ""))
    style = (f"shape=cylinder3;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
             f"fontColor={W_TEXT};fontSize=13;fontStyle=1;"
             f"verticalAlign=middle;align=center;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def section(x, y, w, h, color, num, title):
    uid = next(ID)
    html = (f"<b><font color='{color}' style='font-size:15px'>"
            f"<font color='{W_TEXT}'>  {num}  </font>{title}</font></b>")
    style = (f"swimlane;startSize=38;rounded=1;whiteSpace=wrap;html=1;"
             f"fillColor={SEC_BG};strokeColor={color};strokeWidth=3;"
             f"fontColor={color};fontSize=15;fontStyle=1;"
             f"swimlaneLine=0;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def lbl(x, y, w, h, text, color=LG_TEXT, fs=11):
    uid = next(ID)
    style = (f"text;html=1;strokeColor=none;fillColor=none;"
             f"align=center;verticalAlign=middle;"
             f"fontColor={color};fontSize={fs};fontStyle=0;whiteSpace=wrap;")
    add(mk(uid, text, style, x, y, w, h))
    return uid

# ── canvas constants ───────────────────────────────────────────────────────────
# All coordinates in pixels (1 px unit in draw.io = 1 px)
# Approx. 2400 × 1600 canvas

CW = 2400   # total canvas width
CH = 1700   # total canvas height

# ─────────────────────────────────────────────────────────────────────────────
# TITLE BAR
# ─────────────────────────────────────────────────────────────────────────────
T_UID = next(ID)
add(mk(T_UID,
    f"<font style='font-size:22px;font-weight:bold;color:{W_TEXT};'>"
    "System Design: Weakly-Supervised Violence Detection with VideoMAE-B</font>"
    f"<br/><font style='font-size:13px;color:{LG_TEXT};'>"
    "Boston University CS &nbsp;·&nbsp; Srinivasa Sai Chava &nbsp;·&nbsp; "
    "RTFM + TRN + Boundary Head with Frozen VideoMAE-B Backbone</font>",
    f"text;html=1;strokeColor=none;fillColor=none;align=center;"
    f"verticalAlign=middle;fontColor={W_TEXT};fontSize=20;",
    0, 10, CW, 65))

# ─────────────────────────────────────────────────────────────────────────────
# ① FEATURE EXTRACTION  (top, full width)
# ─────────────────────────────────────────────────────────────────────────────
S1 = section(20, 85, CW-40, 520, AC1, "①", "FEATURE EXTRACTION PIPELINE — One-Time Preprocessing")

# Dataset
N_DATASET = node(40, 170, 260, 160, *BLUE,
    "UCF-Crime Dataset",
    "Sultani et al. CVPR 2018<br/>1,300 videos total<br/>6 violence categories<br/>Video-level weak labels")

# Frame sampler
N_SAMPLER = node(340, 170, 240, 160, *INDIGO,
    "Frame Sampler",
    "16-frame segments<br/>Non-overlapping<br/>Tail segment dropped<br/>T segments / video")

add(hedge(next(ID), N_DATASET, N_SAMPLER, color=AC1))

# I3D branch (lower)
N_I3D = node(640, 290, 300, 150, *ORANGE,
    "I3D Backbone",
    "Frozen · Kinetics-400<br/>Two-stream RGB + Flow<br/>D = 2048 features")

N_I3D_FEAT = node(990, 290, 240, 150, *AMBER,
    "I3D Features",
    "f_t ∈ R^2048<br/>1,300 .npz files<br/>Step-7 cache")

add(hedge(next(ID), N_I3D, N_I3D_FEAT, color=ORANGE[1]))

# VideoMAE branch (upper)
N_VM = node(640, 130, 340, 160, *BLUE,
    "VideoMAE-B Backbone  ★",
    "Frozen ViT-B · Kinetics-400<br/>MCG-NJU checkpoint<br/>D = 768 features<br/>BU SCC · NVIDIA A100")

N_VM_FEAT = node(1030, 130, 240, 160, *TEAL,
    "VideoMAE-B Features",
    "f_t ∈ R^768<br/>1,300 .npz files<br/>494.6 s · 0.38 s/video")

add(hedge(next(ID), N_VM, N_VM_FEAT, color=AC1))

# Branch arrow from sampler
add(hedge(next(ID), N_SAMPLER, N_VM, color=AC1))
add(hedge(next(ID), N_SAMPLER, N_I3D, color=ORANGE[1]))

# Cache cylinder
N_CACHE = cyl(1330, 155, 240, 280, *TEAL,
    "Feature Cache",
    "1,300 .npz per backbone<br/>Cached once · reused<br/>Resume-safe")

add(hedge(next(ID), N_VM_FEAT,  N_CACHE, color=AC1))
add(hedge(next(ID), N_I3D_FEAT, N_CACHE, color=ORANGE[1]))

# Comparison callout
N_COMP = node(1640, 170, 380, 210, *GRAY,
    "Feature Comparison",
    "I3D:       D=2048 · optical flow · 2017<br/>"
    "VideoMAE:  D=768  · masked AE   · 2022<br/><br/>"
    "Same architecture — only backbone changes<br/>"
    "Dimension -63%  →  less overfitting risk")

add(hedge(next(ID), N_CACHE, N_COMP, color="#6b7280"))

# Labels
lbl(640, 285, 100, 28, "Step-7 Baseline", ORANGE[1], 11)
lbl(640, 125, 160, 28, "Our Contribution  ★", AC1, 11)

# ─────────────────────────────────────────────────────────────────────────────
# ② MODEL ARCHITECTURE  (bottom-left)
# ─────────────────────────────────────────────────────────────────────────────
S2 = section(20, 620, 1110, 1060, AC2,
    "②", "MODEL ARCHITECTURE — 1.9M Trainable Parameters · 86M Frozen")

# MIL Scorer
N_MIL = node(40, 730, 320, 160, *VIOLET,
    "MIL Anomaly Scorer",
    "s_t = σ(W₃ Dropout(ReLU(W₂ ReLU(W₁ f_t))))<br/>"
    "Video score: s_vid = max_t(s_t)<br/>"
    "ℒ_MIL ranking loss · anom/norm pairs<br/>"
    "1.1 M params")

# Event Classifier
N_CLS = node(400, 730, 320, 160, *VIOLET,
    "Event Classifier",
    "7-class softmax · top-k pseudo segs<br/>"
    "Confidence filter τ = 0.30<br/>"
    "Inv-freq weighted CE loss<br/>"
    "0.3 M params")

add(hedge(next(ID), N_MIL, N_CLS, color=AC2))

# TRN
N_TRN = node(40, 940, 320, 160, *INDIGO,
    "Temporal Refinement (TRN)",
    "Transformer encoder over {f_t + PE_t}<br/>"
    "Multi-head self-attention<br/>"
    "Refined scores s̃_t  and  h̃_t<br/>"
    "0.4 M params")

# Boundary Head
N_BND = node(400, 940, 320, 160, *INDIGO,
    "Boundary Head",
    "b_t = σ(W_b [h̃_t ; h̃_{t+1} ; |s̃_t − s̃_{t+1}|])<br/>"
    "Boundary consistency loss ℒ_bnd<br/>"
    "Event start / end confidence<br/>"
    "0.1 M params")

add(hedge(next(ID), N_TRN, N_BND, color=AC2))
add(edge(next(ID), N_CLS, N_BND, color=AC2))

# Full objective
N_LOSS = node(40, 1155, 680, 160, *GRAY,
    "Full Training Objective",
    "ℒ = ℒ_MIL + λ₁ℒ_cls + λ₂ℒ_bnd + λ₃ℒ_smooth<br/>"
    "ℒ_smooth = Σ(s̃_t − s̃_{t-1})²<br/>"
    "λ₁=0.5 · λ₂=0.3 · λ₃=0.1<br/>"
    "AdamW · lr=1e-4 · WD=1e-4 · Dropout=0.5 · Batch=32")

add(edge(next(ID), N_BND, N_LOSS, color="#6b7280"))

# Training recipe pills
pill_data = [
    (VIOLET, "① Inv-Freq Weights",
     "w_c = N / (C · n_c)<br/>Normal: w ≈ 0.06<br/>Shooting: w ≈ 1.83<br/>Counters 10:1 imbalance"),
    (VIOLET, "② Conf. Threshold",
     "τ = 0.30<br/>Filter noisy pseudo-<br/>positive segments<br/>in early epochs"),
    (VIOLET, "③ Cosine Anneal LR",
     "η: 1e-4 → 1e-6<br/>T_max = 40 epochs<br/>No warm restarts<br/>No step-decay spikes"),
    (VIOLET, "④ 3-Seed Training",
     "Seeds: 42 · 123 · 456<br/>40 epochs each<br/>Report mean ± std<br/>Reproducibility check"),
]
pw = 240; pg = 15
for i, (th, ti, su) in enumerate(pill_data):
    node(40 + i*(pw+pg), 1370, pw, 160, *th, ti, su, ft=12, fs=10)

lbl(40, 1575, 1070, 30,
    "Total trainable: 1.9M  ·  Frozen backbone: 86M  ·  PyTorch 2.1  ·  Transformers 5.5.4  ·  BU SCC A100",
    LG_TEXT, 11)

# ─────────────────────────────────────────────────────────────────────────────
# ③ INFERENCE & EVALUATION  (bottom-right)
# ─────────────────────────────────────────────────────────────────────────────
S3X = 1150
S3 = section(S3X, 620, CW-S3X-20, 1060, AC3,
    "③", "INFERENCE & EVALUATION — UCF-Crime Violence Subset")

# Inference chain
N_MODEL = node(S3X+30, 720, 270, 140, *GREEN,
    "Trained Model",
    "Best checkpoint (seed 123)<br/>AUC = 93.4% peak epoch<br/>40-epoch training")

N_INF = node(S3X+340, 720, 260, 140, *GREEN,
    "Inference Pipeline",
    "Forward pass per segment<br/>Outputs: s̃_t · p_t · b_t<br/>Temporal smoothing")

N_POST = node(S3X+640, 720, 260, 140, *TEAL,
    "Post-Processing",
    "Threshold candidate segs.<br/>Merge adjacent positives<br/>Boundary peak refinement")

N_OUT = node(S3X+940, 720, 240, 140, *TEAL,
    "Event Tuples",
    "(t_start, t_end, class,<br/>confidence) per video<br/>Temporal localization")

add(hedge(next(ID), N_MODEL, N_INF,  color=AC3))
add(hedge(next(ID), N_INF,  N_POST, color=AC3))
add(hedge(next(ID), N_POST, N_OUT,  color=AC3))

# Metric boxes
lbl(S3X+30, 895, 1200, 30,
    "─────────────  Evaluation Metrics  ─────────────",
    AC3, 13)

N_M1 = node(S3X+30,  930, 370, 120, *GREEN, "Binary Detection",
    "AUC (ROC curve) · AP (PR curve)<br/>Video-level anomaly scores")
N_M2 = node(S3X+420, 930, 370, 120, *TEAL,  "Multi-Class F1",
    "Macro-F1 (6 classes, balanced)<br/>Weighted-F1 (freq-weighted)")
N_M3 = node(S3X+810, 930, 380, 120, *CYAN,  "Temporal Localization",
    "mAP @ tIoU = 0.3 / 0.5 / 0.7<br/>Segment-level event detection")

add(edge(next(ID), N_M1, N_INF, color=AC3, dashed=True))
add(edge(next(ID), N_M2, N_INF, color=AC3, dashed=True))
add(edge(next(ID), N_M3, N_INF, color=AC3, dashed=True))

# Results table header
lbl(S3X+30, 1075, 1200, 30,
    "─────  Results: I3D Baseline  vs  VideoMAE-B (3 Seeds, Mean ± Std)  ─────",
    AC3, 13)

col_x = [S3X+30, S3X+240, S3X+560, S3X+900]
col_w = [200,     310,      330,      290]

# Header
for cx, cw, txt, th in zip(col_x, col_w,
    ["Metric", "I3D Baseline", "VideoMAE-B (mean ± std)", "Gain"],
    [GRAY, RED_T, BLUE, GREEN]):
    node(cx, 1110, cw, 50, *th, txt, "", ft=13)

# Data rows
result_rows = [
    ("Binary AUC",    "87.4%",     "92.2% ± 1.3%",   "+4.8 pp"),
    ("Binary AP",     "82.1%",     "88.7% ± 1.8%",   "+8%"),
    ("Macro-F1",      "19.8%",     "25.5% ± 2.7%",   "+28%"),
    ("mAP @ IoU 0.3", "0.009",     "0.058 ± 0.012",  "x 6.4"),
    ("mAP @ IoU 0.5", "0.004",     "0.038 ± 0.008",  "x 9.4"),
    ("mAP @ IoU 0.7", "0.001",     "0.022 ± 0.010",  "x 21.7"),
]
rh = 52; ry0 = 1170
for i, (met, i3d, vm, gn) in enumerate(result_rows):
    ry = ry0 + i*(rh+4)
    for cx, cw, val, th in zip(col_x, col_w, [met, i3d, vm, gn],
                                [GRAY, RED_T, BLUE, GREEN]):
        node(cx, ry, cw, rh, *th, val, "", ft=12)

# Publication outputs
lbl(S3X+30, 1498, 1200, 28, "Publication Roadmap", AC3, 13)
pub_data = [
    (GREEN,  "arXiv Preprint",
     "Submit immediately<br/>Establishes priority<br/>Public timestamp"),
    (TEAL,   "ACM MM 2026 Workshop",
     "Deadline: July 16, 2026<br/>6-page ACM SIGCONF<br/>Current results sufficient"),
    (CYAN,   "WACV 2027",
     "~Aug 2026 deadline<br/>Full paper + ablations<br/>+ cross-dataset eval"),
    (GREEN,  "CVPR 2027",
     "~Nov 2026 deadline<br/>Add VideoMAE-Large<br/>+ XD-Violence eval"),
]
pw2 = 285; pg2 = 12
for i, (th, ti, su) in enumerate(pub_data):
    node(S3X+30 + i*(pw2+pg2), 1530, pw2, 130, *th, ti, su, ft=12, fs=10)

lbl(S3X+30, 1678, 1200, 28,
    "Embeddings: VideoMAE-B 768-dim · Kinetics-400  ·  Seeds 42/123/456  ·  40 epochs · AdamW  ·  BU SCC A100",
    LG_TEXT, 11)

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SECTION CONNECTORS
# ─────────────────────────────────────────────────────────────────────────────
# ① cache → ② model input
add(edge(next(ID), N_CACHE, N_MIL,
    label="VideoMAE-B features", color=AC2, dashed=True))

# ① cache → ③ inference
add(edge(next(ID), N_CACHE, N_MODEL,
    label="test features", color=AC3, dashed=True))

# ② trained model → ③
add(hedge(next(ID), N_BND, N_MODEL,
    label="weights", color="#6b7280"))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD XML
# ─────────────────────────────────────────────────────────────────────────────
graph = ET.Element("mxGraphModel",
    dx="1500", dy="900",
    grid="1", gridSize="10",
    guides="1", tooltips="1",
    connect="1", arrows="1", fold="1",
    page="0", pageScale="1",
    pageWidth=str(CW), pageHeight=str(CH),
    math="0", shadow="1",
    background=BG)

root_n = ET.SubElement(graph, "root")
ET.SubElement(root_n, "mxCell", id="0")
ET.SubElement(root_n, "mxCell", id="1", parent="0")
for c in cells:
    root_n.append(c)

out_path = "project_workflow_dark.drawio"
ET.indent(graph, space="  ")
buf = io.StringIO()
ET.ElementTree(graph).write(buf, encoding="unicode", xml_declaration=False)

with open(out_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<mxfile host="app.diagrams.net" modified="2026-04-17" agent="Claude">\n')
    f.write('  <diagram name="IVC System Design" id="ivc-system">')
    f.write(buf.getvalue())
    f.write('</diagram>\n</mxfile>\n')

print(f"✓  {out_path}")
print(f"   Cells: {len(cells)}  ·  Canvas: {CW}×{CH} px")
print()
print("How to open:")
print("  1.  https://app.diagrams.net")
print("  2.  File → Import from → Device → select project_workflow_dark.drawio")
print("  3.  Or drag-and-drop the file onto diagrams.net")
print()
print("Dark background tip:")
print("  View → Page → Background Color → #111111")
