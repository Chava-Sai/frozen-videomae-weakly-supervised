"""
generate_drawio_light.py
White-background draw.io system design — matches the example style
(light tinted sections, dashed colored borders, dark text).
"""

import io, xml.etree.ElementTree as ET

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#ffffff"
DARK   = "#111827"
MED    = "#374151"
SUBTLE = "#6b7280"

# Section background tints
S1_BG = "#eff6ff"   # light blue   ①
S2_BG = "#fff7ed"   # light orange ②
S3_BG = "#f0fdf4"   # light green  ③

# Section accent (border) colors
AC1 = "#3b82f6"   # blue
AC2 = "#f59e0b"   # amber
AC3 = "#10b981"   # emerald

# Node (fill, stroke) pairs — light fills
BLUE   = ("#dbeafe", "#2563eb")
INDIGO = ("#e0e7ff", "#4f46e5")
VIOLET = ("#ede9fe", "#7c3aed")
GREEN  = ("#d1fae5", "#059669")
TEAL   = ("#ccfbf1", "#0d9488")
AMBER  = ("#fef3c7", "#d97706")
ORANGE = ("#ffedd5", "#ea580c")
GRAY   = ("#f3f4f6", "#6b7280")
CYAN   = ("#cffafe", "#0891b2")
RED_T  = ("#fee2e2", "#dc2626")
PURPLE = ("#f5f3ff", "#7c3aed")

ID    = iter(range(2, 99999))
cells = []
def add(c): cells.append(c)

# ── XML helpers ───────────────────────────────────────────────────────────────
def mk(uid, value, style, x, y, w, h, parent="1"):
    c = ET.Element("mxCell", id=str(uid), value=value,
                   style=style, vertex="1", parent=str(parent))
    g = ET.SubElement(c, "mxGeometry")
    g.set("x", str(int(x))); g.set("y", str(int(y)))
    g.set("width",  str(int(w))); g.set("height", str(int(h)))
    g.set("as", "geometry")
    return c

def _ecell(uid, src, tgt, label, style):
    c = ET.Element("mxCell", id=str(uid), value=label, style=style,
                   edge="1", source=str(src), target=str(tgt), parent="1")
    g = ET.SubElement(c, "mxGeometry", relative="1")
    g.set("as", "geometry")
    return c

def hedge(uid, src, tgt, label="", color="#888888", dashed=False):
    d = "dashed=1;dashPattern=8 4;" if dashed else ""
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;{d}"
             f"strokeColor={color};strokeWidth=1.8;"
             f"exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
             f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
             f"endArrow=block;endFill=1;fontColor={MED};fontSize=10;")
    return _ecell(uid, src, tgt, label, style)

def vedge(uid, src, tgt, label="", color="#888888", dashed=False):
    d = "dashed=1;dashPattern=8 4;" if dashed else ""
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;{d}"
             f"strokeColor={color};strokeWidth=1.8;"
             f"endArrow=block;endFill=1;fontColor={MED};fontSize=10;")
    return _ecell(uid, src, tgt, label, style)

# ── Shape helpers ─────────────────────────────────────────────────────────────
def node(x, y, w, h, fill, stroke, title, subs="", ft=12, fs=10):
    uid = next(ID)
    html = (f"<b><font color='{DARK}' style='font-size:{ft}px'>{title}</font></b>"
            + (f"<br/><font color='{MED}' style='font-size:{fs}px'>{subs}</font>"
               if subs else ""))
    style = (f"rounded=1;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
             f"fontColor={DARK};fontSize={ft};fontStyle=1;"
             f"verticalAlign=middle;align=center;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def cyl(x, y, w, h, fill, stroke, title, subs=""):
    uid = next(ID)
    html = (f"<b><font color='{DARK}' style='font-size:12px'>{title}</font></b>"
            + (f"<br/><font color='{MED}' style='font-size:10px'>{subs}</font>"
               if subs else ""))
    style = (f"shape=cylinder3;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
             f"fontColor={DARK};fontSize=12;fontStyle=1;"
             f"verticalAlign=middle;align=center;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def sec(x, y, w, h, fill, stroke, num, title):
    """Dashed rounded section background (like the example)."""
    uid = next(ID)
    html = (f"<b><font color='{stroke}' style='font-size:14px'>"
            f"<font style='border:1.5px solid {stroke};"
            f"padding:0 5px;border-radius:4px;'>{num}</font>"
            f"&nbsp;&nbsp;{title}</font></b>")
    style = (f"rounded=1;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;"
             f"fillColor={fill};strokeColor={stroke};strokeWidth=2.5;"
             f"fontColor={stroke};fontSize=14;fontStyle=1;"
             f"verticalAlign=top;align=left;spacingLeft=14;spacingTop=10;")
    add(mk(uid, html, style, x, y, w, h))
    return uid

def lbl(x, y, w, h, text, color=SUBTLE, fs=10, bold=False):
    uid = next(ID)
    fw = "1" if bold else "0"
    style = (f"text;html=1;strokeColor=none;fillColor=none;"
             f"align=center;verticalAlign=middle;"
             f"fontColor={color};fontSize={fs};fontStyle={fw};whiteSpace=wrap;")
    add(mk(uid, text, style, x, y, w, h))
    return uid

# ── Canvas ────────────────────────────────────────────────────────────────────
CW, CH = 2400, 1720

# ═════════════════════════════════════════════════════════════════════════════
# TITLE BAR
# ═════════════════════════════════════════════════════════════════════════════
add(mk(next(ID),
    f"<b><font style='font-size:20px;color:{DARK};'>"
    "System Design — Weakly-Supervised Violence Detection with VideoMAE-B</font></b>"
    f"<br/><font style='font-size:11px;color:{SUBTLE};'>"
    "Boston University &nbsp;·&nbsp; Srinivasa Sai Chava &nbsp;·&nbsp; "
    "RTFM + TRN + Boundary Head with Frozen VideoMAE-B Backbone</font>",
    f"text;html=1;strokeColor=none;fillColor=none;align=center;"
    f"verticalAlign=middle;fontColor={DARK};fontSize=20;",
    0, 10, CW, 55))

# ═════════════════════════════════════════════════════════════════════════════
# ① FEATURE EXTRACTION  — top, full width
# ═════════════════════════════════════════════════════════════════════════════
sec(20, 72, CW-40, 508, S1_BG, AC1, "①",
    "FEATURE EXTRACTION PIPELINE — One-Time Preprocessing")

# Dataset
N_DS = node(55, 130, 260, 145, *BLUE,
    "UCF-Crime Dataset",
    "Sultani et al. CVPR 2018<br/>"
    "1,300 videos total<br/>"
    "6 violence categories<br/>"
    "Video-level weak labels")

# Frame Sampler
N_SAMP = node(365, 130, 235, 145, *INDIGO,
    "Frame Sampler",
    "16-frame segments<br/>"
    "Non-overlapping<br/>"
    "T segments per video<br/>"
    "Tail segment dropped")

add(hedge(next(ID), N_DS, N_SAMP, color=AC1))

# ── VideoMAE branch (top) ──
lbl(650, 118, 195, 22, "Our Contribution  ★", AC1, 10, bold=True)

N_VM = node(650, 140, 330, 148, *BLUE,
    "VideoMAE-B Backbone  ★",
    "Frozen ViT-B · Kinetics-400<br/>"
    "MCG-NJU/videomae-base checkpoint<br/>"
    "D = 768-dim output tokens<br/>"
    "BU SCC · NVIDIA A100")

N_VMF = node(1040, 140, 235, 135, *TEAL,
    "VideoMAE-B Features",
    "f_t &#8712; R^768<br/>"
    "1,300 .npz files<br/>"
    "494.6 s total")

add(hedge(next(ID), N_SAMP, N_VM,  color=AC1))
add(hedge(next(ID), N_VM,  N_VMF,  color=AC1))

# ── I3D branch (below) ──
lbl(650, 300, 155, 22, "Step-7 Baseline", "#ea580c", 10, bold=True)

N_I3D = node(650, 318, 290, 130, *ORANGE,
    "I3D Backbone",
    "Frozen · Kinetics-400<br/>"
    "Two-stream RGB + Flow<br/>"
    "D = 2048-dim features")

N_I3DF = node(1000, 318, 235, 120, *AMBER,
    "I3D Features",
    "f_t &#8712; R^2048<br/>"
    "Step-7 baseline cache<br/>"
    "1,300 .npz files")

add(hedge(next(ID), N_SAMP, N_I3D,  color="#ea580c"))
add(hedge(next(ID), N_I3D,  N_I3DF, color="#ea580c"))

# ── Feature Cache (cylinder) ──
N_CACHE = cyl(1330, 152, 230, 258, *TEAL,
    "Feature Cache",
    "1,300 .npz per backbone<br/>"
    "Cached once · reused<br/>"
    "Resume-safe checkpoints")

add(hedge(next(ID), N_VMF,  N_CACHE, color=AC1))
add(hedge(next(ID), N_I3DF, N_CACHE, color="#ea580c"))

# ── Comparison callout ──
N_COMP = node(1625, 148, 370, 235, *GRAY,
    "Feature Comparison",
    "I3D:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D=2048 · optical flow · 2017<br/>"
    "VideoMAE: D=768 &nbsp;· masked AE &nbsp;· 2022<br/>"
    "&nbsp;<br/>"
    "Same pipeline — only backbone swapped<br/>"
    "Dimension &#8722;63% → less overfitting risk<br/>"
    "AUC: 87.4% → 92.2% (&#43;4.8 pp)")

add(hedge(next(ID), N_CACHE, N_COMP, color=SUBTLE))

# ═════════════════════════════════════════════════════════════════════════════
# ② MODEL ARCHITECTURE  — bottom-left
# ═════════════════════════════════════════════════════════════════════════════
sec(20, 592, 1090, 1092, S2_BG, AC2, "②",
    "MODEL ARCHITECTURE — 1.9M Trainable Parameters · 86M Frozen")

# MIL Anomaly Scorer
N_MIL = node(50, 658, 315, 158, *VIOLET,
    "MIL Anomaly Scorer",
    "s_t = &#963;(W&#8323; Dropout(ReLU(W&#8322; ReLU(W&#8321;f_t))))<br/>"
    "Video score: s<sub>vid</sub> = max<sub>t</sub>(s_t)<br/>"
    "L_MIL ranking loss (anom/norm pairs)<br/>"
    "1.1 M params")

# Event Classifier
N_CLS = node(415, 658, 315, 158, *VIOLET,
    "Event Classifier",
    "7-class softmax head<br/>"
    "Top-k pseudo-label segments<br/>"
    "Confidence threshold &#964; = 0.30<br/>"
    "Inv-freq weighted CE · 0.3 M params")

add(hedge(next(ID), N_MIL, N_CLS, color=AC2))

# TRN
N_TRN = node(50, 865, 315, 155, *AMBER,
    "Temporal Refinement (TRN)",
    "Transformer encoder over {f_t + PE_t}<br/>"
    "Multi-head self-attention<br/>"
    "Refined scores s&#771;_t and h&#771;_t<br/>"
    "0.4 M params")

# Boundary Head
N_BND = node(415, 865, 315, 155, *AMBER,
    "Boundary Head",
    "b_t = &#963;(W<sub>b</sub>[h&#771;_t; h&#771;_{t+1}; |s&#771;_t&#8722;s&#771;_{t+1}|])<br/>"
    "Boundary consistency loss L_bnd<br/>"
    "Event start/end confidence<br/>"
    "0.1 M params")

add(hedge(next(ID), N_TRN, N_BND, color=AC2))
add(vedge(next(ID), N_CLS, N_BND, color=AC2))

# Full Training Objective
N_LOSS = node(50, 1070, 680, 140, *GRAY,
    "Full Training Objective",
    "L = L_MIL + &#955;&#8321;L_cls + &#955;&#8322;L_bnd + &#955;&#8323;L_smooth<br/>"
    "L_smooth = &#8721;(s&#771;_t &#8722; s&#771;_{t&#8722;1})&#178; &nbsp;&nbsp; &#955;&#8321;=0.5 · &#955;&#8322;=0.3 · &#955;&#8323;=0.1<br/>"
    "AdamW · lr=1e-4 · WD=1e-4 · Dropout=0.5 · Batch=32")

add(vedge(next(ID), N_BND, N_LOSS, color=SUBTLE))

# Training Recipe pills
pills = [
    (PURPLE, "Inv-Freq Weights",
     "w_c = N/(C&#183;n_c)<br/>Counters 10:1 imbalance"),
    (PURPLE, "Conf. Threshold",
     "&#964; = 0.30<br/>Filters noisy pseudo-pos."),
    (PURPLE, "Cosine Anneal LR",
     "&#951;: 1e-4 &#8594; 1e-6<br/>T_max = 40 epochs"),
    (PURPLE, "3-Seed Training",
     "Seeds: 42/123/456<br/>Report mean &#177; std"),
]
pw = 238; pg = 14
for i, (th, ti, su) in enumerate(pills):
    node(50 + i*(pw+pg), 1262, pw, 132, *th, ti, su, ft=11, fs=10)

lbl(50, 1440, 1040, 26,
    "PyTorch 2.1  ·  Transformers 5.5.4  ·  BU SCC A100  ·  1.9M trainable  ·  86M frozen",
    SUBTLE, 10)

# ═════════════════════════════════════════════════════════════════════════════
# ③ INFERENCE & EVALUATION  — bottom-right
# ═════════════════════════════════════════════════════════════════════════════
S3X = 1130
sec(S3X, 592, CW-S3X-20, 1092, S3_BG, AC3, "③",
    "INFERENCE &amp; EVALUATION — UCF-Crime Violence Subset")

# Inference chain
N_MOD  = node(S3X+30,  655, 255, 118, *GREEN,
    "Trained Model",
    "Best ckpt (seed 123)<br/>AUC = 93.4% peak epoch<br/>40-epoch training")

N_INF  = node(S3X+325, 655, 250, 118, *GREEN,
    "Inference Pipeline",
    "Forward pass per segment<br/>Outputs: s&#771;_t · p_t · b_t<br/>Temporal smoothing")

N_POST = node(S3X+620, 655, 250, 118, *TEAL,
    "Post-Processing",
    "Threshold candidate segs.<br/>Merge adjacent positives<br/>Boundary peak refinement")

N_OUT  = node(S3X+915, 655, 248, 118, *TEAL,
    "Event Tuples",
    "(t_start, t_end, class,<br/>confidence) per video<br/>Temporal localization")

add(hedge(next(ID), N_MOD,  N_INF,  color=AC3))
add(hedge(next(ID), N_INF,  N_POST, color=AC3))
add(hedge(next(ID), N_POST, N_OUT,  color=AC3))

# Evaluation metrics
lbl(S3X+30, 796, 1185, 28, "&#8213;&#8213;&#8213;  Evaluation Metrics  &#8213;&#8213;&#8213;",
    AC3, 13, bold=True)

N_M1 = node(S3X+30,  830, 370, 108, *GREEN, "Binary Detection",
    "AUC (ROC curve)<br/>AP (Precision-Recall)<br/>Video-level anomaly scores")

N_M2 = node(S3X+420, 830, 370, 108, *TEAL,  "Multi-Class F1",
    "Macro-F1 (6 classes, balanced)<br/>Weighted-F1 (freq-weighted)<br/>Per-class breakdown")

N_M3 = node(S3X+810, 830, 375, 108, *CYAN,  "Temporal Localization",
    "mAP @ tIoU = 0.3 / 0.5 / 0.7<br/>Segment-level event detection<br/>Boundary quality measure")

# Results table
lbl(S3X+30, 960, 1185, 26,
    "&#8213;&#8213;&#8213;  Results: I3D Baseline  vs  VideoMAE-B (3 Seeds, Mean &#177; Std)  &#8213;&#8213;&#8213;",
    AC3, 12, bold=True)

col_x = [S3X+30, S3X+240, S3X+558, S3X+898]
col_w = [200,    308,      330,     285]

# Header row
for cx, cw, txt, th in zip(col_x, col_w,
    ["Metric", "I3D Baseline", "VideoMAE-B (ours)", "Gain"],
    [GRAY, RED_T, BLUE, GREEN]):
    node(cx, 992, cw, 44, *th, txt, "", ft=12)

rows = [
    ("Binary AUC",      "87.4%",  "92.2% &#177; 1.3%",  "&#43;4.8 pp"),
    ("Binary AP",       "82.1%",  "88.7% &#177; 1.8%",  "&#43;8%"),
    ("Macro-F1",        "19.8%",  "25.5% &#177; 2.7%",  "&#43;28%"),
    ("mAP @ IoU 0.3",   "0.009",  "0.058 &#177; 0.012", "&#215;6.4"),
    ("mAP @ IoU 0.5",   "0.004",  "0.038 &#177; 0.008", "&#215;9.4"),
    ("mAP @ IoU 0.7",   "0.001",  "0.022 &#177; 0.010", "&#215;21.7"),
]
rh = 50; ry0 = 1042
for i, (met, i3d, vm, gn) in enumerate(rows):
    ry = ry0 + i*(rh+4)
    for cx, cw, val, th in zip(col_x, col_w,
        [met, i3d, vm, gn], [GRAY, RED_T, BLUE, GREEN]):
        node(cx, ry, cw, rh, *th, val, "", ft=11)

# Publication roadmap
lbl(S3X+30, 1364, 1185, 26, "&#8213;&#8213;&#8213;  Publication Roadmap  &#8213;&#8213;&#8213;",
    AC3, 12, bold=True)

pub = [
    (GREEN, "arXiv Preprint",
     "Submit now (April 2026)<br/>Establishes priority"),
    (TEAL, "ACM MM 2026<br/>Workshop",
     "Deadline: Jul 16, 2026<br/>6-page ACM SIGCONF"),
    (CYAN, "WACV 2027",
     "~Aug 2026 deadline<br/>Full paper + ablations"),
    (BLUE, "CVPR 2027",
     "~Nov 2026 deadline<br/>VideoMAE-L + XD-Violence"),
]
pw2 = 272; pg2 = 12
for i, (th, ti, su) in enumerate(pub):
    node(S3X+30 + i*(pw2+pg2), 1397, pw2, 118, *th, ti, su, ft=11, fs=10)

lbl(S3X+30, 1548, 1185, 26,
    "VideoMAE-B 768-dim · Kinetics-400 · Seeds 42/123/456 · 40 epochs · AdamW · BU SCC A100",
    SUBTLE, 10)

# ═════════════════════════════════════════════════════════════════════════════
# CROSS-SECTION CONNECTORS
# ═════════════════════════════════════════════════════════════════════════════
add(vedge(next(ID), N_CACHE, N_MIL,
    label="VideoMAE features", color=AC2, dashed=True))
add(vedge(next(ID), N_CACHE, N_MOD,
    label="test features",     color=AC3, dashed=True))
add(hedge(next(ID), N_BND, N_MOD,
    label="weights",           color=SUBTLE, dashed=True))

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER STATS BAR
# ═════════════════════════════════════════════════════════════════════════════
add(mk(next(ID), "",
    f"rounded=0;whiteSpace=wrap;html=1;"
    f"fillColor={S1_BG};strokeColor={AC1};strokeWidth=1;",
    20, 1690, 1140, 24))
add(mk(next(ID), "",
    f"rounded=0;whiteSpace=wrap;html=1;"
    f"fillColor={S3_BG};strokeColor={AC3};strokeWidth=1;",
    1175, 1690, 1205, 24))

lbl(20, 1690, 1140, 24,
    "AUC: 92.2%&#177;1.3%  ·  mAP@0.3: 0.058&#177;0.012 (&#215;6.4 over I3D)  ·  mAP@0.7: 0.022&#177;0.010 (&#215;21.7)",
    DARK, 10)
lbl(1175, 1690, 1205, 24,
    "Grading: VideoMAE-B 768-dim  ·  Blind 3-seed  ·  BU CS / IVC Project 2026",
    SUBTLE, 10)

# ═════════════════════════════════════════════════════════════════════════════
# BUILD XML
# ═════════════════════════════════════════════════════════════════════════════
graph = ET.Element("mxGraphModel",
    dx="1500", dy="900",
    grid="1", gridSize="10",
    guides="1", tooltips="1",
    connect="1", arrows="1", fold="1",
    page="0", pageScale="1",
    pageWidth=str(CW), pageHeight=str(CH),
    math="0", shadow="0",
    background=BG)

root_n = ET.SubElement(graph, "root")
ET.SubElement(root_n, "mxCell", id="0")
ET.SubElement(root_n, "mxCell", id="1", parent="0")
for c in cells:
    root_n.append(c)

out_path = "system_design_light.drawio"
ET.indent(graph, space="  ")
buf = io.StringIO()
ET.ElementTree(graph).write(buf, encoding="unicode", xml_declaration=False)

with open(out_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<mxfile host="app.diagrams.net" modified="2026-04-25" agent="Claude">\n')
    f.write('  <diagram name="IVC System Design" id="ivc-system-light">')
    f.write(buf.getvalue())
    f.write('</diagram>\n</mxfile>\n')

print(f"✓  {out_path}")
print(f"   Cells  : {len(cells)}")
print(f"   Canvas : {CW} x {CH} px  (white background)")
print()
print("Open at:  https://app.diagrams.net")
print("  Drag-drop  system_design_light.drawio  onto the browser window")
print("  Export PNG: File → Export as → PNG → 300 dpi")
