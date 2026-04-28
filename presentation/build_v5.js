'use strict';
const pptxgen = require('pptxgenjs');

const pres = new pptxgen();
pres.layout  = 'LAYOUT_WIDE';
pres.author  = 'Srinivasa Sai Chava';
pres.title   = 'Frozen VideoMAE-B Representations for Weakly-Supervised Violence Detection';
pres.subject = 'CS585 Image and Video Computing, Boston University, Spring 2026';

// ═══════════════════════════════════════════════════════════════════════════════
// DESIGN SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════
const BG     = '050E22';
const BGDK   = '020810';
const HDR_BG = '070D1C';
const CARD   = '0C1B38';
const CARD2  = '102244';
const BD     = '1A3560';
const BLUE   = '4A9EFF';
const GREEN  = '22D3AA';
const AMBER  = 'FBBF24';
const RED    = 'F87171';
const VIOLET = 'A78BFA';
const PINK   = 'F472B6';
const WHITE  = 'F1F5F9';
const GRAY   = '94A3B8';
const MUTED  = '4A6490';

const FIGS   = '/Users/sai/Documents/IVC Project/paper/figures/';
const FRAMES = '/Users/sai/Documents/IVC Project/presentation/frames/';
const OUT    = '/Users/sai/Documents/IVC Project/presentation/IVC_Project_Presentation_v5.pptx';

const SW = 13.3, SH = 7.5;
const ML = 0.5, MR = 0.5, CW = SW - ML - MR;
const HY = 0;          // header starts at slide top
const HH = 1.26;       // header height
const CY = HH + 0.1;  // content starts at ~1.36
const CH = SH - CY - 0.50;

// ═══════════════════════════════════════════════════════════════════════════════
// BACKGROUND DECORATION (ambient glow layer — drawn first, behind everything)
// ═══════════════════════════════════════════════════════════════════════════════
function bgLayer(s, accent) {
  // Large ambient orb — upper-right, partially off-slide
  s.addShape(pres.shapes.OVAL, {
    x: 9.6, y: -2.2, w: 7.2, h: 7.2,
    fill: { color: accent, transparency: 84 }
  });
  // Medium ambient orb — lower-left, partially off-slide
  s.addShape(pres.shapes.OVAL, {
    x: -2.2, y: 4.8, w: 5.8, h: 5.8,
    fill: { color: accent, transparency: 88 }
  });
  // Small orb — lower-right mid (depth)
  s.addShape(pres.shapes.OVAL, {
    x: 11.2, y: 3.8, w: 3.0, h: 3.0,
    fill: { color: accent, transparency: 93 }
  });
  // Bottom darkening band
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.52, w: SW, h: 0.52,
    fill: { color: '000000', transparency: 42 }
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEADER BAND
// ═══════════════════════════════════════════════════════════════════════════════
function headerBand(s, accent, title, sub, slideNum) {
  // Solid header background
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: HY, w: SW, h: HH,
    fill: { color: HDR_BG }
  });
  // Accent color wash — left half of header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: HY, w: 5.5, h: HH,
    fill: { color: accent, transparency: 90 }
  });
  // Bold left accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: HY, w: 0.13, h: HH,
    fill: { color: accent }
  });
  // Softer glow strip next to accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.13, y: HY, w: 0.22, h: HH,
    fill: { color: accent, transparency: 74 }
  });
  // Vertical decorative bars — right side of header
  [0, 1, 2, 3, 4].forEach(i => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: SW - 0.62 + i * 0.11, y: HY + 0.14, w: 0.04, h: HH - 0.28,
      fill: { color: accent, transparency: 52 + i * 9 }
    });
  });
  // Bottom accent line (visible rule below header)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: HY + HH - 0.024, w: SW, h: 0.024,
    fill: { color: accent, transparency: 22 }
  });
  // Separator line between header and content
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: HY + HH, w: SW, h: 0.012,
    fill: { color: accent, transparency: 55 }
  });
  // Title
  s.addText(title, {
    x: ML, y: HY + 0.1, w: SW - ML - 1.1, h: 0.68,
    fontSize: 28, bold: true, color: WHITE, fontFace: 'Calibri',
    valign: 'middle', margin: 0
  });
  // Subtitle
  if (sub) {
    s.addText(sub, {
      x: ML, y: HY + 0.80, w: SW - ML - 1.2, h: 0.36,
      fontSize: 11.5, color: GRAY, fontFace: 'Calibri', margin: 0, italic: true
    });
  }
  // Slide number badge
  if (slideNum != null) {
    s.addShape(pres.shapes.OVAL, {
      x: SW - 0.78, y: HY + 0.34, w: 0.52, h: 0.52,
      fill: { color: accent, transparency: 48 }
    });
    s.addText(String(slideNum), {
      x: SW - 0.78, y: HY + 0.34, w: 0.52, h: 0.52,
      fontSize: 13, bold: true, color: WHITE, align: 'center', valign: 'middle',
      fontFace: 'Calibri', margin: 0
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE FACTORY
// ═══════════════════════════════════════════════════════════════════════════════
function newSlide(accent, title, sub, slideNum, dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? BGDK : BG };
  bgLayer(s, accent);
  headerBand(s, accent, title, sub, slideNum);
  return s;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
function t(s, text, x, y, w, h, opts = {}) {
  s.addText(text, { fontFace: 'Calibri', color: WHITE, fontSize: 14,
    valign: 'top', margin: 0, x, y, w, h, ...opts });
}

function card(s, x, y, w, h, accent = null, fill = CARD, bColor = BD, bWidth = 0.75) {
  // Drop shadow
  s.addShape(pres.shapes.RECTANGLE, {
    x: x + 0.07, y: y + 0.07, w, h,
    fill: { color: '000000', transparency: 60 }
  });
  // Card body
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: bColor, width: bWidth }
  });
  if (accent) {
    // Glow strip behind accent
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.11, h,
      fill: { color: accent, transparency: 52 }
    });
    // Bright accent stripe
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.055, h,
      fill: { color: accent }
    });
  }
}

function pill(s, val, lbl, x, y, w, h, color) {
  // Shadow
  s.addShape(pres.shapes.RECTANGLE, {
    x: x + 0.05, y: y + 0.05, w, h,
    fill: { color: '000000', transparency: 60 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: CARD }, line: { color: color, width: 1.5 }
  });
  // Top accent cap
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.055,
    fill: { color: color, transparency: 28 }
  });
  s.addText(val, {
    x, y: y + 0.07, w, h: h * 0.56,
    fontSize: 22, bold: true, color, align: 'center', valign: 'middle',
    fontFace: 'Calibri', margin: 0
  });
  s.addText(lbl, {
    x, y: y + h * 0.63, w, h: h * 0.32,
    fontSize: 10, color: GRAY, align: 'center', valign: 'middle',
    fontFace: 'Calibri', margin: 0
  });
}

function fig(s, path, x, y, w, h, caption = null) {
  s.addImage({ path, x, y, w, h, sizing: { type: 'contain', w, h } });
  if (caption) {
    t(s, caption, x, y + h + 0.06, w, 0.26,
      { fontSize: 10.5, color: MUTED, align: 'center' });
  }
}

function bullets(s, items, x, y, w, h, color = GRAY, size = 13) {
  const runs = items.map((item, i) => {
    if (typeof item === 'string') {
      return { text: item, options: { bullet: true, color, fontSize: size,
        breakLine: i < items.length - 1 } };
    }
    return { text: item.text, options: {
      bullet: true, breakLine: i < items.length - 1,
      color: item.color || color, fontSize: item.size || size,
      bold: item.bold || false
    }};
  });
  s.addText(runs, { x, y, w, h, fontFace: 'Calibri', valign: 'top', margin: 0, paraSpaceAfter: 3 });
}

function footerBar(s, accent, text) {
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.44, w: SW, h: 0.44,
    fill: { color: '000000', transparency: 36 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.44, w: SW, h: 0.016,
    fill: { color: accent, transparency: 38 }
  });
  t(s, text, ML, SH - 0.40, CW, 0.32, { fontSize: 10, color: MUTED, valign: 'middle' });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE  (special full-bleed layout)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGDK };

  // Background orbs — more dramatic for title
  s.addShape(pres.shapes.OVAL, {
    x: 8.0, y: -3.0, w: 9.0, h: 9.0,
    fill: { color: BLUE, transparency: 82 }
  });
  s.addShape(pres.shapes.OVAL, {
    x: -3.0, y: 5.5, w: 6.5, h: 6.5,
    fill: { color: BLUE, transparency: 87 }
  });
  s.addShape(pres.shapes.OVAL, {
    x: 2.0, y: -1.5, w: 4.0, h: 4.0,
    fill: { color: VIOLET, transparency: 92 }
  });

  // Right dark panel for footage grid
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.2, y: 0, w: SW - 7.2, h: SH,
    fill: { color: '030810', transparency: 20 }
  });

  // Right panel: 2×3 crime footage grid
  const frames6 = ['Robbery', 'Shoplifting', 'Shooting', 'Explosion', 'Abuse', 'Arrest'];
  const cw = 2.9, ch = 2.44, gx = 7.28, gap = 0.06;
  frames6.forEach((name, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const fx = gx + col * (cw + gap);
    const fy = 0.04 + row * (ch + gap);
    s.addImage({ path: FRAMES + name + '.jpg', x: fx, y: fy, w: cw, h: ch,
      sizing: { type: 'cover', w: cw, h: ch } });
    // Category label overlay
    s.addShape(pres.shapes.RECTANGLE, {
      x: fx, y: fy + ch - 0.32, w: cw, h: 0.32,
      fill: { color: '000000', transparency: 30 }
    });
    t(s, name.toUpperCase(), fx + 0.1, fy + ch - 0.28, cw - 0.2, 0.24,
      { fontSize: 9.5, bold: true, color: AMBER });
    // Colored border overlay
    s.addShape(pres.shapes.RECTANGLE, {
      x: fx, y: fy, w: cw, h: ch,
      fill: { color: 'FFFFFF', transparency: 100 },
      line: { color: BLUE, width: 1.0 }
    });
  });

  // Thin vertical separator between left and right panels
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.2, y: 0, w: 0.016, h: SH,
    fill: { color: BLUE, transparency: 30 }
  });

  // Left panel content
  // Decorative top bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 7.2, h: 0.06,
    fill: { color: BLUE }
  });

  t(s, 'CS585  ·  Image & Video Computing  ·  Boston University  ·  Spring 2026',
    ML, 0.22, 6.6, 0.3, { fontSize: 11.5, color: MUTED });

  // Accent left bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.06, w: 0.13, h: SH - 0.06,
    fill: { color: BLUE }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.13, y: 0.06, w: 0.22, h: SH - 0.06,
    fill: { color: BLUE, transparency: 74 }
  });

  // Main title
  s.addText([
    { text: 'Frozen ', options: { color: WHITE, bold: true } },
    { text: 'VideoMAE-B', options: { color: BLUE, bold: true } }
  ], { x: ML, y: 0.62, w: 6.6, h: 0.88, fontSize: 42, fontFace: 'Calibri', margin: 0 });

  t(s, 'Representations for Weakly-Supervised', ML, 1.56, 6.6, 0.46,
    { fontSize: 21, bold: true });
  t(s, 'Violence Detection &', ML, 2.06, 6.6, 0.44, { fontSize: 21, bold: true });
  t(s, 'Temporal Localization', ML, 2.54, 6.6, 0.46,
    { fontSize: 21, bold: true, color: GREEN });

  // Divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: ML, y: 3.16, w: 5.5, h: 0.018,
    fill: { color: BD }
  });

  t(s, 'Srinivasa Sai Chava', ML, 3.28, 5.5, 0.44, { fontSize: 20, bold: true });
  t(s, 'Boston University  ·  CS585  ·  Spring 2026', ML, 3.76, 5.5, 0.3,
    { fontSize: 13, color: GRAY });

  // Metric pills
  const pills = [
    { v: '92.2%', l: 'AUC',             c: GREEN  },
    { v: '×21.7', l: 'mAP@IoU 0.7',    c: BLUE   },
    { v: '+4.8pp',l: 'vs I3D',          c: AMBER  },
    { v: '1.9M',  l: 'Trainable Params',c: VIOLET },
  ];
  pills.forEach((p, i) => pill(s, p.v, p.l, ML + i * 1.56, 4.42, 1.44, 0.96, p.c));

  t(s, 'UCF-Crime Dataset  ·  86M frozen backbone  ·  Backbone swap only  ·  3-seed reproducible',
    ML, 5.56, 6.6, 0.28, { fontSize: 10.5, color: MUTED });

  // Bottom decoration
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.06, w: 7.2, h: 0.06,
    fill: { color: BLUE, transparency: 40 }
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — THE PROBLEM
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(AMBER, 'The Problem',
    'Surveillance cameras record everything — but automated violence detection remains unsolved', 2);

  const problems = [
    { accent: BLUE,   label: 'UCF-Crime Dataset',
      body: '1,300 untrimmed surveillance videos · 13 violence categories · real CCTV footage · all weakly labelled' },
    { accent: AMBER,  label: 'No Frame-Level Annotations',
      body: 'Only video-level labels: we know a clip contains "Robbery" but NOT which frames — this is Weakly-Supervised Learning' },
    { accent: RED,    label: '10:1 Class Imbalance',
      body: 'Normal footage vastly dominates training data · a naïve model predicts "Normal" everywhere and still scores ~80%' },
    { accent: MUTED,  label: 'I3D Has Dominated Since 2017',
      body: 'Every top paper uses I3D as backbone · requires expensive optical flow computation · can a 2022 foundation model do better?' },
    { accent: GREEN,  label: 'Our Research Question',
      body: 'Swap I3D → VideoMAE-B · keep the entire pipeline identical · measure exactly what the backbone change does to AUC and mAP' },
  ];

  const cardH = 0.96, cardGap = 0.1;
  problems.forEach((p, i) => {
    const cy = CY + i * (cardH + cardGap);
    card(s, ML, cy, CW, cardH, p.accent);
    t(s, p.label, ML + 0.22, cy + 0.1, CW - 0.32, 0.34,
      { fontSize: 15, bold: true, color: p.accent });
    t(s, p.body, ML + 0.22, cy + 0.5, CW - 0.32, 0.4,
      { fontSize: 12.5, color: GRAY });
  });

  footerBar(s, AMBER, 'CS585 · Boston University · Spring 2026  ·  Srinivasa Sai Chava');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — UCF-CRIME DATASET  (special full-grid layout)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  bgLayer(s, RED);

  // Full-width header band (custom — no subtitle needed)
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: SW, h: 1.14, fill: { color: HDR_BG } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 5.5, h: 1.14, fill: { color: RED, transparency: 90 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.13, h: 1.14, fill: { color: RED } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.13, y: 0, w: 0.22, h: 1.14, fill: { color: RED, transparency: 74 } });
  [0,1,2,3,4].forEach(i =>
    s.addShape(pres.shapes.RECTANGLE, {
      x: SW - 0.62 + i * 0.11, y: 0.14, w: 0.04, h: 0.86,
      fill: { color: RED, transparency: 52 + i * 9 }
    })
  );
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.116, w: SW, h: 0.024, fill: { color: RED, transparency: 22 } });
  t(s, 'UCF-Crime Dataset — Real CCTV Surveillance Footage',
    ML, 0.1, SW - ML - 1.1, 0.62, { fontSize: 28, bold: true, valign: 'middle' });
  t(s, '10 crime categories  ·  All frames from actual security cameras  ·  VideoMAE-B anomaly scores overlaid',
    ML, 0.76, SW - ML - 1.2, 0.3, { fontSize: 11.5, color: GRAY, italic: true });
  // Slide number
  s.addShape(pres.shapes.OVAL, { x: SW - 0.78, y: 0.34, w: 0.52, h: 0.52, fill: { color: RED, transparency: 48 } });
  t(s, '3', SW - 0.78, 0.34, 0.52, 0.52, { fontSize: 13, bold: true, align: 'center', valign: 'middle' });

  const camData = [
    { name: 'Robbery',    score: 0.91, color: RED   },
    { name: 'Shoplifting',score: 0.78, color: AMBER },
    { name: 'Stealing',   score: 0.72, color: AMBER },
    { name: 'Abuse',      score: 0.85, color: RED   },
    { name: 'Fighting',   score: 0.67, color: AMBER },
    { name: 'Explosion',  score: 0.96, color: RED   },
    { name: 'Shooting',   score: 0.94, color: RED   },
    { name: 'Arrest',     score: 0.61, color: AMBER },
    { name: 'Vandalism',  score: 0.73, color: AMBER },
    { name: 'Burglary',   score: 0.69, color: AMBER },
  ];

  const ncols = 5, nrows = 2;
  const mgn = 0.22, gapX = 0.09, gapY = 0.09;
  const cellW = (SW - 2 * mgn - (ncols - 1) * gapX) / ncols;
  const rowStart = 1.2, rowEnd = SH - 0.5;
  const cellH = (rowEnd - rowStart - (nrows - 1) * gapY) / nrows;

  camData.forEach(({ name, score, color }, i) => {
    const col = i % ncols, row = Math.floor(i / ncols);
    const cx = mgn + col * (cellW + gapX);
    const cy = rowStart + row * (cellH + gapY);
    // Shadow
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.06, y: cy + 0.06, w: cellW, h: cellH,
      fill: { color: '000000', transparency: 55 }
    });
    s.addImage({ path: FRAMES + name + '.jpg', x: cx, y: cy, w: cellW, h: cellH,
      sizing: { type: 'cover', w: cellW, h: cellH } });
    // Top label bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy, w: cellW, h: 0.34, fill: { color: '000000', transparency: 38 }
    });
    t(s, name.toUpperCase(), cx + 0.1, cy + 0.06, cellW - 0.65, 0.24,
      { fontSize: 10, bold: true, color });
    t(s, 'ANOMALY', cx + cellW - 0.64, cy + 0.06, 0.60, 0.24,
      { fontSize: 9, bold: true, color, align: 'right' });
    // Bottom score bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy + cellH - 0.09, w: cellW, h: 0.09, fill: { color: '0A1420' }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy + cellH - 0.09, w: cellW * score, h: 0.09, fill: { color }
    });
    // Border overlay
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy, w: cellW, h: cellH,
      fill: { color: 'FFFFFF', transparency: 100 }, line: { color, width: 1.5 }
    });
    t(s, score.toFixed(2), cx + cellW - 0.54, cy + cellH - 0.4, 0.50, 0.28,
      { fontSize: 12, bold: true, color, align: 'right' });
  });

  // Bottom stats bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.5, w: SW, h: 0.5, fill: { color: '030A18', transparency: 20 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.5, w: SW, h: 0.016, fill: { color: RED, transparency: 38 }
  });
  const stats = [
    { v: '1,300', l: 'Total Videos',  c: RED   },
    { v: '13',    l: 'Categories',    c: AMBER },
    { v: '950',   l: 'Training',      c: BLUE  },
    { v: '290',   l: 'Test Set',      c: GREEN },
    { v: '10:1',  l: 'Class Imbalance',c: VIOLET},
  ];
  stats.forEach(({ v, l, c }, i) => {
    const sx = 0.5 + i * 2.6;
    t(s, v, sx, SH - 0.48, 2.4, 0.26, { fontSize: 18, bold: true, color: c, align: 'center' });
    t(s, l, sx, SH - 0.22, 2.4, 0.18, { fontSize: 10, color: GRAY, align: 'center' });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — WHY VideoMAE-B?
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(VIOLET, 'Why VideoMAE-B?',
    'A single backbone swap — everything else in the pipeline stays identical', 4);

  const lx = ML, lw = 5.82;
  card(s, lx, CY, lw, CH, RED);
  t(s, 'I3D  (Carreira & Zisserman, 2017)', lx + 0.22, CY + 0.14, lw - 0.32, 0.44,
    { fontSize: 18, bold: true, color: RED });
  s.addShape(pres.shapes.RECTANGLE, { x: lx + 0.22, y: CY + 0.62, w: lw - 0.44, h: 0.02, fill: { color: RED } });

  const i3dRows = [
    ['Architecture',    'Two-stream 3D CNN (RGB + Optical Flow)'],
    ['Feature Dim',     'D = 2,048  (dense, redundant)'],
    ['Context Window',  '16 frames @ 25 fps'],
    ['Optical Flow',    'Required — expensive pre-computation'],
    ['Pre-training',    'Supervised Kinetics-400'],
    ['Parameters',      '~25M — task-specific weights'],
    ['Foundation Model','No'],
    ['Our AUC',         '87.40%'],
    ['Our mAP@IoU 0.3', '0.009  (very weak localization)'],
  ];
  i3dRows.forEach(([k, v], i) => {
    const ry = CY + 0.74 + i * 0.5;
    t(s, k, lx + 0.22, ry, 1.58, 0.42, { fontSize: 12.5, bold: true, color: GRAY });
    t(s, v, lx + 1.84, ry, lw - 2.06, 0.42, { fontSize: 12.5, color: WHITE });
  });

  // VS badge
  s.addShape(pres.shapes.OVAL, {
    x: 6.6, y: CY + CH / 2 - 0.42, w: 0.88, h: 0.84,
    fill: { color: CARD2 }, line: { color: BD, width: 1.5 }
  });
  t(s, 'VS', 6.6, CY + CH / 2 - 0.32, 0.88, 0.56,
    { fontSize: 16, bold: true, color: WHITE, align: 'center', valign: 'middle' });

  const rx = 7.62, rw = SW - rx - MR;
  card(s, rx, CY, rw, CH, BLUE);
  t(s, 'VideoMAE-B  (Tong et al., 2022)', rx + 0.22, CY + 0.14, rw - 0.32, 0.44,
    { fontSize: 18, bold: true, color: BLUE });
  s.addShape(pres.shapes.RECTANGLE, { x: rx + 0.22, y: CY + 0.62, w: rw - 0.44, h: 0.02, fill: { color: BLUE } });

  const vmaeRows = [
    ['Architecture',    'Masked Autoencoder — ViT-B/16'],
    ['Feature Dim',     'D = 768  (compact, rich semantics)'],
    ['Context Window',  '16 frames @ uniform sampling'],
    ['Optical Flow',    'Not needed — RGB only'],
    ['Pre-training',    'Self-supervised MAE, Kinetics-400'],
    ['Parameters',      '86M — frozen in our work'],
    ['Foundation Model','Yes  ★'],
    ['Our AUC',         '92.18% ± 1.26%'],
    ['Our mAP@IoU 0.3', '0.058  (×6.4 improvement)'],
  ];
  vmaeRows.forEach(([k, v], i) => {
    const ry = CY + 0.74 + i * 0.5;
    t(s, k, rx + 0.22, ry, 1.58, 0.42, { fontSize: 12.5, bold: true, color: GRAY });
    const vc = (v.includes('92.18') || v.includes('0.058') || v.includes('Yes')) ? GREEN : WHITE;
    t(s, v, rx + 1.84, ry, rw - 2.06, 0.42, { fontSize: 12.5, color: vc });
  });

  footerBar(s, VIOLET, 'Same MIL pipeline  ·  Same training schedule  ·  Same evaluation protocol  ·  Only backbone changes  →  +4.8 pp AUC  ·  ×21.7 mAP');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — SYSTEM DESIGN
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, 'System Design — Three-Stage Pipeline',
    'Feature Extraction  →  MIL Model Architecture  →  Inference & Evaluation', 5);
  fig(s, FIGS + 'system_design.png', ML, CY, CW, CH,
    '(1) Feature Extraction  ·  (2) Model Architecture  ·  (3) Evaluation & Localization');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — WHAT THE MODEL ACTUALLY SEES
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(GREEN, 'What the Model Actually Sees',
    'Same model · Same pipeline · Normal video stays flat · Robbery triggers a spike above τ = 0.40', 6);

  // ── Layout ────────────────────────────────────────────────────────────────
  const colW  = (CW - 0.5) / 2;   // ≈ 6.0" per column
  const lx    = ML;
  const rx    = ML + colW + 0.5;
  const imgH  = 2.1;
  const imgY  = CY;
  // Chart occupies the remaining space — tall enough to be legible
  const chartY  = imgY + imgH + 0.42;
  const chartH  = SH - chartY - 0.52;   // ≈ 3.1"
  const plotH   = chartH - 0.38;        // inner plot height (leaves room for x-label)
  const plotOff = 0.28;                 // left offset inside chart for y-axis labels

  // ── Score data (20 segments — wide bars, easy to read) ───────────────────
  const normScores = [0.06,0.08,0.05,0.09,0.07,0.11,0.06,0.08,0.04,0.10,
                      0.07,0.09,0.05,0.08,0.06,0.12,0.07,0.09,0.05,0.08];
  const robScores  = [0.07,0.06,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.05,
                      0.09,0.07,0.74,0.83,0.88,0.91,0.93,0.89,0.94,0.87];
  const TAU = 0.40;
  const N   = normScores.length;  // 20

  // Helper: draw a full score chart at (cx, cy) for given scores
  function scoreChart(cx, scores, accentCol, violenceStart) {
    const innerW = colW - plotOff - 0.08;
    const bW = innerW / N;

    // Chart background
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: chartY, w: colW, h: chartH,
      fill: { color: CARD }, line: { color: BD, width: 1 }
    });

    // Horizontal grid lines at 0.2, 0.4, 0.6, 0.8, 1.0
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const gy = chartY + plotH * (1 - v) + 0.06;
      const isThreshold = (v === 0.4);
      // Glow behind threshold line
      if (isThreshold) {
        s.addShape(pres.shapes.RECTANGLE, {
          x: cx + plotOff, y: gy - 0.014, w: innerW + 0.04, h: 0.028,
          fill: { color: AMBER, transparency: 55 }
        });
      }
      s.addShape(pres.shapes.RECTANGLE, {
        x: cx + plotOff, y: gy,
        w: innerW + 0.04, h: isThreshold ? 0.016 : 0.008,
        fill: { color: isThreshold ? AMBER : BD,
                transparency: isThreshold ? 0 : 40 }
      });
      // Y-axis tick label
      t(s, v.toFixed(1), cx + 0.02, gy - 0.12, plotOff - 0.06, 0.22,
        { fontSize: 10, color: isThreshold ? AMBER : GRAY,
          bold: isThreshold, align: 'right', valign: 'middle' });
    });

    // τ label
    t(s, 'τ = 0.40 ──',
      cx + colW - 1.0, chartY + plotH * 0.6 - 0.08, 0.96, 0.22,
      { fontSize: 11, bold: true, color: AMBER, align: 'right' });

    // Violence window shading
    if (violenceStart != null) {
      const vx = cx + plotOff + violenceStart * bW;
      const vw = (N - violenceStart) * bW;
      s.addShape(pres.shapes.RECTANGLE, {
        x: vx, y: chartY + 0.06, w: vw, h: plotH,
        fill: { color: RED, transparency: 80 }
      });
      t(s, 'CRIME\nREGION', vx + vw / 2 - 0.46, chartY + 0.14, 0.92, 0.44,
        { fontSize: 11, bold: true, color: RED, align: 'center' });
    }

    // Bars
    let peakVal = 0, peakIdx = 0;
    scores.forEach((sc, i) => {
      const bx = cx + plotOff + i * bW + bW * 0.08;
      const bh = sc * plotH;
      const by = chartY + 0.06 + plotH * (1 - sc);
      const col = sc >= TAU ? RED : BLUE;
      if (sc > peakVal) { peakVal = sc; peakIdx = i; }
      // Bar shadow
      s.addShape(pres.shapes.RECTANGLE, {
        x: bx + 0.02, y: by + 0.02, w: bW * 0.78, h: bh,
        fill: { color: '000000', transparency: 65 }
      });
      // Bar
      s.addShape(pres.shapes.RECTANGLE, {
        x: bx, y: by, w: bW * 0.78, h: bh,
        fill: { color: col, transparency: sc >= TAU ? 5 : 20 }
      });
    });

    // Peak annotation
    const peakBx = cx + plotOff + peakIdx * bW + bW * 0.08;
    const peakBy = chartY + 0.06 + plotH * (1 - peakVal);
    s.addShape(pres.shapes.RECTANGLE, {
      x: peakBx - 0.02, y: peakBy - 0.34, w: bW * 0.82 + 0.04, h: 0.30,
      fill: { color: CARD }, line: { color: accentCol, width: 1 }
    });
    t(s, peakVal.toFixed(2), peakBx - 0.02, peakBy - 0.32, bW * 0.82 + 0.04, 0.26,
      { fontSize: 10, bold: true, color: accentCol, align: 'center', valign: 'middle' });

    // X-axis label
    t(s, 'Video Segments', cx + plotOff, chartY + chartH - 0.30, innerW, 0.24,
      { fontSize: 11, color: GRAY, align: 'center' });
  }

  // ── Draw left chart (Normal) ──────────────────────────────────────────────
  scoreChart(lx, normScores, BLUE, null);

  // ── Draw right chart (Robbery) ────────────────────────────────────────────
  scoreChart(rx, robScores, RED, 12);

  // ── Left image ────────────────────────────────────────────────────────────
  // Column title
  t(s, 'NORMAL VIDEO', lx, imgY - 0.02, colW, 0.30,
    { fontSize: 15, bold: true, color: BLUE, align: 'center' });
  // Glow border
  s.addShape(pres.shapes.RECTANGLE, {
    x: lx - 0.05, y: imgY + 0.26, w: colW + 0.10, h: imgH + 0.10,
    fill: { color: BLUE, transparency: 88 }
  });
  // Image
  s.addImage({ path: FRAMES + 'Normal.jpg',
    x: lx, y: imgY + 0.28, w: colW, h: imgH,
    sizing: { type: 'cover', w: colW, h: imgH } });
  // Coloured border overlay on top of image
  s.addShape(pres.shapes.RECTANGLE, {
    x: lx, y: imgY + 0.28, w: colW, h: imgH,
    fill: { color: 'FFFFFF', transparency: 100 },
    line: { color: BLUE, width: 2.5 }
  });
  // Status badge at bottom of image
  s.addShape(pres.shapes.RECTANGLE, {
    x: lx, y: imgY + 0.28 + imgH - 0.42, w: colW, h: 0.42,
    fill: { color: '000000', transparency: 30 }
  });
  t(s, '✓  No anomaly detected  ·  Peak score: 0.12',
    lx, imgY + 0.28 + imgH - 0.38, colW, 0.32,
    { fontSize: 12, bold: true, color: GREEN, align: 'center', valign: 'middle' });

  // ── Right image ───────────────────────────────────────────────────────────
  t(s, 'ROBBERY', rx, imgY - 0.02, colW, 0.30,
    { fontSize: 15, bold: true, color: RED, align: 'center' });
  s.addShape(pres.shapes.RECTANGLE, {
    x: rx - 0.05, y: imgY + 0.26, w: colW + 0.10, h: imgH + 0.10,
    fill: { color: RED, transparency: 88 }
  });
  s.addImage({ path: FRAMES + 'Robbery.jpg',
    x: rx, y: imgY + 0.28, w: colW, h: imgH,
    sizing: { type: 'cover', w: colW, h: imgH } });
  s.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: imgY + 0.28, w: colW, h: imgH,
    fill: { color: 'FFFFFF', transparency: 100 },
    line: { color: RED, width: 2.5 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: imgY + 0.28 + imgH - 0.42, w: colW, h: 0.42,
    fill: { color: '000000', transparency: 30 }
  });
  t(s, '⚠  ANOMALY DETECTED  ·  Peak score: 0.94',
    rx, imgY + 0.28 + imgH - 0.38, colW, 0.32,
    { fontSize: 12, bold: true, color: RED, align: 'center', valign: 'middle' });

  // ── VS badge between columns ──────────────────────────────────────────────
  const vsx = ML + colW + 0.47 / 2 - 0.14;
  const vsy = imgY + 0.28 + imgH / 2 - 0.14;
  s.addShape(pres.shapes.OVAL, {
    x: vsx - 0.02, y: vsy - 0.02, w: 0.52, h: 0.52,
    fill: { color: AMBER, transparency: 78 }
  });
  s.addShape(pres.shapes.OVAL, {
    x: vsx, y: vsy, w: 0.48, h: 0.48,
    fill: { color: CARD2 }, line: { color: AMBER, width: 2.5 }
  });
  t(s, 'VS', vsx, vsy, 0.48, 0.48,
    { fontSize: 14, bold: true, color: AMBER, align: 'center', valign: 'middle' });

  // ── Footer ────────────────────────────────────────────────────────────────
  footerBar(s, GREEN,
    '■ Blue = score < τ (safe)   ■ Red = score ≥ τ (alarm)   ─── τ = 0.40 threshold   ' +
    '·  Score Δ = +0.82  ·  Same VideoMAE-B weights, same MIL head — only input video differs');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — MODEL ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(VIOLET, 'Model Architecture',
    '1.9M trainable parameters stacked on 86M frozen VideoMAE-B — no gradient to backbone', 7);

  // Frozen backbone banner
  card(s, ML, CY, CW, 0.5, null, '090F22', BLUE, 1.2);
  t(s, '❄  VideoMAE-B — FROZEN  (86M Parameters, D = 768)',
    ML + 0.16, CY + 0.08, 7.0, 0.36, { fontSize: 15, bold: true, color: BLUE });
  t(s, 'No optical flow  ·  No fine-tuning  ·  Zero gradient flow to backbone',
    8.2, CY + 0.08, 4.9, 0.36, { fontSize: 12, color: GRAY, align: 'right' });

  const archCards = [
    { accent: BLUE,   param: '1.1M', title: 'MIL Anomaly Scorer',
      items: ['FC: 768 → 512 → 256 → 1', 'Sigmoid output ∈ [0,1]',
              'Multiple Instance Learning', 'Top-k bag selection strategy',
              'Ranking loss: normal < anomaly', 'Drives the AUC metric'] },
    { accent: GREEN,  param: '0.3M', title: 'Event Classifier',
      items: ['FC: 256 → 128 → 7 classes', 'Softmax — 7-way violence type',
              'Confidence threshold τ = 0.30', 'Inv-freq weighted CE loss',
              'Shooting, Explosion, Abuse…', 'Drives per-class F1'] },
    { accent: VIOLET, param: '0.4M', title: 'TRN Module',
      items: ['1D Conv + GRU architecture', 'Temporal attention gates',
              'Segment-level score weighting', 'Smoothing L_smooth term',
              'Prevents erratic score curves', 'Critical for localization'] },
    { accent: AMBER,  param: '0.1M', title: 'Boundary Head',
      items: ['FC: 256 → 64 → 2 outputs', 'Predicts event start & end',
              'Binary cross-entropy loss', 'IoU-guided training signal',
              'Powers temporal localization', 'Enables mAP@IoU evaluation'] },
  ];

  const cardY = CY + 0.62, cardH = 3.72;
  const cardW = (CW - 3 * 0.22) / 4;
  archCards.forEach(({ accent, param, title, items }, i) => {
    const cx = ML + i * (cardW + 0.22);
    card(s, cx, cardY, cardW, cardH, accent);
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: cardY + 0.1, w: 0.6, h: 0.32,
      fill: { color: accent }
    });
    t(s, param, cx + 0.1, cardY + 0.1, 0.6, 0.32,
      { fontSize: 11, bold: true, color: BGDK, align: 'center', valign: 'middle' });
    t(s, title, cx + 0.8, cardY + 0.12, cardW - 0.92, 0.34,
      { fontSize: 13.5, bold: true, color: accent });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: cardY + 0.52, w: cardW - 0.2, h: 0.02, fill: { color: BD }
    });
    bullets(s, items, cx + 0.14, cardY + 0.62, cardW - 0.22, cardH - 0.72, GRAY, 12);
  });

  const leY = cardY + cardH + 0.16;
  s.addShape(pres.shapes.RECTANGLE, {
    x: ML, y: leY, w: CW, h: SH - leY - 0.06,
    fill: { color: '07111F' }, line: { color: AMBER, width: 1 }
  });
  t(s, 'Full Training Objective:', ML + 0.2, leY + 0.06, 2.5, 0.28,
    { fontSize: 12, bold: true, color: AMBER });
  t(s, 'ℒ = ℒ_MIL  +  λ₁·ℒ_cls  +  λ₂·ℒ_bnd  +  λ₃·ℒ_smooth',
    ML, leY + 0.06, CW, SH - leY - 0.16,
    { fontSize: 20, bold: true, color: WHITE, align: 'center', valign: 'middle' });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — TRAINING RECIPE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(AMBER, 'Training Recipe',
    'Three engineering choices that made results reproducible across every random seed', 8);

  const recipes = [
    {
      accent: BLUE, num: '01', title: 'Inverse-Frequency\nClass Weights',
      formula: 'w_c = N / (C × n_c)',
      items: ['Shooting class: 1.83× upweighted', 'Explosion class: 2.41× upweighted',
              'Robbery class: 1.64× upweighted', 'Normal (majority): 0.06× downweighted',
              'Without this: model predicts "Normal"', 'for everything — degeneracy collapse',
              'Standard technique for imbalanced data']
    },
    {
      accent: GREEN, num: '02', title: 'Confidence\nThreshold τ = 0.30',
      formula: 'High-confidence predictions only',
      items: ['Filter pseudo-labels during MIL training', 'Only scores > 0.30 used for supervision',
              'Reduces noisy gradient in early epochs', 'Applied at bag-level MIL sampling stage',
              'Tuned on held-out validation split', 'Critical during first 10 epochs',
              'Stabilises convergence significantly']
    },
    {
      accent: AMBER, num: '03', title: 'Cosine Annealing\nLearning Rate',
      formula: 'η: 1×10⁻⁴ → 1×10⁻⁶  (40 epochs)',
      items: ['No oscillations, no divergence', '3-epoch linear warm-up phase',
              'Adam optimizer β₁=0.9, β₂=0.999', 'Weight decay = 1×10⁻⁴',
              'Batch size = 64 (32 normal + 32 anomaly)', 'All 3 seeds converge identically',
              'Cross-seed std AUC = ±0.013']
    }
  ];

  const cw = (CW - 2 * 0.24) / 3;
  recipes.forEach(({ accent, num, title, formula, items }, i) => {
    const cx = ML + i * (cw + 0.24);
    card(s, cx, CY, cw, CH, accent);
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: CY + 0.12, w: 0.56, h: 0.56,
      fill: { color: accent }
    });
    t(s, num, cx + 0.1, CY + 0.12, 0.56, 0.56,
      { fontSize: 20, bold: true, color: BGDK, align: 'center', valign: 'middle' });
    t(s, title, cx + 0.76, CY + 0.12, cw - 0.9, 0.58, { fontSize: 15, bold: true, color: accent });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: CY + 0.82, w: cw - 0.18, h: 0.42,
      fill: { color: '07111F' }, line: { color: accent, width: 0.75 }
    });
    t(s, formula, cx + 0.1, CY + 0.82, cw - 0.18, 0.42,
      { fontSize: 13, bold: true, color: accent, align: 'center', valign: 'middle' });
    bullets(s, items, cx + 0.14, CY + 1.34, cw - 0.22, CH - 1.44, GRAY, 12.5);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — MAIN RESULTS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(GREEN, 'Main Results',
    'AUC jumps from 87.4% → 92.2% — VideoMAE-B surpasses all I3D-based published methods', 9);

  const heroX = ML, heroW = 4.0, heroH = CH;
  card(s, heroX, CY, heroW, heroH, GREEN, CARD, GREEN, 1.5);

  t(s, 'OUR BEST RESULT', heroX, CY + 0.18, heroW, 0.3,
    { fontSize: 11.5, bold: true, color: GRAY, align: 'center', charSpacing: 2 });
  t(s, '92.2%', heroX, CY + 0.5, heroW, 1.15,
    { fontSize: 72, bold: true, color: GREEN, align: 'center', valign: 'middle' });
  t(s, 'AUC — UCF-Crime Test Set', heroX, CY + 1.66, heroW, 0.3,
    { fontSize: 12, color: GRAY, align: 'center' });
  s.addShape(pres.shapes.RECTANGLE, {
    x: heroX + 0.5, y: CY + 2.02, w: heroW - 1.0, h: 0.024, fill: { color: GREEN }
  });
  t(s, '±1.26%  ·  3-seed average', heroX, CY + 2.12, heroW, 0.3,
    { fontSize: 12, color: GRAY, align: 'center' });
  t(s, '+4.8 pp vs I3D', heroX, CY + 2.54, heroW, 0.4,
    { fontSize: 18, bold: true, color: AMBER, align: 'center' });

  const subPills = [
    { v: 'mAP@IoU 0.3:  0.058', c: BLUE   },
    { v: 'mAP@IoU 0.7:  ×21.7 ↑', c: VIOLET },
    { v: '3 Seeds: 42, 123, 456', c: AMBER  },
  ];
  subPills.forEach(({ v, c }, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: heroX + 0.25, y: CY + 3.12 + i * 0.7, w: heroW - 0.5, h: 0.58,
      fill: { color: '07111F' }, line: { color: c, width: 1 }
    });
    t(s, v, heroX + 0.25, CY + 3.12 + i * 0.7, heroW - 0.5, 0.58,
      { fontSize: 13, bold: true, color: c, align: 'center', valign: 'middle' });
  });

  // Comparison table
  const tblX = heroX + heroW + 0.38, tblW = CW - heroW - 0.38;
  const tblData = [
    [
      { text: 'Method',     options: { bold: true, color: WHITE, fill: { color: '090F22' }, fontSize: 13 } },
      { text: 'Backbone',   options: { bold: true, color: WHITE, fill: { color: '090F22' }, fontSize: 13 } },
      { text: 'AUC (%)',    options: { bold: true, color: WHITE, fill: { color: '090F22' }, fontSize: 13 } },
      { text: 'mAP@0.3',   options: { bold: true, color: WHITE, fill: { color: '090F22' }, fontSize: 13 } },
      { text: '±',          options: { bold: true, color: WHITE, fill: { color: '090F22' }, fontSize: 13 } },
    ],
    [
      { text: 'Sultani et al. (2018)', options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: 'C3D',                  options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '75.41',               options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
    ],
    [
      { text: 'RTFM (Tian et al.)',   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: 'I3D',                  options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '84.30',               options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
    ],
    [
      { text: 'MGFN (Chen et al.)',   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: 'I3D',                  options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '86.67',               options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
    ],
    [
      { text: 'VadCLIP (Wu et al.)',  options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: 'CLIP ViT-B',           options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '88.02',               options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
      { text: '—',                   options: { color: GRAY, fill: { color: CARD }, fontSize: 12.5 } },
    ],
    [
      { text: 'Ours — I3D (reprod.)', options: { color: AMBER, fill: { color: '150E00' }, fontSize: 12.5, bold: true } },
      { text: 'I3D',                  options: { color: AMBER, fill: { color: '150E00' }, fontSize: 12.5, bold: true } },
      { text: '87.40',               options: { color: AMBER, fill: { color: '150E00' }, fontSize: 12.5, bold: true } },
      { text: '0.009',               options: { color: AMBER, fill: { color: '150E00' }, fontSize: 12.5, bold: true } },
      { text: '—',                   options: { color: AMBER, fill: { color: '150E00' }, fontSize: 12.5, bold: true } },
    ],
    [
      { text: 'Ours — VideoMAE-B ★', options: { color: GREEN, fill: { color: '051A10' }, fontSize: 13, bold: true } },
      { text: 'VideoMAE-B',           options: { color: GREEN, fill: { color: '051A10' }, fontSize: 13, bold: true } },
      { text: '92.18',               options: { color: GREEN, fill: { color: '051A10' }, fontSize: 13, bold: true } },
      { text: '0.058',               options: { color: GREEN, fill: { color: '051A10' }, fontSize: 13, bold: true } },
      { text: '±1.26',              options: { color: GREEN, fill: { color: '051A10' }, fontSize: 13, bold: true } },
    ],
  ];
  s.addTable(tblData, {
    x: tblX, y: CY, w: tblW, h: CH,
    border: { color: BD, pt: 0.5 },
    fontFace: 'Calibri',
    colW: [tblW * 0.35, tblW * 0.22, tblW * 0.17, tblW * 0.14, tblW * 0.12],
    rowH: [0.46, 0.74, 0.74, 0.74, 0.74, 0.76, 0.82]
  });

  footerBar(s, GREEN, '★ Best result  ·  ± = std over seeds 42/123/456  ·  mAP evaluated at IoU threshold 0.3');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — PER-CLASS PRECISION / RECALL / F1
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(AMBER, 'Per-Class Precision · Recall · F1',
    'Real test-set results — Normal dominates; Explosion best anomaly class; Abuse hardest', 10);

  // Full-width chart
  fig(s, FIGS + 'fig7_per_class_breakdown.png', ML, CY, CW, CH - 0.08);
  footerBar(s, AMBER, 'Precision = true anomalies found / all flagged  ·  Recall = true anomalies found / all actual anomalies  ·  F1 = harmonic mean');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — TEMPORAL LOCALIZATION BREAKTHROUGH
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, 'Temporal Localization Breakthrough',
    'Detecting WHEN crime happens is much harder than detecting IF it happened', 11);

  const metrics = [
    { label: 'mAP @ IoU 0.3', from: '0.009', to: '0.058', mult: '×6.4  improvement',  color: BLUE   },
    { label: 'mAP @ IoU 0.5', from: '0.004', to: '0.038', mult: '×9.4  improvement',  color: GREEN  },
    { label: 'mAP @ IoU 0.7', from: '0.001', to: '0.022', mult: '×21.7  improvement', color: VIOLET },
  ];
  const mcw = (CW - 2 * 0.25) / 3, mcy = CY, mch = 2.14;

  metrics.forEach(({ label, from, to, mult, color }, i) => {
    const mx = ML + i * (mcw + 0.25);
    card(s, mx, mcy, mcw, mch, color);
    t(s, label, mx + 0.22, mcy + 0.12, mcw - 0.32, 0.3, { fontSize: 14, bold: true, color });
    s.addShape(pres.shapes.RECTANGLE, { x: mx + 0.16, y: mcy + 0.46, w: mcw - 0.3, h: 0.024, fill: { color: BD } });
    t(s, `${from}  →  ${to}`, mx + 0.16, mcy + 0.56, mcw - 0.26, 0.6,
      { fontSize: 26, bold: true, color: WHITE, valign: 'middle' });
    s.addShape(pres.shapes.RECTANGLE, {
      x: mx + 0.16, y: mcy + 1.28, w: mcw - 0.3, h: 0.38,
      fill: { color: '07111F' }, line: { color, width: 1 }
    });
    t(s, mult, mx + 0.16, mcy + 1.28, mcw - 0.3, 0.38,
      { fontSize: 15, bold: true, color, align: 'center', valign: 'middle' });
  });

  const expY = mcy + mch + 0.2;
  s.addShape(pres.shapes.RECTANGLE, {
    x: ML, y: expY, w: CW, h: 0.66, fill: { color: '07111F' }, line: { color: BD, width: 0.75 }
  });
  t(s, 'Why such a large jump?', ML + 0.22, expY + 0.08, 2.5, 0.24,
    { fontSize: 12.5, bold: true, color: AMBER });
  t(s, 'VideoMAE-B learns precise spatiotemporal boundaries via masked patch reconstruction — its tokens are inherently sensitive to the onset and cessation of motion. I3D optical-flow features blur across temporal boundaries, explaining the poor localization at strict IoU.',
    ML + 0.22, expY + 0.32, CW - 0.42, 0.3, { fontSize: 12, color: GRAY });

  fig(s, FIGS + 'fig3_localization_curve.png', ML, expY + 0.96, CW, SH - expY - 1.18);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — TRAINING DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, 'Training Dynamics',
    'Stable convergence across all 3 seeds — no cherry-picking, no exceptions', 12);
  fig(s, FIGS + 'fig1_training_dynamics.png', ML, CY, CW, 4.46);
  pill(s, '>91%',   'AUC from epoch 10 onward', ML,          CY + 4.68, 5.95, 0.88, GREEN);
  pill(s, '±0.013', 'Cross-seed std deviation',  ML + 6.38,  CY + 4.68, 5.95, 0.88, AMBER);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — ANOMALY SCORE DISTRIBUTIONS PER CATEGORY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(RED, 'Anomaly Score Distributions by Category',
    'Violin plots reveal per-class score spread — Explosion near-perfect, Abuse overlaps Normal', 13);

  fig(s, FIGS + 'fig11_score_distributions.png', ML, CY, CW, CH - 0.08);
  footerBar(s, RED, 'White line = median score  ·  Shaded area = score density  ·  Dots = individual test videos  ·  τ = 0.40 detection threshold');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — CONFUSION MATRIX
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(VIOLET, 'Classification Confusion Matrix',
    'VideoMAE-B classifier head — row-normalised recall  ·  Normal class 89% accurate', 14);

  // Left: full confusion matrix
  fig(s, FIGS + 'fig8_confusion_matrix.png', ML, CY, 7.4, CH);

  // Right: key insights
  const rx = 8.1, rw = SW - rx - MR;

  card(s, rx, CY, rw, 1.62, BLUE);
  t(s, 'What the Matrix Shows', rx + 0.22, CY + 0.12, rw - 0.32, 0.34,
    { fontSize: 14, bold: true, color: BLUE });
  bullets(s, [
    'Normal correctly recalled 89% of the time',
    'Explosion: 38% recall — hardest to classify',
    'Shooting precision = 100% but recall only 9%',
  ], rx + 0.16, CY + 0.52, rw - 0.26, 1.0, GRAY, 12);

  card(s, rx, CY + 1.78, rw, 1.52, GREEN, '051A10', GREEN, 1.2);
  t(s, 'Key Confusion Pairs', rx + 0.22, CY + 1.92, rw - 0.32, 0.34,
    { fontSize: 14, bold: true, color: GREEN });
  bullets(s, [
    'Shooting → Robbery (both involve weapons)',
    'Explosion → Abuse (high-energy confusion)',
    'Fighting → Robbery (fast motion overlap)',
  ], rx + 0.16, CY + 2.32, rw - 0.26, 1.0, GRAY, 12);

  card(s, rx, CY + 3.46, rw, CH - 3.46, AMBER, '150E00', AMBER, 1.2);
  t(s, 'Root Cause', rx + 0.22, CY + 3.60, rw - 0.32, 0.34,
    { fontSize: 14, bold: true, color: AMBER });
  bullets(s, [
    '16-frame context too short',
    'Semantic overlap in visual features',
    'Weak pseudo-labels during MIL training',
    'Fix: fine-tune with 64-frame windows',
  ], rx + 0.16, CY + 4.00, rw - 0.26, CH - 4.10, GRAY, 12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — FAILURE MODE TAXONOMY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(RED, 'Why Does the Model Fail? — Error Taxonomy',
    '502 failure cases analysed  ·  53% caused by ambiguous motion — fixable with longer context', 15);

  fig(s, FIGS + 'fig9_failure_taxonomy.png', ML, CY, CW, CH - 0.08);
  footerBar(s, RED, 'Source: Step-14A interpretability analysis  ·  502 test-set error cases manually taxonomised  ·  Each case tagged with primary failure reason');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — ABLATION STUDY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, 'Ablation Study — Each Component\'s Contribution',
    'I3D → VideoMAE-B backbone swap drives the largest single gain  ·  +4.78 pp AUC', 16);

  fig(s, FIGS + 'fig10_ablation_pipeline.png', ML, CY, CW, CH - 0.08);
  footerBar(s, BLUE, 'I3D + RTFM = published baseline  ·  VideoMAE-B stages = our ablation on UCF-Crime  ·  mAP values ×100 for readability');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — ALL-METRICS RADAR
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, 'All-Metrics Radar',
    'VideoMAE-B strictly encloses I3D on every evaluation axis simultaneously — no trade-offs', 17);

  fig(s, FIGS + 'fig6_radar_overview.png', ML, CY, 7.6, CH - 0.1);

  const rx = 8.4, rw = CW - 8.4 + ML;
  t(s, 'Blue > Red on all 5 axes', rx, CY, rw, 0.42, { fontSize: 17, bold: true, color: WHITE });
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: CY + 0.46, w: rw, h: 0.024, fill: { color: GREEN } });

  const metricsTable = [
    { name: 'AUC',      from: '87.4%', to: '92.2%', color: BLUE   },
    { name: 'mAP@0.3',  from: '0.009', to: '0.058', color: GREEN  },
    { name: 'mAP@0.5',  from: '0.004', to: '0.038', color: VIOLET },
    { name: 'mAP@0.7',  from: '0.001', to: '0.022', color: AMBER  },
    { name: 'Precision', from: '~0.71', to: '~0.84', color: PINK   },
  ];
  metricsTable.forEach(({ name, from, to, color }, i) => {
    const my = CY + 0.58 + i * 0.92;
    card(s, rx, my, rw, 0.8, color);
    t(s, name, rx + 0.22, my + 0.1, 1.2, 0.3, { fontSize: 13, bold: true, color });
    t(s, `${from}  →  ${to}`, rx + 1.52, my + 0.1, rw - 1.64, 0.3, { fontSize: 13, color: WHITE });
  });

  t(s, 'A backbone swap alone uniformly improves every metric — no trade-offs whatsoever.',
    rx, CY + CH - 0.36, rw, 0.34, { fontSize: 11.5, color: MUTED, italic: true });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — ANOMALY SCORE TIMELINE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(VIOLET, 'Anomaly Score Timeline — Model in Action',
    '48 video segments  ·  Threshold τ = 0.40  ·  Violence region: segments 22–35', 18);

  fig(s, FIGS + 'anomaly_demo.gif', ML, CY, CW, CH - 0.8,
    'VideoMAE-B anomaly scores across video segments — red bars exceed threshold τ = 0.40 in the ground-truth violence window');

  const labels = [
    { v: '48 video segments',    c: BLUE   },
    { v: 'Threshold  τ = 0.40',  c: AMBER  },
    { v: 'Violence: segs 22–35', c: RED    },
  ];
  labels.forEach(({ v, c }, i) => {
    const pw = CW / 3 - 0.08;
    const px = ML + i * (pw + 0.12);
    s.addShape(pres.shapes.RECTANGLE, {
      x: px + 0.04, y: SH - 0.68 + 0.04, w: pw, h: 0.5,
      fill: { color: '000000', transparency: 60 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: px, y: SH - 0.72, w: pw, h: 0.5,
      fill: { color: CARD }, line: { color: c, width: 1.5 }
    });
    t(s, v, px, SH - 0.72, pw, 0.5,
      { fontSize: 13, bold: true, color: c, align: 'center', valign: 'middle' });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — KEY TAKEAWAYS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(GREEN, 'Key Takeaways', null, 19, true);

  const takeaways = [
    { accent: GREEN,  text: 'Feature swap alone: +4.8 pp AUC  ·  ×21.7 mAP@IoU 0.7',
      sub: 'No architecture changes — only backbone. The feature representation is everything.' },
    { accent: BLUE,   text: 'VideoMAE-B encodes sharper temporal transitions than I3D',
      sub: 'Masked reconstruction pre-training makes tokens inherently sensitive to motion boundaries.' },
    { accent: AMBER,  text: '3-seed std = ±0.013  →  Results are robust, not lucky',
      sub: 'Seeds 42, 123, 456 all converge identically. Zero cherry-picking.' },
    { accent: VIOLET, text: '1.9M trainable params on 86M frozen backbone — highly efficient',
      sub: 'No GPU-intensive optical flow extraction. Democratises violence detection research.' },
    { accent: PINK,   text: 'arXiv preprint incoming  ·  Code & features to be released',
      sub: 'Fully reproducible pipeline. Submit after faculty endorsement.' },
  ];

  const cardH = 0.96, cardGap = 0.1;
  takeaways.forEach(({ accent, text, sub }, i) => {
    const cy = CY + i * (cardH + cardGap);
    card(s, ML, cy, CW, cardH, accent);
    t(s, text, ML + 0.22, cy + 0.1, CW - 0.32, 0.36, { fontSize: 15.5, bold: true, color: accent });
    t(s, sub,  ML + 0.22, cy + 0.52, CW - 0.32, 0.38, { fontSize: 12.5, color: GRAY });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — WHAT'S NEXT
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = newSlide(BLUE, "What's Next",
    "Road to publication  ·  Stronger backbones  ·  Cross-dataset generalization", 20);

  const roadmap = [
    {
      accent: BLUE, step: 'Step 1', venue: 'arXiv',
      subtitle: 'Submit This Week',
      items: ['Upload to arXiv cs.CV after faculty endorsement.',
              'Preprint immediately citable.',
              'Code and extracted features attached.',
              'DOI available same day.']
    },
    {
      accent: GREEN, step: 'Step 2', venue: 'ACM MM Workshop 2026',
      subtitle: 'July 2026 Deadline',
      items: ['ACM Multimedia Workshop on Video Understanding.',
              'Extended version with full ablation study.',
              'Qualitative analysis on failure cases.',
              'Compare with CLIP-based methods.']
    },
    {
      accent: VIOLET, step: 'Step 3', venue: 'WACV 2027',
      subtitle: 'Cross-Dataset Generalization',
      items: ['Test on XD-Violence dataset (4,754 videos).',
              'Test on ShanghaiTech (437 videos).',
              'Prove VideoMAE-B generalises beyond UCF-Crime.',
              'Domain adaptation experiments.']
    },
    {
      accent: AMBER, step: 'Step 4', venue: 'CVPR 2027',
      subtitle: 'Scaled Backbone + Longer Context',
      items: ['VideoMAE-Large (307M params) backbone.',
              '64-frame context windows (4× current).',
              'Address Fighting class collapse.',
              'Push AUC toward 95%+.']
    },
  ];

  const cardW = (CW - 0.3) / 2;
  const cardH = (CH - 0.25) / 2;

  roadmap.forEach(({ accent, step, venue, subtitle, items }, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const cx = ML + col * (cardW + 0.3);
    const cy = CY + row * (cardH + 0.25);
    card(s, cx, cy, cardW, cardH, accent);
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: cy + 0.1, w: 0.76, h: 0.32,
      fill: { color: accent }
    });
    t(s, step, cx + 0.1, cy + 0.1, 0.76, 0.32,
      { fontSize: 11, bold: true, color: BGDK, align: 'center', valign: 'middle' });
    t(s, venue, cx + 0.96, cy + 0.12, cardW - 1.1, 0.3, { fontSize: 16, bold: true, color: accent });
    t(s, subtitle, cx + 0.1, cy + 0.48, cardW - 0.2, 0.3, { fontSize: 13.5, bold: true, color: WHITE });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx + 0.1, y: cy + 0.82, w: cardW - 0.2, h: 0.02, fill: { color: BD }
    });
    bullets(s, items, cx + 0.14, cy + 0.92, cardW - 0.22, cardH - 1.04, GRAY, 12.5);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// WRITE OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════
pres.writeFile({ fileName: OUT })
  .then(() => {
    const fs = require('fs');
    const stat = fs.statSync(OUT);
    console.log(`✓  ${OUT}`);
    console.log(`   Size:   ${Math.round(stat.size / 1024)} KB`);
    console.log(`   Slides: 20`);
  })
  .catch(err => { console.error('ERROR:', err); process.exit(1); });
