#!/usr/bin/env python3
"""Step-14D: Boundary precision analysis.

Primary target: UCF-Crime test set with temporal annotations.
Optional: include XD-Violence if temporal spans are available in manifest/master.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_ablation_checkpoint import (
    infer_and_postprocess_video,
    infer_full_sequence_chunked_variant,
    load_model,
)
from train_rtfm_trn_boundary import (
    SEGMENT_LEN,
    choose_device,
    moving_average,
    parse_class_names,
    parse_float_list,
    read_csv,
    resolve,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-14D boundary precision analysis")
    p.add_argument("--project-root", type=Path, default=Path.cwd())

    p.add_argument(
        "--ucf-feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--ucf-master-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    p.add_argument(
        "--ucf-temporal-root",
        type=Path,
        default=Path("data/ucf_crime/annotations/temporal_segments"),
    )
    p.add_argument("--ucf-split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument("--include-xd", action="store_true", help="Include XD if temporal spans are available")
    p.add_argument(
        "--xd-feature-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv"),
    )
    p.add_argument(
        "--xd-master-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_master.csv"),
    )
    p.add_argument(
        "--xd-temporal-root",
        type=Path,
        default=Path("data/xd_violence/annotations/temporal_segments"),
    )
    p.add_argument("--xd-split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument("--model-kind", choices=["step6_trn", "step7_boundary", "step8_progressive"], default="step7_boundary")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/step10_ablations/k1/train/checkpoints/best.pt"))

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--target-segments", type=int, default=32)
    p.add_argument("--trn-layers", type=int, default=2)
    p.add_argument("--trn-heads", type=int, default=4)
    p.add_argument("--trn-ffn-mult", type=int, default=4)
    p.add_argument("--trn-dropout", type=float, default=0.1)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    p.add_argument("--topk-ratio", type=float, default=0.125)
    p.add_argument("--infer-window", type=int, default=32)
    p.add_argument("--infer-stride", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--min-event-len", type=int, default=5)
    p.add_argument("--merge-gap", type=int, default=0)
    p.add_argument("--boundary-radius", type=int, default=2)
    p.add_argument("--boundary-refine", dest="boundary_refine", action="store_true")
    p.add_argument("--no-boundary-refine", dest="boundary_refine", action="store_false")
    p.set_defaults(boundary_refine=True)

    p.add_argument("--tiou-thresholds", type=str, default="0.3,0.5,0.7")
    p.add_argument("--boundary-peak-tol", type=int, default=2, help="peak hit tolerance in segments")
    p.add_argument("--boundary-profile-window", type=int, default=8, help="half window for boundary confidence profile")
    p.add_argument("--qual-videos", type=int, default=5)

    p.add_argument("--class-names", type=str, default="normal,fighting,shooting,explosion,robbery,abuse")
    p.add_argument("--normal-class", type=str, default="normal")

    p.add_argument("--out-dir", type=Path, default=Path("outputs/step14_interpretability/step14d"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    txt = "".join(out).strip("_")
    return txt[:200] if txt else "video"


def temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    ua = (a1 - a0 + 1)
    ub = (b1 - b0 + 1)
    union = ua + ub - inter
    return float(inter) / float(max(union, 1))


def greedy_match(
    gt_spans: Sequence[Tuple[int, int]],
    pred_spans: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_spans):
        for pi, p in enumerate(pred_spans):
            iou = temporal_iou(g, p)
            pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_g = set()
    used_p = set()
    matched: List[Tuple[int, int, float]] = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        if iou <= 0.0:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gi, pi, float(iou)))

    unmatched_gt = [i for i in range(len(gt_spans)) if i not in used_g]
    unmatched_pred = [i for i in range(len(pred_spans)) if i not in used_p]
    return matched, unmatched_gt, unmatched_pred


def parse_gt_spans_from_json(ann_path: Path, num_segments: int) -> List[Tuple[int, int]]:
    if not ann_path.exists():
        return []
    try:
        payload = json.loads(ann_path.read_text())
    except Exception:
        return []

    segs = payload.get("segments", []) if isinstance(payload, dict) else []
    out: List[Tuple[int, int]] = []
    for seg in segs:
        try:
            sf = int(seg.get("start_frame", 0))
            ef = int(seg.get("end_frame", 0))
        except Exception:
            continue
        s = max(0, sf // SEGMENT_LEN)
        e = max(s, ef // SEGMENT_LEN)
        if num_segments > 0:
            s = min(s, num_segments - 1)
            e = min(e, num_segments - 1)
        out.append((int(s), int(e)))
    return out


def find_annotation_path(
    dataset: str,
    video_id: str,
    master_row: Dict[str, str],
    project_root: Path,
    temporal_root: Path,
) -> Optional[Path]:
    ann_raw = str(master_row.get("temporal_annotation_path", "")).strip()
    if ann_raw:
        p = Path(ann_raw)
        ap = p if p.is_absolute() else resolve(project_root, p)
        if ap.exists():
            return ap

    # Fallback convention used in UCF prep script and potential dataset-specific mirrors.
    fallback = resolve(project_root, temporal_root) / f"{video_id}.json"
    if fallback.exists():
        return fallback

    # XD fallback from old relative style if present.
    if dataset == "xd_violence":
        fallback2 = resolve(project_root, Path("data/xd_violence/annotations/temporal_segments")) / f"{video_id}.json"
        if fallback2.exists():
            return fallback2

    return None


def boundary_mean_near_edge(bnd: np.ndarray, edge_idx: int, tol: int) -> float:
    if bnd.size == 0:
        return float("nan")
    l = max(0, int(edge_idx) - int(tol))
    r = min(int(bnd.shape[0]) - 1, int(edge_idx) + int(tol))
    if l > r:
        return float("nan")
    return float(np.mean(bnd[l : r + 1]))


def make_profile_values(
    bnd: np.ndarray,
    edges: Sequence[int],
    half_window: int,
) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {o: [] for o in range(-half_window, half_window + 1)}
    if bnd.size == 0:
        return out
    n = int(bnd.shape[0])
    for e in edges:
        ei = int(e)
        for off in range(-half_window, half_window + 1):
            j = ei + off
            if 0 <= j < n:
                out[off].append(float(bnd[j]))
    return out


def merge_profile(dst: Dict[int, List[float]], src: Dict[int, List[float]]) -> None:
    for k, vals in src.items():
        dst[k].extend(vals)


def summarize_match_rows(rows: List[Dict[str, object]], total_gt: int, total_pred: int, tiou_ths: Sequence[float]) -> Dict[str, object]:
    ious = np.array([as_float(r.get("t_iou", 0.0), 0.0) for r in rows], dtype=np.float64)
    start_err = np.array([as_float(r.get("start_error", 0.0), 0.0) for r in rows], dtype=np.float64)
    end_err = np.array([as_float(r.get("end_error", 0.0), 0.0) for r in rows], dtype=np.float64)

    out: Dict[str, object] = {
        "matched_pairs": int(len(rows)),
        "gt_events": int(total_gt),
        "pred_events": int(total_pred),
        "unmatched_gt": int(max(total_gt - len(rows), 0)),
        "unmatched_pred": int(max(total_pred - len(rows), 0)),
    }

    if len(rows) == 0:
        out.update(
            {
                "mean_tiou": float("nan"),
                "median_tiou": float("nan"),
                "mean_abs_start_error": float("nan"),
                "mean_abs_end_error": float("nan"),
                "signed_start_bias": float("nan"),
                "signed_end_bias": float("nan"),
                "mean_abs_duration_error": float("nan"),
            }
        )
        for t in tiou_ths:
            out[f"pct_iou_ge_{t}"] = float("nan")
        return out

    out.update(
        {
            "mean_tiou": float(np.mean(ious)),
            "median_tiou": float(np.median(ious)),
            "mean_abs_start_error": float(np.mean(np.abs(start_err))),
            "mean_abs_end_error": float(np.mean(np.abs(end_err))),
            "signed_start_bias": float(np.mean(start_err)),
            "signed_end_bias": float(np.mean(end_err)),
            "mean_abs_duration_error": float(np.mean(np.abs(np.array([as_float(r.get("duration_error", 0.0), 0.0) for r in rows])))),
        }
    )
    for t in tiou_ths:
        out[f"pct_iou_ge_{t}"] = float(np.mean((ious >= float(t)).astype(np.float64)))
    return out


def plot_iou_distribution(iou_before: np.ndarray, iou_after: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0.0, 1.0, 21)
    if iou_before.size > 0:
        ax.hist(iou_before, bins=bins, alpha=0.45, color="#ff7f0e", label="before refine", density=False)
    if iou_after.size > 0:
        ax.hist(iou_after, bins=bins, alpha=0.45, color="#1f77b4", label="after refine", density=False)
    ax.set_xlabel("Temporal IoU")
    ax.set_ylabel("Count")
    ax.set_title("Matched Event IoU Distribution")
    ax.grid(alpha=0.2)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bias_distribution(start_err: np.ndarray, end_err: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.arange(-40, 41, 2)

    axes[0].hist(start_err, bins=bins, color="#2ca02c", alpha=0.75)
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_title("Start Bias (pred_start - gt_start)")
    axes[0].set_xlabel("Segments")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)

    axes[1].hist(end_err, bins=bins, color="#d62728", alpha=0.75)
    axes[1].axvline(0, color="black", ls="--", lw=1)
    axes[1].set_title("End Bias (pred_end - gt_end)")
    axes[1].set_xlabel("Segments")
    axes[1].grid(alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_profile(
    start_prof: Dict[int, List[float]],
    end_prof: Dict[int, List[float]],
    away_mean: float,
    out_path: Path,
) -> None:
    offs = sorted(start_prof.keys())
    start_mean = [float(np.mean(start_prof[o])) if start_prof[o] else float("nan") for o in offs]
    end_mean = [float(np.mean(end_prof[o])) if end_prof[o] else float("nan") for o in offs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(offs, start_mean, marker="o", ms=3, lw=1.5, color="#1f77b4", label="around GT starts")
    ax.plot(offs, end_mean, marker="o", ms=3, lw=1.5, color="#ff7f0e", label="around GT ends")
    if np.isfinite(away_mean):
        ax.axhline(float(away_mean), color="#6c757d", ls="--", lw=1.0, label="away-boundary mean")
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_xlabel("Offset from GT boundary (segments)")
    ax.set_ylabel("Boundary confidence b_t")
    ax.set_title("Boundary Confidence Profile Around True Boundaries")
    ax.grid(alpha=0.2)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qual_case(
    rec: Dict[str, object],
    threshold: float,
    out_path: Path,
) -> None:
    seg_scores = np.array(rec.get("seg_scores", []), dtype=np.float64)
    bnd = np.array(rec.get("bnd_scores", []), dtype=np.float64)
    gt_spans = [(int(a), int(b)) for a, b in rec.get("gt_spans", [])]
    pred_spans = [(int(a), int(b)) for a, b in rec.get("pred_spans_after", [])]

    t = int(seg_scores.shape[0])
    x = np.arange(t, dtype=np.int32)
    xb = np.arange(bnd.shape[0], dtype=np.int32)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.1], hspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, seg_scores, color="#1f77b4", lw=1.8, label="anomaly score")
    ax1.axhline(float(threshold), color="#6c757d", ls="--", lw=1.0, label=f"threshold={threshold:.2f}")

    for i, (s, e) in enumerate(gt_spans):
        ax1.axvspan(s, e, color="#2ca02c", alpha=0.18, label="GT span" if i == 0 else "")
    for i, (s, e) in enumerate(pred_spans):
        ax1.axvspan(s, e, color="#d62728", alpha=0.18, label="Pred span" if i == 0 else "")

    ax1.set_xlim(0, max(1, t - 1))
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("Anomaly score")
    ax1.grid(alpha=0.2)

    handles, labels = ax1.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    if h2:
        ax1.legend(h2, l2, fontsize=8, loc="upper right")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if bnd.size > 0:
        ax2.plot(xb, bnd, color="#9467bd", lw=1.5, label="boundary confidence b_t")
    for i, (s, e) in enumerate(gt_spans):
        s_edge = max(0, s - 1)
        e_edge = e
        ax2.axvline(s_edge, color="#2ca02c", ls="--", lw=1.0, alpha=0.6, label="GT start edge" if i == 0 else "")
        ax2.axvline(e_edge, color="#17becf", ls=":", lw=1.0, alpha=0.6, label="GT end edge" if i == 0 else "")
    ax2.set_ylabel("b_t")
    ax2.set_xlabel("Segment index")
    ax2.grid(alpha=0.2)

    h3, l3 = ax2.get_legend_handles_labels()
    seen2 = set()
    hh, ll = [], []
    for h, l in zip(h3, l3):
        if l in seen2:
            continue
        seen2.add(l)
        hh.append(h)
        ll.append(l)
    if hh:
        ax2.legend(hh, ll, fontsize=8, loc="upper right")

    title = (
        f"{rec.get('dataset','')} | {rec.get('video_id','')} | "
        f"group={rec.get('qual_group','')} | best_iou_after={as_float(rec.get('best_iou_after', float('nan'))):.3f}"
    )
    fig.suptitle(title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = resolve(root, args.out_dir)
    plots_dir = out_dir / "plots"
    qual_dir = out_dir / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    qual_dir.mkdir(parents=True, exist_ok=True)

    tiou_ths = parse_float_list(args.tiou_thresholds)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not in class_names")
    normal_idx = class_to_idx[args.normal_class]

    set_seed(args.seed)
    device = choose_device(args.device)

    args.checkpoint = resolve(root, args.checkpoint)
    model, _ = load_model(args, num_classes=len(class_names), device=device)

    dataset_cfgs: List[Dict[str, object]] = [
        {
            "name": "ucf_crime",
            "feature_manifest": resolve(root, args.ucf_feature_manifest),
            "master_manifest": resolve(root, args.ucf_master_manifest),
            "temporal_root": resolve(root, args.ucf_temporal_root),
            "split": args.ucf_split,
        }
    ]
    if args.include_xd:
        dataset_cfgs.append(
            {
                "name": "xd_violence",
                "feature_manifest": resolve(root, args.xd_feature_manifest),
                "master_manifest": resolve(root, args.xd_master_manifest),
                "temporal_root": resolve(root, args.xd_temporal_root),
                "split": args.xd_split,
            }
        )

    matched_after_rows: List[Dict[str, object]] = []
    matched_before_rows: List[Dict[str, object]] = []
    video_rows: List[Dict[str, object]] = []
    video_cache: List[Dict[str, object]] = []

    # boundary confidence aggregate stats
    near_vals_all: List[float] = []
    away_vals_all: List[float] = []
    bnd_all: List[float] = []
    bnd_exact_mask_all: List[int] = []

    gt_edge_total = 0
    gt_edge_hits = 0
    pred_peak_total = 0
    pred_peak_hits = 0

    start_profile_all: Dict[int, List[float]] = {o: [] for o in range(-args.boundary_profile_window, args.boundary_profile_window + 1)}
    end_profile_all: Dict[int, List[float]] = {o: [] for o in range(-args.boundary_profile_window, args.boundary_profile_window + 1)}

    dataset_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cfg in dataset_cfgs:
        ds = str(cfg["name"])
        feat_rows = [r for r in read_csv(Path(cfg["feature_manifest"])) if r.get("status", "ok") == "ok"]
        split = str(cfg["split"])
        if split != "all":
            feat_rows = [r for r in feat_rows if r.get("split") == split]

        master_rows = read_csv(Path(cfg["master_manifest"]))
        master_by_vid = {str(r.get("video_id", "")): r for r in master_rows}

        temporal_root = Path(cfg["temporal_root"])

        # Keep only positive videos with available GT spans.
        cand_rows: List[Tuple[Dict[str, str], List[Tuple[int, int]], Path]] = []
        for r in feat_rows:
            if as_int(r.get("binary_label", 0), 0) != 1:
                continue
            vid = str(r.get("video_id", ""))
            mrow = master_by_vid.get(vid, {})
            ann_path = find_annotation_path(ds, vid, mrow, root, temporal_root)
            if ann_path is None:
                continue
            # num_segments from manifest if available; if stale this is still only for clipping.
            nseg = as_int(r.get("num_segments", 0), 0)
            gt_spans = parse_gt_spans_from_json(ann_path, nseg)
            if not gt_spans:
                continue
            cand_rows.append((r, gt_spans, ann_path))

        total = len(cand_rows)
        print(f"{ds}: videos with GT spans = {total}")

        for i, (r, gt_spans, ann_path) in enumerate(cand_rows, 1):
            vid = str(r.get("video_id", ""))

            pred = infer_and_postprocess_video(
                model=model,
                model_kind=args.model_kind,
                row=r,
                project_root=root,
                class_names=class_names,
                normal_idx=normal_idx,
                topk_ratio=args.topk_ratio,
                infer_window=args.infer_window,
                infer_stride=args.infer_stride,
                threshold=args.threshold,
                smooth_window=args.smooth_window,
                min_event_len=args.min_event_len,
                merge_gap=args.merge_gap,
                boundary_radius=args.boundary_radius,
                boundary_refine=args.boundary_refine,
                device=device,
            )

            feat_path = resolve(root, Path(str(r.get("feature_path", ""))))
            with np.load(feat_path, allow_pickle=True) as d:
                feats = d["features"].astype(np.float32)

            seg_scores_raw, _, bnd_scores = infer_full_sequence_chunked_variant(
                model=model,
                model_kind=args.model_kind,
                features=feats,
                num_classes=len(class_names),
                window=args.infer_window,
                stride=args.infer_stride,
                device=device,
            )
            seg_scores = moving_average(seg_scores_raw, int(args.smooth_window)).astype(np.float32)
            bnd = bnd_scores.astype(np.float32)

            pred_spans_before = [(int(a), int(b)) for a, b in pred.get("spans_before_refine", [])]
            pred_spans_after = [(int(a), int(b)) for a, b in pred.get("spans_after_refine", [])]

            match_after, un_gt_after, un_pred_after = greedy_match(gt_spans, pred_spans_after)
            match_before, un_gt_before, un_pred_before = greedy_match(gt_spans, pred_spans_before)

            best_iou_after = max([m[2] for m in match_after], default=0.0)
            best_iou_before = max([m[2] for m in match_before], default=0.0)

            # Matched pair rows for metrics.
            for gi, pi, iou in match_after:
                gs, ge = gt_spans[gi]
                ps, pe = pred_spans_after[pi]

                start_edge = max(0, gs - 1)
                end_edge = min(max(0, bnd.shape[0] - 1), ge) if bnd.size > 0 else 0
                start_conf = boundary_mean_near_edge(bnd, start_edge, args.boundary_peak_tol)
                end_conf = boundary_mean_near_edge(bnd, end_edge, args.boundary_peak_tol)

                matched_after_rows.append(
                    {
                        "dataset": ds,
                        "video_id": vid,
                        "gt_start": int(gs),
                        "gt_end": int(ge),
                        "pred_start": int(ps),
                        "pred_end": int(pe),
                        "t_iou": float(iou),
                        "start_error": int(ps - gs),
                        "end_error": int(pe - ge),
                        "duration_error": int((pe - ps) - (ge - gs)),
                        "gt_duration": int(ge - gs + 1),
                        "pred_duration": int(pe - ps + 1),
                        "video_anomaly_score": float(pred.get("video_anomaly_score", 0.0)),
                        "boundary_conf_start_mean": float(start_conf) if np.isfinite(start_conf) else float("nan"),
                        "boundary_conf_end_mean": float(end_conf) if np.isfinite(end_conf) else float("nan"),
                    }
                )

            for gi, pi, iou in match_before:
                gs, ge = gt_spans[gi]
                ps, pe = pred_spans_before[pi]
                matched_before_rows.append(
                    {
                        "dataset": ds,
                        "video_id": vid,
                        "gt_start": int(gs),
                        "gt_end": int(ge),
                        "pred_start": int(ps),
                        "pred_end": int(pe),
                        "t_iou": float(iou),
                        "start_error": int(ps - gs),
                        "end_error": int(pe - ge),
                        "duration_error": int((pe - ps) - (ge - gs)),
                        "gt_duration": int(ge - gs + 1),
                        "pred_duration": int(pe - ps + 1),
                        "video_anomaly_score": float(pred.get("video_anomaly_score", 0.0)),
                    }
                )

            # Boundary confidence analysis.
            start_edges = [max(0, s - 1) for s, _ in gt_spans]
            end_edges = [min(max(0, bnd.shape[0] - 1), e) if bnd.size > 0 else 0 for _, e in gt_spans]
            gt_edges = sorted(set(start_edges + end_edges))

            if bnd.size > 0:
                near_mask = np.zeros((bnd.shape[0],), dtype=bool)
                exact_mask = np.zeros((bnd.shape[0],), dtype=bool)

                for e in gt_edges:
                    l = max(0, int(e) - int(args.boundary_peak_tol))
                    r2 = min(int(bnd.shape[0]) - 1, int(e) + int(args.boundary_peak_tol))
                    near_mask[l : r2 + 1] = True
                    if 0 <= int(e) < int(bnd.shape[0]):
                        exact_mask[int(e)] = True

                near_vals_all.extend([float(x) for x in bnd[near_mask].tolist()])
                away_vals_all.extend([float(x) for x in bnd[~near_mask].tolist()])

                bnd_all.extend([float(x) for x in bnd.tolist()])
                bnd_exact_mask_all.extend([int(x) for x in exact_mask.astype(np.int32).tolist()])

                # boundary profile around starts/ends
                start_prof = make_profile_values(bnd, start_edges, args.boundary_profile_window)
                end_prof = make_profile_values(bnd, end_edges, args.boundary_profile_window)
                merge_profile(start_profile_all, start_prof)
                merge_profile(end_profile_all, end_prof)

                # peak hit / precision with top-K peaks
                k = max(1, len(gt_edges))
                top_idx = np.argpartition(bnd, -k)[-k:]
                top_idx = top_idx[np.argsort(-bnd[top_idx])]
                peaks = [int(x) for x in top_idx.tolist()]

                gt_edge_total += len(gt_edges)
                pred_peak_total += len(peaks)

                for gei in gt_edges:
                    if any(abs(int(p) - int(gei)) <= int(args.boundary_peak_tol) for p in peaks):
                        gt_edge_hits += 1

                for p in peaks:
                    if any(abs(int(p) - int(gei)) <= int(args.boundary_peak_tol) for gei in gt_edges):
                        pred_peak_hits += 1

            video_rows.append(
                {
                    "dataset": ds,
                    "video_id": vid,
                    "annotation_path": str(ann_path),
                    "num_segments": int(pred.get("num_segments", feats.shape[0])),
                    "gt_events": int(len(gt_spans)),
                    "pred_events_before": int(len(pred_spans_before)),
                    "pred_events_after": int(len(pred_spans_after)),
                    "matched_before": int(len(match_before)),
                    "matched_after": int(len(match_after)),
                    "unmatched_gt_before": int(len(un_gt_before)),
                    "unmatched_pred_before": int(len(un_pred_before)),
                    "unmatched_gt_after": int(len(un_gt_after)),
                    "unmatched_pred_after": int(len(un_pred_after)),
                    "best_iou_before": float(best_iou_before),
                    "best_iou_after": float(best_iou_after),
                    "video_anomaly_score": float(pred.get("video_anomaly_score", 0.0)),
                    "seg_scores": seg_scores.tolist(),
                    "bnd_scores": bnd.tolist(),
                    "gt_spans": [[int(a), int(b)] for a, b in gt_spans],
                    "pred_spans_before": [[int(a), int(b)] for a, b in pred_spans_before],
                    "pred_spans_after": [[int(a), int(b)] for a, b in pred_spans_after],
                }
            )

            dataset_summary[ds]["videos_analyzed"] += 1
            dataset_summary[ds]["gt_events"] += len(gt_spans)
            dataset_summary[ds]["pred_events_before"] += len(pred_spans_before)
            dataset_summary[ds]["pred_events_after"] += len(pred_spans_after)

            if i % 20 == 0 or i == total:
                print(f"  {ds}: processed {i}/{total}")

    # Aggregate metrics overall.
    total_gt = sum(int(v["gt_events"]) for v in video_rows)
    total_pred_before = sum(int(v["pred_events_before"]) for v in video_rows)
    total_pred_after = sum(int(v["pred_events_after"]) for v in video_rows)

    metrics_before = summarize_match_rows(matched_before_rows, total_gt=total_gt, total_pred=total_pred_before, tiou_ths=tiou_ths)
    metrics_after = summarize_match_rows(matched_after_rows, total_gt=total_gt, total_pred=total_pred_after, tiou_ths=tiou_ths)

    # Per-dataset metrics (after refine primary).
    per_dataset_after: Dict[str, Dict[str, object]] = {}
    for ds in sorted({str(r["dataset"]) for r in matched_after_rows} | set(dataset_summary.keys())):
        rows_ds = [r for r in matched_after_rows if str(r.get("dataset", "")) == ds]
        gt_ds = int(dataset_summary[ds].get("gt_events", 0))
        pred_ds = int(dataset_summary[ds].get("pred_events_after", 0))
        per_dataset_after[ds] = summarize_match_rows(rows_ds, total_gt=gt_ds, total_pred=pred_ds, tiou_ths=tiou_ths)

    # Boundary confidence aggregates.
    near_mean = float(np.mean(near_vals_all)) if near_vals_all else float("nan")
    away_mean = float(np.mean(away_vals_all)) if away_vals_all else float("nan")

    corr_boundary = float("nan")
    if bnd_all and bnd_exact_mask_all:
        xb = np.array(bnd_all, dtype=np.float64)
        yb = np.array(bnd_exact_mask_all, dtype=np.float64)
        if xb.size > 1 and np.std(xb) > 1e-12 and np.std(yb) > 1e-12:
            corr_boundary = float(np.corrcoef(xb, yb)[0, 1])

    peak_hit_rate = float(gt_edge_hits / gt_edge_total) if gt_edge_total > 0 else float("nan")
    peak_precision = float(pred_peak_hits / pred_peak_total) if pred_peak_total > 0 else float("nan")

    # Short-vs-long event difficulty on after-match.
    matched_after_iou = np.array([as_float(r.get("t_iou", 0.0), 0.0) for r in matched_after_rows], dtype=np.float64)
    gt_durs = np.array([as_float(r.get("gt_duration", 0.0), 0.0) for r in matched_after_rows], dtype=np.float64)
    duration_split = float(np.median(gt_durs)) if gt_durs.size > 0 else float("nan")
    short_iou = float("nan")
    long_iou = float("nan")
    if gt_durs.size > 0:
        short_mask = gt_durs <= duration_split
        long_mask = gt_durs > duration_split
        if np.any(short_mask):
            short_iou = float(np.mean(matched_after_iou[short_mask]))
        if np.any(long_mask):
            long_iou = float(np.mean(matched_after_iou[long_mask]))

    # Boundary-head impact (before -> after).
    boundary_help = {
        "mean_tiou_delta": as_float(metrics_after.get("mean_tiou", float("nan"))) - as_float(metrics_before.get("mean_tiou", float("nan")), float("nan")),
        "median_tiou_delta": as_float(metrics_after.get("median_tiou", float("nan"))) - as_float(metrics_before.get("median_tiou", float("nan")), float("nan")),
        "mean_abs_start_error_delta": as_float(metrics_after.get("mean_abs_start_error", float("nan"))) - as_float(metrics_before.get("mean_abs_start_error", float("nan")), float("nan")),
        "mean_abs_end_error_delta": as_float(metrics_after.get("mean_abs_end_error", float("nan"))) - as_float(metrics_before.get("mean_abs_end_error", float("nan")), float("nan")),
    }
    for t in tiou_ths:
        k = f"pct_iou_ge_{t}"
        boundary_help[f"{k}_delta"] = as_float(metrics_after.get(k, float("nan"))) - as_float(metrics_before.get(k, float("nan")), float("nan"))

    # Select qualitative videos: 2 good + 3 bad
    vids_sorted_good = sorted(video_rows, key=lambda x: as_float(x.get("best_iou_after", 0.0), 0.0), reverse=True)
    vids_sorted_bad = sorted(video_rows, key=lambda x: as_float(x.get("best_iou_after", 0.0), 0.0))

    qual_selected: List[Dict[str, object]] = []
    used_vids = set()

    for v in vids_sorted_good:
        if len([q for q in qual_selected if q.get("qual_group") == "good"]) >= 2:
            break
        vid_key = (v.get("dataset"), v.get("video_id"))
        if vid_key in used_vids:
            continue
        vv = dict(v)
        vv["qual_group"] = "good"
        qual_selected.append(vv)
        used_vids.add(vid_key)

    for v in vids_sorted_bad:
        if len([q for q in qual_selected if q.get("qual_group") == "bad"]) >= 3:
            break
        vid_key = (v.get("dataset"), v.get("video_id"))
        if vid_key in used_vids:
            continue
        vv = dict(v)
        vv["qual_group"] = "bad"
        qual_selected.append(vv)
        used_vids.add(vid_key)

    qual_selected = qual_selected[: int(max(1, args.qual_videos))]

    # Build plots.
    iou_before = np.array([as_float(r.get("t_iou", 0.0), 0.0) for r in matched_before_rows], dtype=np.float64)
    iou_after = np.array([as_float(r.get("t_iou", 0.0), 0.0) for r in matched_after_rows], dtype=np.float64)
    start_err_after = np.array([as_float(r.get("start_error", 0.0), 0.0) for r in matched_after_rows], dtype=np.float64)
    end_err_after = np.array([as_float(r.get("end_error", 0.0), 0.0) for r in matched_after_rows], dtype=np.float64)

    plot_iou_path = plots_dir / "iou_distribution_before_after.png"
    plot_bias_path = plots_dir / "start_end_bias_distribution.png"
    plot_profile_path = plots_dir / "boundary_confidence_profile.png"

    plot_iou_distribution(iou_before, iou_after, plot_iou_path)
    plot_bias_distribution(start_err_after, end_err_after, plot_bias_path)
    plot_boundary_profile(start_profile_all, end_profile_all, away_mean, plot_profile_path)

    qual_paths: List[str] = []
    for i, v in enumerate(qual_selected, 1):
        outp = qual_dir / f"{i:02d}_{v.get('qual_group','case')}_{safe_name(str(v.get('dataset','')))}_{safe_name(str(v.get('video_id','')))}.png"
        plot_qual_case(v, threshold=args.threshold, out_path=outp)
        qual_paths.append(str(outp))

    # Save CSVs.
    matched_csv = out_dir / "step14d_matched_events_after.csv"
    with matched_csv.open("w", newline="") as f:
        fields = [
            "dataset",
            "video_id",
            "gt_start",
            "gt_end",
            "pred_start",
            "pred_end",
            "t_iou",
            "start_error",
            "end_error",
            "duration_error",
            "gt_duration",
            "pred_duration",
            "video_anomaly_score",
            "boundary_conf_start_mean",
            "boundary_conf_end_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in matched_after_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    video_csv = out_dir / "step14d_video_summary.csv"
    with video_csv.open("w", newline="") as f:
        fields = [
            "dataset",
            "video_id",
            "annotation_path",
            "num_segments",
            "gt_events",
            "pred_events_before",
            "pred_events_after",
            "matched_before",
            "matched_after",
            "unmatched_gt_before",
            "unmatched_pred_before",
            "unmatched_gt_after",
            "unmatched_pred_after",
            "best_iou_before",
            "best_iou_after",
            "video_anomaly_score",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in video_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # Summary JSON.
    summary = {
        "dataset_eval_summary": {
            "datasets": {k: dict(v) for k, v in dataset_summary.items()},
            "videos_analyzed_total": int(sum(v.get("videos_analyzed", 0) for v in dataset_summary.values())),
            "gt_events_total": int(total_gt),
            "pred_events_before_total": int(total_pred_before),
            "pred_events_after_total": int(total_pred_after),
            "matched_pairs_after_total": int(len(matched_after_rows)),
        },
        "boundary_metrics_after": metrics_after,
        "boundary_metrics_before": metrics_before,
        "boundary_head_effect": boundary_help,
        "boundary_confidence_analysis": {
            "mean_bt_near_boundaries": near_mean,
            "mean_bt_away_boundaries": away_mean,
            "bt_exact_boundary_correlation": corr_boundary,
            "boundary_peak_hit_rate": peak_hit_rate,
            "boundary_peak_precision": peak_precision,
            "peak_tolerance_segments": int(args.boundary_peak_tol),
        },
        "short_vs_long": {
            "duration_split_segments_median": duration_split,
            "mean_tiou_short_events": short_iou,
            "mean_tiou_long_events": long_iou,
        },
        "per_dataset_after": per_dataset_after,
        "artifacts": {
            "matched_events_csv": str(matched_csv),
            "video_summary_csv": str(video_csv),
            "iou_plot": str(plot_iou_path),
            "bias_plot": str(plot_bias_path),
            "boundary_profile_plot": str(plot_profile_path),
            "qualitative_plots": qual_paths,
            "report_txt": str(out_dir / "step14d_report.txt"),
            "summary_json": str(out_dir / "step14d_summary.json"),
        },
    }
    (out_dir / "step14d_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Interpretive text snippets.
    start_bias = as_float(metrics_after.get("signed_start_bias", float("nan")), float("nan"))
    end_bias = as_float(metrics_after.get("signed_end_bias", float("nan")), float("nan"))
    start_bias_txt = "late" if np.isfinite(start_bias) and start_bias > 0 else ("early" if np.isfinite(start_bias) else "unknown")
    end_bias_txt = "late" if np.isfinite(end_bias) and end_bias > 0 else ("early" if np.isfinite(end_bias) else "unknown")

    if np.isfinite(boundary_help.get("mean_tiou_delta", float("nan"))) and boundary_help["mean_tiou_delta"] > 0:
        bnd_help_txt = "boundary head/refinement improves localization on average"
    elif np.isfinite(boundary_help.get("mean_tiou_delta", float("nan"))):
        bnd_help_txt = "boundary head/refinement is weak or noisy on average"
    else:
        bnd_help_txt = "boundary-head impact is inconclusive"

    report_sentence = (
        "Boundary analysis shows "
        f"mean tIoU={as_float(metrics_after.get('mean_tiou', float('nan'))):.3f} after refinement with "
        f"{bnd_help_txt}, {start_bias_txt} start bias and {end_bias_txt} end bias, and "
        f"boundary confidence {'aligning' if np.isfinite(peak_hit_rate) and peak_hit_rate >= 0.5 else 'weakly aligning'} "
        "with true transitions."
    )

    # Report text.
    lines: List[str] = []
    lines.append("1) Dataset/eval summary")
    lines.append(f"- videos analyzed: {summary['dataset_eval_summary']['videos_analyzed_total']}")
    lines.append(f"- GT events: {total_gt}")
    lines.append(f"- predicted events (after refine): {total_pred_after}")
    lines.append(f"- matched pairs (after refine): {len(matched_after_rows)}")

    lines.append("")
    lines.append("2) Boundary metrics (after refine)")
    lines.append(f"- mean tIoU: {as_float(metrics_after.get('mean_tiou', float('nan'))):.6f}")
    lines.append(f"- median tIoU: {as_float(metrics_after.get('median_tiou', float('nan'))):.6f}")
    for t in tiou_ths:
        lines.append(f"- % IoU >= {t}: {as_float(metrics_after.get(f'pct_iou_ge_{t}', float('nan'))):.6f}")
    lines.append(f"- mean abs start error: {as_float(metrics_after.get('mean_abs_start_error', float('nan'))):.6f}")
    lines.append(f"- mean abs end error: {as_float(metrics_after.get('mean_abs_end_error', float('nan'))):.6f}")
    lines.append(f"- signed start bias: {as_float(metrics_after.get('signed_start_bias', float('nan'))):.6f}")
    lines.append(f"- signed end bias: {as_float(metrics_after.get('signed_end_bias', float('nan'))):.6f}")

    lines.append("")
    lines.append("3) Boundary-confidence analysis")
    lines.append(f"- mean b_t near boundaries: {near_mean:.6f}" if np.isfinite(near_mean) else "- mean b_t near boundaries: nan")
    lines.append(f"- mean b_t away boundaries: {away_mean:.6f}" if np.isfinite(away_mean) else "- mean b_t away boundaries: nan")
    lines.append(f"- b_t correlation with exact boundary mask: {corr_boundary:.6f}" if np.isfinite(corr_boundary) else "- b_t correlation with exact boundary mask: nan")
    lines.append(f"- boundary peak hit rate (@±{args.boundary_peak_tol}): {peak_hit_rate:.6f}" if np.isfinite(peak_hit_rate) else f"- boundary peak hit rate (@±{args.boundary_peak_tol}): nan")
    lines.append(f"- boundary peak precision (@±{args.boundary_peak_tol}): {peak_precision:.6f}" if np.isfinite(peak_precision) else f"- boundary peak precision (@±{args.boundary_peak_tol}): nan")

    lines.append("")
    lines.append("4) Error breakdown")
    lines.append(f"- start bias direction: {start_bias_txt}")
    lines.append(f"- end bias direction: {end_bias_txt}")
    lines.append(
        f"- short vs long events (mean tIoU): short={short_iou:.6f}, long={long_iou:.6f}, split={duration_split:.2f} segments"
        if np.isfinite(duration_split)
        else "- short vs long events: unavailable"
    )
    lines.append(f"- boundary head effect (mean tIoU delta after-before): {as_float(boundary_help.get('mean_tiou_delta', float('nan'))):.6f}")

    lines.append("")
    lines.append("5) Artifact list")
    lines.append(f"- IoU plot: {plot_iou_path}")
    lines.append(f"- bias plot: {plot_bias_path}")
    lines.append(f"- boundary-confidence profile plot: {plot_profile_path}")
    lines.append(f"- qualitative boundary plots: {qual_dir}")
    lines.append(f"- matched events CSV: {matched_csv}")
    lines.append(f"- video summary CSV: {video_csv}")
    lines.append(f"- summary JSON: {out_dir / 'step14d_summary.json'}")

    lines.append("")
    lines.append("6) Report-ready sentence")
    lines.append(f"- {report_sentence}")

    (out_dir / "step14d_report.txt").write_text("\n".join(lines) + "\n")

    print("Step-14D complete")
    print(f"- videos analyzed: {summary['dataset_eval_summary']['videos_analyzed_total']}")
    print(f"- matched pairs: {len(matched_after_rows)}")
    print(f"- report: {out_dir / 'step14d_report.txt'}")
    print(f"- summary: {out_dir / 'step14d_summary.json'}")
    print(f"- plots: {plots_dir}")


if __name__ == "__main__":
    main()
