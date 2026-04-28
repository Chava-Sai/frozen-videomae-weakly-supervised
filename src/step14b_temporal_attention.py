#!/usr/bin/env python3
"""Step-14B: Temporal attention visualization for selected case studies.

Uses the locked model checkpoint and Step-14A case list to produce:
- per-case anomaly score + predicted/GT spans + TRN-head attention heatmap
- raw attention tensors per case
- per-case observations and cross-dataset takeaway report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    read_csv,
    resolve,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-14B temporal attention visualization")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--step14a-cases",
        type=Path,
        default=Path("outputs/step14_interpretability/step14a/step14a_case_studies.json"),
    )

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
        "--xd-feature-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--xd-master-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_master.csv"),
    )
    p.add_argument(
        "--rwf-feature-manifest",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_features_i3d.csv"),
    )
    p.add_argument(
        "--rwf-master-manifest",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_master.csv"),
    )
    p.add_argument(
        "--sh-feature-manifest",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_features_i3d.csv"),
    )
    p.add_argument(
        "--sh-master-manifest",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_master.csv"),
    )
    p.add_argument(
        "--ucf-temporal-root",
        type=Path,
        default=Path("data/ucf_crime/annotations/temporal_segments"),
    )

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

    p.add_argument("--class-names", type=str, default="normal,fighting,shooting,explosion,robbery,abuse")
    p.add_argument("--normal-class", type=str, default="normal")

    p.add_argument("--attention-layer", type=int, default=-1, help="Layer index for attention heatmap (-1 means last)")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--dpi", type=int, default=140)

    p.add_argument("--out-dir", type=Path, default=Path("outputs/step14_interpretability/step14b"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
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
    return txt[:200] if txt else "case"


def resolve_with_fallback(primary: Path, fallbacks: Sequence[Path]) -> Path:
    if primary.exists():
        return primary
    for p in fallbacks:
        if p.exists():
            return p
    return primary


def parse_temporal_segments_json(path: Path, num_segments: int) -> List[Tuple[int, int]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    segs: List[Tuple[int, int]] = []
    items = payload.get("segments", []) if isinstance(payload, dict) else []
    for seg in items:
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
        segs.append((int(s), int(e)))
    return segs


def find_gt_spans(
    dataset: str,
    row: Dict[str, str],
    master_row: Dict[str, str],
    project_root: Path,
    ucf_temporal_root: Path,
    num_segments: int,
) -> List[Tuple[int, int]]:
    if int(row.get("binary_label", "0")) == 0:
        return []

    ann_raw = str(master_row.get("temporal_annotation_path", "")).strip()
    if ann_raw:
        p = Path(ann_raw)
        ann_path = p if p.is_absolute() else resolve(project_root, p)
        spans = parse_temporal_segments_json(ann_path, num_segments)
        if spans:
            return spans

    if dataset == "ucf_crime":
        vid = str(row.get("video_id", ""))
        fallback = resolve(project_root, ucf_temporal_root) / f"{vid}.json"
        return parse_temporal_segments_json(fallback, num_segments)

    return []


def forward_with_attention(
    model: torch.nn.Module,
    model_kind: str,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
    if model_kind == "step6_trn":
        seg_anom_logits, seg_class_logits, _, _, attn_maps = model(x, return_attention=True)
        return seg_anom_logits, seg_class_logits, None, attn_maps
    if model_kind == "step7_boundary":
        seg_anom_logits, seg_class_logits, _, bnd_scores, _, attn_maps = model(x, return_attention=True)
        return seg_anom_logits, seg_class_logits, bnd_scores, attn_maps
    seg_anom_logits, seg_class_logits, _, bnd_scores, _, attn_maps = model(
        x,
        return_attention=True,
        apply_trn=True,
        apply_boundary=True,
    )
    return seg_anom_logits, seg_class_logits, bnd_scores, attn_maps


def normalize_distribution(v: np.ndarray) -> np.ndarray:
    x = np.maximum(v.astype(np.float64), 1e-12)
    s = float(x.sum())
    if s <= 0:
        return np.full_like(x, 1.0 / max(1, x.size), dtype=np.float64)
    return x / s


def normalized_entropy(v: np.ndarray) -> float:
    p = normalize_distribution(v)
    n = max(1, p.size)
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    return ent / math.log(n) if n > 1 else 0.0


def overlap_ratio(indices: Sequence[int], spans: Sequence[Tuple[int, int]]) -> float:
    if not indices:
        return 0.0
    if not spans:
        return 0.0
    hit = 0
    for i in indices:
        inside = any(int(s) <= int(i) <= int(e) for s, e in spans)
        if inside:
            hit += 1
    return float(hit) / float(len(indices))


def topk_indices(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0:
        return []
    kk = int(max(1, min(k, scores.shape[0])))
    idx = np.argpartition(scores, -kk)[-kk:]
    idx = idx[np.argsort(-scores[idx])]
    return [int(x) for x in idx.tolist()]


def dedup_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2: List[object] = []
    l2: List[str] = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8)


def plot_case(
    case: Dict[str, object],
    seg_scores: np.ndarray,
    pred_spans: Sequence[Tuple[int, int]],
    gt_spans: Sequence[Tuple[int, int]],
    head_focus: np.ndarray,
    layer_idx: int,
    threshold: float,
    out_path: Path,
    dpi: int,
) -> None:
    t = int(seg_scores.shape[0])
    x = np.arange(t, dtype=np.int32)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, seg_scores, color="#1f77b4", lw=1.8, label="anomaly score")
    ax1.axhline(y=float(threshold), color="#6c757d", ls="--", lw=1.0, label=f"threshold={threshold:.2f}")

    for i, (s, e) in enumerate(pred_spans):
        ax1.axvspan(int(s), int(e), color="#d62728", alpha=0.20, label="pred event" if i == 0 else "")
    for i, (s, e) in enumerate(gt_spans):
        ax1.axvspan(int(s), int(e), color="#2ca02c", alpha=0.18, label="gt event" if i == 0 else "")

    ax1.set_xlim(0, max(1, t - 1))
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Segment index")
    ax1.grid(alpha=0.25)

    title = (
        f"{case['video_id']} | {case['dataset']} | gt={case['gt_label']} | "
        f"pred={case['predicted_label']} | status={case['case_group']}"
    )
    ax1.set_title(title, fontsize=11)
    dedup_legend(ax1)

    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(
        head_focus,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="magma",
    )
    ax2.set_xlabel("Segment index")
    ax2.set_ylabel("TRN head")
    ax2.set_title(f"Layer {layer_idx} attention heatmap (head x segment, mean over queries)")
    ax2.set_yticks(np.arange(head_focus.shape[0]))
    ax2.set_yticklabels([f"h{h}" for h in range(head_focus.shape[0])])
    cbar = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.01)
    cbar.set_label("Attention")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def case_observation(
    dataset: str,
    case_group: str,
    status: str,
    entropy_mean: float,
    pred_overlap: float,
    gt_overlap: Optional[float],
    score: float,
) -> str:
    sharp = entropy_mean < 0.60
    diffuse = entropy_mean > 0.80

    if case_group == "success":
        if sharp and pred_overlap >= 0.40:
            return (
                f"Attention is relatively sharp (entropy={entropy_mean:.3f}) and aligns with predicted high-score regions "
                f"(overlap={pred_overlap:.2f}), consistent with a correct {status}."
            )
        return (
            f"Prediction is correct ({status}); attention is usable but less concentrated (entropy={entropy_mean:.3f}). "
            f"The clip-level score remains {'low' if score < 0.5 else 'high'} enough for a stable decision."
        )

    if dataset == "shanghaitech":
        return (
            f"Failure shows likely domain mismatch: attention is {'diffuse' if diffuse else 'not strongly selective'} "
            f"(entropy={entropy_mean:.3f}) with weak anomaly activation (score={score:.4f})."
        )

    if dataset == "rwf_2000":
        return (
            f"Fight miss is consistent with under-attention to fight cues: score={score:.4f}, "
            f"predicted-overlap={pred_overlap:.2f}, entropy={entropy_mean:.3f}."
        )

    if gt_overlap is not None:
        return (
            f"Failure likely from {'diffuse' if diffuse else 'misplaced'} focus: entropy={entropy_mean:.3f}, "
            f"pred-overlap={pred_overlap:.2f}, gt-overlap={gt_overlap:.2f}."
        )

    return (
        f"Failure indicates weak focus on actionable segments (entropy={entropy_mean:.3f}, "
        f"pred-overlap={pred_overlap:.2f}, score={score:.4f})."
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = resolve(root, args.out_dir)
    plots_dir = out_dir / "plots"
    tensors_dir = out_dir / "attention_tensors"
    details_dir = out_dir / "case_details"
    for d in (out_dir, plots_dir, tensors_dir, details_dir):
        d.mkdir(parents=True, exist_ok=True)

    cases_path = resolve(root, args.step14a_cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Step-14A case file not found: {cases_path}")
    cases_all = json.loads(cases_path.read_text())
    if not isinstance(cases_all, list) or not cases_all:
        raise ValueError(f"Step-14A case file is empty or invalid: {cases_path}")
    cases = cases_all[: int(max(1, args.max_cases))]

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class_names")
    normal_idx = class_to_idx[args.normal_class]

    set_seed(args.seed)
    device = choose_device(args.device)

    args.checkpoint = resolve(root, args.checkpoint)
    model, _ = load_model(args, num_classes=len(class_names), device=device)

    manifest_cfg = {
        "ucf_crime": {
            "feature": resolve(root, args.ucf_feature_manifest),
            "master": resolve(root, args.ucf_master_manifest),
        },
        "xd_violence": {
            "feature": resolve_with_fallback(
                resolve(root, args.xd_feature_manifest),
                [resolve(root, Path("data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv"))],
            ),
            "master": resolve(root, args.xd_master_manifest),
        },
        "rwf_2000": {
            "feature": resolve(root, args.rwf_feature_manifest),
            "master": resolve(root, args.rwf_master_manifest),
        },
        "shanghaitech": {
            "feature": resolve(root, args.sh_feature_manifest),
            "master": resolve(root, args.sh_master_manifest),
        },
    }

    feat_by_ds_vid: Dict[str, Dict[str, Dict[str, str]]] = {}
    master_by_ds_vid: Dict[str, Dict[str, Dict[str, str]]] = {}
    for ds, cfg in manifest_cfg.items():
        feat_rows = [r for r in read_csv(cfg["feature"]) if r.get("status", "ok") == "ok"] if cfg["feature"].exists() else []
        master_rows = read_csv(cfg["master"]) if cfg["master"].exists() else []
        feat_by_ds_vid[ds] = {str(r.get("video_id", "")): r for r in feat_rows}
        master_by_ds_vid[ds] = {str(r.get("video_id", "")): r for r in master_rows}

    layer_values: List[int] = []
    head_values: List[int] = []
    extraction_failures: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    for idx, case in enumerate(cases, 1):
        ds = str(case.get("dataset", ""))
        vid = str(case.get("video_id", ""))
        if ds not in feat_by_ds_vid:
            extraction_failures.append(f"{ds}/{vid}: dataset config missing")
            continue

        feat_row = feat_by_ds_vid[ds].get(vid)
        master_row = master_by_ds_vid[ds].get(vid, {})
        if not feat_row:
            extraction_failures.append(f"{ds}/{vid}: missing feature row")
            continue

        row = dict(feat_row)
        for k in ("video_path", "split", "binary_label", "category_label", "fps"):
            if k in master_row and master_row[k] not in ("", None):
                row[k] = master_row[k]

        pred = infer_and_postprocess_video(
            model=model,
            model_kind=args.model_kind,
            row=row,
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

        feat_path = resolve(root, Path(row["feature_path"]))
        with np.load(feat_path, allow_pickle=True) as d:
            feats = d["features"].astype(np.float32)

        seg_scores_raw, _, _ = infer_full_sequence_chunked_variant(
            model=model,
            model_kind=args.model_kind,
            features=feats,
            num_classes=len(class_names),
            window=args.infer_window,
            stride=args.infer_stride,
            device=device,
        )
        seg_scores = moving_average(seg_scores_raw, int(args.smooth_window)).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            _, _, _, attn_maps = forward_with_attention(model, args.model_kind, x)

        if not attn_maps:
            extraction_failures.append(f"{ds}/{vid}: attention not returned")
            continue

        n_layers = len(attn_maps)
        layer_idx = args.attention_layer if args.attention_layer >= 0 else (n_layers + args.attention_layer)
        layer_idx = max(0, min(layer_idx, n_layers - 1))
        attn = attn_maps[layer_idx][0].detach().cpu().numpy().astype(np.float32)  # [H, T, T]

        if attn.ndim != 3:
            extraction_failures.append(f"{ds}/{vid}: unexpected attention shape {tuple(attn.shape)}")
            continue

        head_focus = attn.mean(axis=1)  # [H, T]
        n_heads, t_len = int(head_focus.shape[0]), int(head_focus.shape[1])
        layer_values.append(layer_idx)
        head_values.append(n_heads)

        pred_spans = [(int(s), int(e)) for s, e in pred.get("spans_after_refine", [])]
        gt_spans = find_gt_spans(
            dataset=ds,
            row=row,
            master_row=master_row,
            project_root=root,
            ucf_temporal_root=args.ucf_temporal_root,
            num_segments=int(pred.get("num_segments", feats.shape[0])),
        )

        k = max(1, int(math.ceil(t_len * float(args.topk_ratio))))
        peak_idx = topk_indices(seg_scores, k=k)
        pred_overlap = overlap_ratio(peak_idx, pred_spans)
        gt_overlap = overlap_ratio(peak_idx, gt_spans) if gt_spans else None
        entropy_per_head = [normalized_entropy(head_focus[h]) for h in range(n_heads)]
        entropy_mean = float(np.mean(entropy_per_head)) if entropy_per_head else 1.0

        cgroup = str(case.get("case_group", "")).lower()
        bstatus = str(case.get("binary_status", ""))
        obs = case_observation(
            dataset=ds,
            case_group=cgroup,
            status=bstatus,
            entropy_mean=entropy_mean,
            pred_overlap=pred_overlap,
            gt_overlap=gt_overlap,
            score=float(pred.get("video_anomaly_score", 0.0)),
        )

        base = f"{idx:02d}_{ds}_{safe_name(vid)}"
        fig_path = plots_dir / f"{base}.png"
        tensor_path = tensors_dir / f"{base}.npz"
        detail_path = details_dir / f"{base}.json"

        plot_case(
            case=case,
            seg_scores=seg_scores,
            pred_spans=pred_spans,
            gt_spans=gt_spans,
            head_focus=head_focus,
            layer_idx=layer_idx,
            threshold=float(args.threshold),
            out_path=fig_path,
            dpi=int(args.dpi),
        )

        np.savez_compressed(
            tensor_path,
            attention=attn,
            head_focus=head_focus.astype(np.float32),
            anomaly_scores=seg_scores.astype(np.float32),
            pred_spans=np.array(pred_spans, dtype=np.int32) if pred_spans else np.zeros((0, 2), dtype=np.int32),
            gt_spans=np.array(gt_spans, dtype=np.int32) if gt_spans else np.zeros((0, 2), dtype=np.int32),
        )

        detail = {
            "video_id": vid,
            "dataset": ds,
            "split": row.get("split", ""),
            "gt_label": row.get("category_label", case.get("gt_label", "")),
            "predicted_label": pred.get("pred_video_class", case.get("predicted_label", "")),
            "case_group": cgroup,
            "binary_status": bstatus,
            "anomaly_score": float(pred.get("video_anomaly_score", 0.0)),
            "num_segments": int(pred.get("num_segments", feats.shape[0])),
            "layer_used": int(layer_idx),
            "num_heads": int(n_heads),
            "attention_shape": [int(x) for x in attn.shape],
            "entropy_per_head": [float(x) for x in entropy_per_head],
            "entropy_mean": float(entropy_mean),
            "topk_segment_count": int(k),
            "peak_segment_indices": peak_idx,
            "pred_spans": [[int(s), int(e)] for s, e in pred_spans],
            "gt_spans": [[int(s), int(e)] for s, e in gt_spans],
            "topk_overlap_with_pred": float(pred_overlap),
            "topk_overlap_with_gt": None if gt_overlap is None else float(gt_overlap),
            "observation": obs,
            "artifacts": {
                "plot": str(fig_path),
                "attention_tensor": str(tensor_path),
            },
            "sequence": {
                "anomaly_scores": [float(x) for x in seg_scores.tolist()],
                "predicted_event_segments": [[int(s), int(e)] for s, e in pred_spans],
                "gt_event_segments": [[int(s), int(e)] for s, e in gt_spans],
            },
        }
        detail_path.write_text(json.dumps(detail, indent=2) + "\n")

        summary_rows.append(
            {
                "video_id": vid,
                "dataset": ds,
                "case_group": cgroup,
                "binary_status": bstatus,
                "gt_label": detail["gt_label"],
                "predicted_label": detail["predicted_label"],
                "anomaly_score": detail["anomaly_score"],
                "layer_used": detail["layer_used"],
                "num_heads": detail["num_heads"],
                "entropy_mean": detail["entropy_mean"],
                "topk_overlap_with_pred": detail["topk_overlap_with_pred"],
                "topk_overlap_with_gt": detail["topk_overlap_with_gt"],
                "observation": obs,
                "plot_path": str(fig_path),
                "detail_json": str(detail_path),
                "attention_tensor": str(tensor_path),
            }
        )
        print(f"  [{idx}/{len(cases)}] done {ds}/{vid}")

    # Cross-dataset takeaway on selected 10 cases.
    by_ds: Dict[str, List[Dict[str, object]]] = {}
    for r in summary_rows:
        by_ds.setdefault(str(r["dataset"]), []).append(r)

    def mean_of(ds: str, key: str) -> float:
        vals = [as_float(x.get(key, 0.0), 0.0) for x in by_ds.get(ds, [])]
        return float(np.mean(vals)) if vals else float("nan")

    def frac_fail(ds: str) -> float:
        rows = by_ds.get(ds, [])
        if not rows:
            return float("nan")
        n = sum(1 for r in rows if str(r.get("case_group", "")) == "failure")
        return float(n) / float(len(rows))

    takeaway = {
        "ucf_attention": (
            f"UCF attention is {'sharper' if mean_of('ucf_crime', 'entropy_mean') < 0.7 else 'more diffuse'} "
            f"on selected cases (mean entropy={mean_of('ucf_crime', 'entropy_mean'):.3f})."
            if by_ds.get("ucf_crime")
            else "UCF cases not present."
        ),
        "xd_attention": (
            f"XD errors are {'weak-focus' if mean_of('xd_violence', 'topk_overlap_with_pred') < 0.25 else 'wrong-focus'} "
            f"dominant on selected cases (mean pred-overlap={mean_of('xd_violence', 'topk_overlap_with_pred'):.3f}, "
            f"failure_frac={frac_fail('xd_violence'):.2f})."
            if by_ds.get("xd_violence")
            else "XD cases not present."
        ),
        "rwf_attention": (
            f"RWF failures show {'under-attention to fight motion' if frac_fail('rwf_2000') >= 0.5 else 'mixed behavior'} "
            f"(failure_frac={frac_fail('rwf_2000'):.2f}, mean pred-overlap={mean_of('rwf_2000', 'topk_overlap_with_pred'):.3f})."
            if by_ds.get("rwf_2000")
            else "RWF cases not present."
        ),
        "sh_attention": (
            f"ShanghaiTech collapse appears as domain mismatch in attention "
            f"(failure_frac={frac_fail('shanghaitech'):.2f}, mean entropy={mean_of('shanghaitech', 'entropy_mean'):.3f})."
            if by_ds.get("shanghaitech")
            else "ShanghaiTech cases not present."
        ),
    }

    attention_summary = {
        "videos_visualized": len(summary_rows),
        "requested_cases": len(cases),
        "extracted_all_requested": len(summary_rows) == len(cases),
        "num_heads_unique": sorted(set(head_values)),
        "layer_indices_used": sorted(set(layer_values)),
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "attention_extraction_failures": extraction_failures,
    }

    report_lines: List[str] = []
    report_lines.append("1) Attention extraction summary")
    report_lines.append(f"- videos visualized: {attention_summary['videos_visualized']}")
    report_lines.append(f"- requested cases: {attention_summary['requested_cases']}")
    report_lines.append(f"- extraction success for all requested: {attention_summary['extracted_all_requested']}")
    report_lines.append(f"- model/checkpoint: {args.model_kind} | {args.checkpoint}")
    report_lines.append(f"- number of heads (unique): {attention_summary['num_heads_unique']}")
    report_lines.append(f"- layer used (unique): {attention_summary['layer_indices_used']}")
    report_lines.append(f"- device: {device}")
    if extraction_failures:
        report_lines.append(f"- extraction failures: {len(extraction_failures)}")
        for e in extraction_failures:
            report_lines.append(f"  - {e}")

    report_lines.append("")
    report_lines.append("2) Per-case observations")
    for r in summary_rows:
        report_lines.append(
            f"- {r['video_id']} | dataset={r['dataset']} | {r['case_group']} | "
            f"gt={r['gt_label']} | pred={r['predicted_label']} | anomaly_score={as_float(r['anomaly_score']):.6f}"
        )
        report_lines.append(f"  observation: {r['observation']}")

    report_lines.append("")
    report_lines.append("3) Cross-dataset takeaway")
    report_lines.append(f"- {takeaway['ucf_attention']}")
    report_lines.append(f"- {takeaway['xd_attention']}")
    report_lines.append(f"- {takeaway['rwf_attention']}")
    report_lines.append(f"- {takeaway['sh_attention']}")

    report_lines.append("")
    report_lines.append("4) Artifact list")
    report_lines.append(f"- plots_dir: {plots_dir}")
    report_lines.append(f"- tensors_dir: {tensors_dir}")
    report_lines.append(f"- details_dir: {details_dir}")
    report_lines.append(f"- summary_json: {out_dir / 'step14b_summary.json'}")
    report_lines.append(f"- per_case_json: {out_dir / 'step14b_per_case.json'}")

    (out_dir / "step14b_report.txt").write_text("\n".join(report_lines) + "\n")

    payload = {
        "attention_extraction_summary": attention_summary,
        "per_case": summary_rows,
        "cross_dataset_takeaway": takeaway,
        "artifacts": {
            "plots_dir": str(plots_dir),
            "tensors_dir": str(tensors_dir),
            "details_dir": str(details_dir),
            "report_txt": str(out_dir / "step14b_report.txt"),
            "summary_json": str(out_dir / "step14b_summary.json"),
            "per_case_json": str(out_dir / "step14b_per_case.json"),
        },
    }
    (out_dir / "step14b_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "step14b_per_case.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    print("Step-14B complete")
    print(f"- cases visualized: {len(summary_rows)}/{len(cases)}")
    print(f"- report: {out_dir / 'step14b_report.txt'}")
    print(f"- plots: {plots_dir}")
    print(f"- attention tensors: {tensors_dir}")


if __name__ == "__main__":
    main()
