#!/usr/bin/env python3
"""Evaluate a trained checkpoint with fixed full-sequence decoding settings (Step-10 support)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from train_rtfm_classifier import RTFMClassifier
from train_rtfm_progressive import ProgressiveRTFM
from train_rtfm_trn import RTFMTRNClassifier
from train_rtfm_trn_boundary import (
    RTFMTRNBoundary,
    choose_device,
    evaluate_localization_map,
    load_gt_events_for_test,
    moving_average,
    parse_class_names,
    parse_float_list,
    read_csv,
    refine_spans_with_boundary,
    resolve,
    safe_ap,
    safe_auc,
    set_seed,
    spans_from_scores,
    spans_to_events,
    topk_mean_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ablation checkpoint with fixed decoding settings")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--master-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    p.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("data/ucf_crime/annotations/temporal_segments"),
    )
    p.add_argument(
        "--model-kind",
        choices=["step5_classifier", "step6_trn", "step7_boundary", "step8_progressive"],
        required=True,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=Path(""))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")

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

    p.add_argument("--localization-tiou", type=str, default="0.3,0.5,0.7")
    p.add_argument(
        "--class-names",
        type=str,
        default="normal,fighting,shooting,explosion,robbery,abuse",
    )
    p.add_argument("--normal-class", type=str, default="normal")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def sliding_window_starts(total_len: int, window: int, stride: int) -> List[int]:
    if total_len <= window:
        return [0]
    starts = list(range(0, total_len - window + 1, stride))
    last = total_len - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def load_model(args: argparse.Namespace, num_classes: int, device: str) -> Tuple[torch.nn.Module, bool]:
    if args.model_kind == "step5_classifier":
        model = RTFMClassifier(
            input_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
        has_boundary = False
    elif args.model_kind == "step6_trn":
        model = RTFMTRNClassifier(
            input_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            target_segments=args.target_segments,
            trn_layers=args.trn_layers,
            trn_heads=args.trn_heads,
            trn_ffn_mult=args.trn_ffn_mult,
            trn_dropout=args.trn_dropout,
            proj_dropout=args.dropout,
            pos_encoding=args.pos_encoding,
        )
        has_boundary = False
    elif args.model_kind == "step7_boundary":
        model = RTFMTRNBoundary(
            input_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            target_segments=args.target_segments,
            trn_layers=args.trn_layers,
            trn_heads=args.trn_heads,
            trn_ffn_mult=args.trn_ffn_mult,
            trn_dropout=args.trn_dropout,
            proj_dropout=args.dropout,
            pos_encoding=args.pos_encoding,
        )
        has_boundary = True
    else:
        model = ProgressiveRTFM(
            input_dim=2048,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            target_segments=args.target_segments,
            trn_layers=args.trn_layers,
            trn_heads=args.trn_heads,
            trn_ffn_mult=args.trn_ffn_mult,
            trn_dropout=args.trn_dropout,
            proj_dropout=args.dropout,
            pos_encoding=args.pos_encoding,
        )
        has_boundary = True

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    if missing:
        print(f"Warning: missing keys when loading checkpoint ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:10]}")

    return model, has_boundary


def forward_variant(
    model: torch.nn.Module,
    model_kind: str,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if model_kind == "step5_classifier":
        seg_anom_logits, seg_class_logits, _, _ = model(x)
        return seg_anom_logits, seg_class_logits, None

    if model_kind == "step6_trn":
        seg_anom_logits, seg_class_logits, _, _, _ = model(x, return_attention=False)
        return seg_anom_logits, seg_class_logits, None

    if model_kind == "step7_boundary":
        seg_anom_logits, seg_class_logits, _, bnd_scores, _, _ = model(x, return_attention=False)
        return seg_anom_logits, seg_class_logits, bnd_scores

    seg_anom_logits, seg_class_logits, _, bnd_scores, _, _ = model(
        x,
        return_attention=False,
        apply_trn=True,
        apply_boundary=True,
    )
    return seg_anom_logits, seg_class_logits, bnd_scores


def infer_full_sequence_chunked_variant(
    model: torch.nn.Module,
    model_kind: str,
    features: np.ndarray,
    num_classes: int,
    window: int,
    stride: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = int(features.shape[0])
    if t <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, num_classes), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    sum_anom = np.zeros((t,), dtype=np.float64)
    cnt_anom = np.zeros((t,), dtype=np.float64)
    sum_cls = np.zeros((t, num_classes), dtype=np.float64)
    cnt_cls = np.zeros((t,), dtype=np.float64)

    if t > 1:
        sum_bnd = np.zeros((t - 1,), dtype=np.float64)
        cnt_bnd = np.zeros((t - 1,), dtype=np.float64)
    else:
        sum_bnd = np.zeros((0,), dtype=np.float64)
        cnt_bnd = np.zeros((0,), dtype=np.float64)

    starts = sliding_window_starts(t, window, stride)

    with torch.no_grad():
        for s in starts:
            e = min(s + window, t)
            chunk = features[s:e]
            real_len = int(e - s)

            if real_len < window:
                pad = np.repeat(chunk[-1:, :], repeats=(window - real_len), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)

            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            seg_logits, seg_cls_logits, bnd_scores = forward_variant(model, model_kind, x)

            seg_scores = torch.sigmoid(seg_logits)[0].detach().cpu().numpy().astype(np.float64)[:real_len]
            cls_probs = torch.softmax(seg_cls_logits, dim=-1)[0].detach().cpu().numpy().astype(np.float64)[:real_len]

            sum_anom[s:e] += seg_scores
            cnt_anom[s:e] += 1.0
            sum_cls[s:e, :] += cls_probs
            cnt_cls[s:e] += 1.0

            if bnd_scores is not None and real_len >= 2 and bnd_scores.shape[1] > 0:
                bnd = bnd_scores[0].detach().cpu().numpy().astype(np.float64)[: real_len - 1]
                sum_bnd[s : e - 1] += bnd
                cnt_bnd[s : e - 1] += 1.0

    anom = np.divide(sum_anom, np.maximum(cnt_anom, 1e-8), out=np.zeros_like(sum_anom), where=cnt_anom > 0)
    cls = np.divide(
        sum_cls,
        np.maximum(cnt_cls[:, None], 1e-8),
        out=np.zeros_like(sum_cls),
        where=cnt_cls[:, None] > 0,
    )

    if t > 1:
        bnd = np.divide(sum_bnd, np.maximum(cnt_bnd, 1e-8), out=np.zeros_like(sum_bnd), where=cnt_bnd > 0)
    else:
        bnd = np.zeros((0,), dtype=np.float64)

    return anom.astype(np.float32), cls.astype(np.float32), bnd.astype(np.float32)


def top_boundary_peaks(bnd_scores: np.ndarray, k: int = 5) -> List[Dict[str, object]]:
    if bnd_scores.size == 0:
        return []
    kk = int(min(k, bnd_scores.shape[0]))
    idx = np.argpartition(bnd_scores, -kk)[-kk:]
    idx = idx[np.argsort(-bnd_scores[idx])]
    return [{"edge_index": int(i), "value": float(bnd_scores[i])} for i in idx.tolist()]


def evaluate_video_level(
    results: Sequence[Dict[str, object]],
    class_names: List[str],
    class_to_idx: Dict[str, int],
    normal_idx: int,
    threshold: float,
) -> Dict[str, object]:
    y_true_bin = np.array([int(r["binary_label"]) for r in results], dtype=np.int64)
    y_score_bin = np.array([float(r["video_anomaly_score"]) for r in results], dtype=np.float64)
    y_pred_bin = (y_score_bin >= threshold).astype(np.int64)

    binary_auc = safe_auc(y_true_bin, y_score_bin)
    binary_ap = safe_ap(y_true_bin, y_score_bin)
    if y_true_bin.size > 0:
        binary_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist()
        binary_f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    else:
        binary_cm = [[0, 0], [0, 0]]
        binary_f1 = float("nan")

    y_true_cls = np.array([class_to_idx.get(str(r["category_label"]), normal_idx) for r in results], dtype=np.int64)
    y_pred_cls = np.array([class_to_idx.get(str(r["pred_video_class"]), normal_idx) for r in results], dtype=np.int64)

    labels = list(range(len(class_names)))
    cls_cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_cls,
        y_pred_cls,
        labels=labels,
        zero_division=0,
    )
    per_class = [
        {
            "class_name": class_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    ]
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_cls, y_pred_cls, labels=labels, average="weighted", zero_division=0))

    return {
        "binary": {
            "auc": binary_auc,
            "ap": binary_ap,
            "f1": binary_f1,
            "confusion_matrix_threshold": threshold,
            "confusion_matrix": binary_cm,
        },
        "classification": {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cls_cm,
            "class_names": class_names,
            "per_class": per_class,
        },
    }


def build_pred_events(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    pred_events_all: List[Dict[str, object]] = []
    for r in results:
        for ev in r.get("events_after_refine", []):
            pred_events_all.append(
                {
                    "video_id": r["video_id"],
                    "predicted_class": ev["predicted_class"],
                    "start_segment": int(ev["start_segment"]),
                    "end_segment": int(ev["end_segment"]),
                    "event_score": float(ev["event_score"]),
                }
            )
    return pred_events_all


def infer_and_postprocess_video(
    model: torch.nn.Module,
    model_kind: str,
    row: Dict[str, str],
    project_root: Path,
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    infer_window: int,
    infer_stride: int,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
    boundary_refine: bool,
    device: str,
) -> Dict[str, object]:
    feat_path = resolve(project_root, Path(row["feature_path"]))
    with np.load(feat_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        starts = data["segment_start_frames"].astype(np.int64)
        ends = data["segment_end_frames"].astype(np.int64)

    seg_scores, cls_probs, bnd_scores = infer_full_sequence_chunked_variant(
        model=model,
        model_kind=model_kind,
        features=feats,
        num_classes=len(class_names),
        window=infer_window,
        stride=infer_stride,
        device=device,
    )
    seg_scores_smooth = moving_average(seg_scores, smooth_window)
    # Global class distribution summary from full sequence (used by Step-11 overlap eval).
    if cls_probs.shape[0] > 0:
        w = np.maximum(seg_scores_smooth.astype(np.float64), 1e-6)
        class_prob_mean = (cls_probs.astype(np.float64) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-6)
    else:
        class_prob_mean = np.zeros((len(class_names),), dtype=np.float64)

    spans_before = spans_from_scores(seg_scores_smooth, threshold=threshold, min_len=min_event_len, merge_gap=merge_gap)

    can_refine = boundary_refine and bnd_scores.size > 0
    if can_refine:
        spans_after = refine_spans_with_boundary(
            spans=spans_before,
            boundary_scores=bnd_scores,
            total_segments=int(seg_scores_smooth.shape[0]),
            radius=boundary_radius,
            min_len=min_event_len,
            merge_gap=merge_gap,
        )
    else:
        spans_after = list(spans_before)

    fps = float(row["fps"])
    events_after = spans_to_events(
        spans=spans_after,
        scores=seg_scores_smooth,
        class_probs=cls_probs,
        segment_starts=starts,
        segment_ends=ends,
        fps=fps,
        class_names=class_names,
        normal_idx=normal_idx,
    )

    video_anomaly_score = topk_mean_np(seg_scores_smooth, topk_ratio)
    if video_anomaly_score < threshold or not events_after:
        pred_class_idx = normal_idx
        pred_class = class_names[pred_class_idx]
        pred_class_prob = 1.0 - video_anomaly_score
    else:
        best_event = events_after[0]
        pred_class = str(best_event["predicted_class"])
        pred_class_idx = int(best_event["predicted_class_idx"])
        pred_class_prob = float(best_event["class_confidence"])

    return {
        "video_id": row["video_id"],
        "split": row["split"],
        "binary_label": int(row["binary_label"]),
        "category_label": str(row["category_label"]),
        "num_segments": int(feats.shape[0]),
        "video_anomaly_score": float(video_anomaly_score),
        "pred_video_class": pred_class,
        "pred_video_class_idx": int(pred_class_idx),
        "pred_video_class_prob": float(pred_class_prob),
        "video_class_prob_vector": [float(x) for x in class_prob_mean.tolist()],
        "spans_before_refine": [[int(s), int(e)] for (s, e) in spans_before],
        "spans_after_refine": [[int(s), int(e)] for (s, e) in spans_after],
        "events_after_refine": events_after,
        "top_boundary_peaks": top_boundary_peaks(bnd_scores),
        "boundary_refine_applied": bool(can_refine),
    }


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    feature_manifest = resolve(project_root, args.feature_manifest)
    master_manifest = resolve(project_root, args.master_manifest)
    temporal_root = resolve(project_root, args.temporal_root)
    checkpoint = resolve(project_root, args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    tiou_thresholds = parse_float_list(args.localization_tiou)

    set_seed(args.seed)
    device = choose_device(args.device)

    model, has_boundary = load_model(args, num_classes=len(class_names), device=device)

    feat_rows = [r for r in read_csv(feature_manifest) if r.get("status") == "ok"]
    split_rows = [r for r in feat_rows if r.get("split") == args.split]

    master_rows = read_csv(master_manifest)
    master_by_video = {r["video_id"]: r for r in master_rows}

    video_results: List[Dict[str, object]] = []
    total = len(split_rows)
    for i, r in enumerate(split_rows, 1):
        vr = infer_and_postprocess_video(
            model=model,
            model_kind=args.model_kind,
            row=r,
            project_root=project_root,
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
        video_results.append(vr)
        if i % 25 == 0 or i == total:
            print(f"  inference {i}/{total}")

    metrics = evaluate_video_level(
        results=video_results,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        threshold=args.threshold,
    )

    gt_events = load_gt_events_for_test(
        test_rows=split_rows,
        master_rows_by_video=master_by_video,
        temporal_root=temporal_root,
    )
    pred_events = build_pred_events(video_results)

    localization = evaluate_localization_map(
        pred_events_all=pred_events,
        gt_events_all=gt_events,
        class_names=class_names,
        normal_idx=normal_idx,
        tiou_thresholds=tiou_thresholds,
    )

    tiou = localization.get("tiou", {}) if isinstance(localization, dict) else {}
    map03 = tiou.get("0.3", {}).get("mAP") if isinstance(tiou, dict) else float("nan")
    map05 = tiou.get("0.5", {}).get("mAP") if isinstance(tiou, dict) else float("nan")
    map07 = tiou.get("0.7", {}).get("mAP") if isinstance(tiou, dict) else float("nan")

    summary = {
        "model_kind": args.model_kind,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "decode_settings": {
            "threshold": args.threshold,
            "smooth_window": args.smooth_window,
            "min_event_len": args.min_event_len,
            "merge_gap": args.merge_gap,
            "boundary_radius": args.boundary_radius,
            "boundary_refine_requested": bool(args.boundary_refine),
            "model_has_boundary_head": bool(has_boundary),
        },
        "metrics": metrics,
        "temporal_localization": localization,
        "mAP@0.3": map03,
        "mAP@0.5": map05,
        "mAP@0.7": map07,
        "pred_event_count": int(localization.get("pred_event_count", 0)),
        "gt_event_count": int(localization.get("gt_event_count", 0)),
    }

    if args.output_json:
        out_path = resolve(project_root, args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved summary: {out_path}")

    print("\nEvaluation summary")
    print(f"- model_kind: {args.model_kind}")
    print(f"- checkpoint: {checkpoint}")
    print(f"- AUC/AP: {metrics['binary']['auc']:.6f} / {metrics['binary']['ap']:.6f}")
    print(
        f"- Macro/Weighted F1: {metrics['classification']['macro_f1']:.6f} / "
        f"{metrics['classification']['weighted_f1']:.6f}"
    )
    m03 = float(map03) if map03 is not None and not math.isnan(float(map03)) else float("nan")
    m05 = float(map05) if map05 is not None and not math.isnan(float(map05)) else float("nan")
    m07 = float(map07) if map07 is not None and not math.isnan(float(map07)) else float("nan")
    print(f"- mAP@0.3/0.5/0.7: {m03:.6f} / {m05:.6f} / {m07:.6f}")
    print(f"- Pred vs GT events: {summary['pred_event_count']} vs {summary['gt_event_count']}")


if __name__ == "__main__":
    main()
