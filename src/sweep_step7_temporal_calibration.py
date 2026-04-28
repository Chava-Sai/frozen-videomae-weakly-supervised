#!/usr/bin/env python3
"""Step-9: Temporal inference calibration sweep on Step-7 checkpoint (no model re-training)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from train_rtfm_trn_boundary import (
    RTFMTRNBoundary,
    best_iou_against_gt,
    choose_device,
    choose_failure_case,
    evaluate_localization_map,
    infer_full_sequence_chunked,
    load_gt_events_for_test,
    metric_key,
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
    p = argparse.ArgumentParser(description="Step-9 sweep on Step-7 checkpoint")
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
        "--checkpoint",
        type=Path,
        default=Path("outputs/rtfm_trn_boundary/checkpoints/best.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step9_step7_calibration"),
    )

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

    p.add_argument("--threshold-grid", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--smooth-window-grid", type=str, default="1,3,5,7")
    p.add_argument("--min-event-len-grid", type=str, default="1,2,3,4,5")
    p.add_argument("--merge-gap-grid", type=str, default="0,1,2,3,4")
    p.add_argument("--boundary-radius-grid", type=str, default="0,1,2,3,4")
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


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for x in raw.split(","):
        s = x.strip()
        if s:
            vals.append(int(s))
    if not vals:
        raise ValueError(f"No integer values parsed from: {raw}")
    return vals


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def load_model(model_path: Path, device: str, num_classes: int, args: argparse.Namespace) -> RTFMTRNBoundary:
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
    ).to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_video_raw(
    model: RTFMTRNBoundary,
    row: Dict[str, str],
    project_root: Path,
    device: str,
    num_classes: int,
    infer_window: int,
    infer_stride: int,
) -> Dict[str, object]:
    feat_path = resolve(project_root, Path(row["feature_path"]))
    with np.load(feat_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        starts = data["segment_start_frames"].astype(np.int64)
        ends = data["segment_end_frames"].astype(np.int64)

    seg_scores, cls_probs, bnd_scores = infer_full_sequence_chunked(
        model=model,
        features=feats,
        device=device,
        num_classes=num_classes,
        window=infer_window,
        stride=infer_stride,
    )

    return {
        "video_id": row["video_id"],
        "split": row["split"],
        "binary_label": int(row["binary_label"]),
        "category_label": str(row["category_label"]),
        "fps": float(row["fps"]),
        "duration_sec": float(row["duration_sec"]),
        "num_segments": int(feats.shape[0]),
        "segment_start_frames": starts,
        "segment_end_frames": ends,
        "segment_scores_raw": seg_scores,
        "segment_class_probs_raw": cls_probs,
        "boundary_scores_raw": bnd_scores,
    }


def postprocess_raw_video(
    raw: Dict[str, object],
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
    use_boundary_refine: bool,
) -> Dict[str, object]:
    seg_scores_raw = np.asarray(raw["segment_scores_raw"], dtype=np.float32)
    cls_probs_raw = np.asarray(raw["segment_class_probs_raw"], dtype=np.float32)
    bnd_scores_raw = np.asarray(raw["boundary_scores_raw"], dtype=np.float32)
    starts = np.asarray(raw["segment_start_frames"], dtype=np.int64)
    ends = np.asarray(raw["segment_end_frames"], dtype=np.int64)

    seg_scores_smooth = moving_average(seg_scores_raw, smooth_window)

    spans_before = spans_from_scores(seg_scores_smooth, threshold=threshold, min_len=min_event_len, merge_gap=merge_gap)
    if use_boundary_refine:
        spans_after = refine_spans_with_boundary(
            spans=spans_before,
            boundary_scores=bnd_scores_raw,
            total_segments=int(seg_scores_smooth.shape[0]),
            radius=boundary_radius,
            min_len=min_event_len,
            merge_gap=merge_gap,
        )
    else:
        spans_after = list(spans_before)

    fps = float(raw["fps"])
    events_before = spans_to_events(
        spans=spans_before,
        scores=seg_scores_smooth,
        class_probs=cls_probs_raw,
        segment_starts=starts,
        segment_ends=ends,
        fps=fps,
        class_names=class_names,
        normal_idx=normal_idx,
    )
    events_after = spans_to_events(
        spans=spans_after,
        scores=seg_scores_smooth,
        class_probs=cls_probs_raw,
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

    topk_bnd = []
    if bnd_scores_raw.size > 0:
        k = int(min(5, bnd_scores_raw.shape[0]))
        idx = np.argpartition(bnd_scores_raw, -k)[-k:]
        idx = idx[np.argsort(-bnd_scores_raw[idx])]
        topk_bnd = [{"edge_index": int(i), "value": float(bnd_scores_raw[i])} for i in idx.tolist()]

    return {
        "video_id": raw["video_id"],
        "split": raw["split"],
        "binary_label": int(raw["binary_label"]),
        "category_label": raw["category_label"],
        "num_segments": int(raw["num_segments"]),
        "fps": float(raw["fps"]),
        "duration_sec": float(raw["duration_sec"]),
        "video_anomaly_score": float(video_anomaly_score),
        "pred_video_class": pred_class,
        "pred_video_class_idx": pred_class_idx,
        "pred_video_class_prob": float(pred_class_prob),
        "spans_before_refine": [[int(s), int(e)] for s, e in spans_before],
        "spans_after_refine": [[int(s), int(e)] for s, e in spans_after],
        "events_before_refine": events_before,
        "events_after_refine": events_after,
        "top_boundary_peaks": topk_bnd,
        "use_boundary_refine": bool(use_boundary_refine),
    }


def process_raw_list(
    raw_list: Sequence[Dict[str, object]],
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
    use_boundary_refine: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for raw in raw_list:
        out.append(
            postprocess_raw_video(
                raw=raw,
                class_names=class_names,
                normal_idx=normal_idx,
                topk_ratio=topk_ratio,
                threshold=threshold,
                smooth_window=smooth_window,
                min_event_len=min_event_len,
                merge_gap=merge_gap,
                boundary_radius=boundary_radius,
                use_boundary_refine=use_boundary_refine,
            )
        )
    return out


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
    binary_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist() if y_true_bin.size > 0 else [[0, 0], [0, 0]]
    binary_f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0)) if y_true_bin.size > 0 else float("nan")

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


def evaluate_setting_on_split(
    raw_list: Sequence[Dict[str, object]],
    gt_events: Sequence[Dict[str, object]],
    class_names: List[str],
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
    use_boundary_refine: bool,
    tiou_thresholds: List[float],
) -> Dict[str, object]:
    results = process_raw_list(
        raw_list=raw_list,
        class_names=class_names,
        normal_idx=normal_idx,
        topk_ratio=topk_ratio,
        threshold=threshold,
        smooth_window=smooth_window,
        min_event_len=min_event_len,
        merge_gap=merge_gap,
        boundary_radius=boundary_radius,
        use_boundary_refine=use_boundary_refine,
    )
    metrics = evaluate_video_level(
        results=results,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        threshold=threshold,
    )

    pred_events = build_pred_events(results)
    localization = evaluate_localization_map(
        pred_events_all=pred_events,
        gt_events_all=list(gt_events),
        class_names=class_names,
        normal_idx=normal_idx,
        tiou_thresholds=tiou_thresholds,
    )

    return {
        "results": results,
        "metrics": metrics,
        "localization": localization,
        "pred_events": pred_events,
    }


def setting_rank_key(record: Dict[str, object], has_val_temporal: bool) -> Tuple[float, float, float]:
    if has_val_temporal:
        v05 = record.get("val_map_05")
        v03 = record.get("val_map_03")
        gt = record.get("val_gt_events", 0)
        pred = record.get("val_pred_events", 0)
        m05 = metric_key(float(v05) if v05 is not None else float("nan"))
        m03 = metric_key(float(v03) if v03 is not None else float("nan"))
        closeness = -abs(float(pred) - float(gt))
        return (m05, m03, closeness)

    # fallback when val temporal GT is unavailable
    bf1 = metric_key(float(record.get("val_binary_f1", float("nan"))))
    # encourage fewer fragments per positive video
    avg_evt = float(record.get("val_avg_events_per_positive_video", 0.0))
    fragment_penalty = -max(0.0, avg_evt - 1.0)
    return (bf1, fragment_penalty, -float(record.get("val_pred_events", 0)))


def select_examples(
    chosen_results: Sequence[Dict[str, object]],
    baseline_results: Sequence[Dict[str, object]],
    gt_by_video: Dict[str, List[Dict[str, object]]],
    threshold: float,
) -> Dict[str, object]:
    base_by_vid = {str(r["video_id"]): r for r in baseline_results}

    improved = []
    partial = []

    for r in chosen_results:
        vid = str(r["video_id"])
        if int(r["binary_label"]) == 0 or vid not in base_by_vid:
            continue
        gt = gt_by_video.get(vid, [])
        if not gt:
            continue

        spans_base = [tuple(x) for x in base_by_vid[vid].get("spans_after_refine", [])]
        spans_new = [tuple(x) for x in r.get("spans_after_refine", [])]
        iou_base = best_iou_against_gt(spans_base, gt)
        iou_new = best_iou_against_gt(spans_new, gt)
        gain = iou_new - iou_base

        rec = {
            "video_id": vid,
            "ground_truth_class": r.get("category_label"),
            "gt_spans": [[int(g["start_segment"]), int(g["end_segment"])] for g in gt],
            "baseline_spans": [[int(a), int(b)] for (a, b) in spans_base],
            "chosen_spans": [[int(a), int(b)] for (a, b) in spans_new],
            "baseline_best_iou": float(iou_base),
            "chosen_best_iou": float(iou_new),
            "improvement": float(gain),
        }
        if gain >= 0.20:
            improved.append(rec)
        elif gain > 1e-6:
            partial.append(rec)

    improved.sort(key=lambda x: x["improvement"], reverse=True)
    partial.sort(key=lambda x: x["improvement"], reverse=True)

    failure = choose_failure_case(list(chosen_results), gt_by_video, threshold)

    return {
        "strong_success": improved[0] if improved else (partial[0] if partial else None),
        "partial_improvement": partial[0] if partial else (improved[1] if len(improved) > 1 else None),
        "failure": failure,
    }


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    feature_manifest = resolve(project_root, args.feature_manifest)
    master_manifest = resolve(project_root, args.master_manifest)
    temporal_root = resolve(project_root, args.temporal_root)
    checkpoint = resolve(project_root, args.checkpoint)
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    thresholds = parse_float_list(args.threshold_grid)
    smooth_windows = parse_int_list(args.smooth_window_grid)
    min_event_lens = parse_int_list(args.min_event_len_grid)
    merge_gaps = parse_int_list(args.merge_gap_grid)
    boundary_radii = parse_int_list(args.boundary_radius_grid)
    tiou_thresholds = parse_float_list(args.localization_tiou)

    set_seed(args.seed)
    device = choose_device(args.device)

    model = load_model(checkpoint, device=device, num_classes=len(class_names), args=args)

    feat_rows = [r for r in read_csv(feature_manifest) if r.get("status") == "ok"]
    val_rows = [r for r in feat_rows if r.get("split") == "val"]
    test_rows = [r for r in feat_rows if r.get("split") == "test"]

    master_rows = read_csv(master_manifest)
    master_by_video = {r["video_id"]: r for r in master_rows}

    print("Caching full-sequence inference for val...")
    val_raw: List[Dict[str, object]] = []
    for i, r in enumerate(val_rows, 1):
        val_raw.append(
            infer_video_raw(
                model=model,
                row=r,
                project_root=project_root,
                device=device,
                num_classes=len(class_names),
                infer_window=args.infer_window,
                infer_stride=args.infer_stride,
            )
        )
        if i % 20 == 0 or i == len(val_rows):
            print(f"  val raw {i}/{len(val_rows)}")

    print("Caching full-sequence inference for test...")
    test_raw: List[Dict[str, object]] = []
    for i, r in enumerate(test_rows, 1):
        test_raw.append(
            infer_video_raw(
                model=model,
                row=r,
                project_root=project_root,
                device=device,
                num_classes=len(class_names),
                infer_window=args.infer_window,
                infer_stride=args.infer_stride,
            )
        )
        if i % 25 == 0 or i == len(test_rows):
            print(f"  test raw {i}/{len(test_rows)}")

    val_gt_events = load_gt_events_for_test(
        test_rows=val_rows,
        master_rows_by_video=master_by_video,
        temporal_root=temporal_root,
    )
    test_gt_events = load_gt_events_for_test(
        test_rows=test_rows,
        master_rows_by_video=master_by_video,
        temporal_root=temporal_root,
    )

    has_val_temporal = len(val_gt_events) > 0
    print(f"Val temporal GT events: {len(val_gt_events)}")
    print(f"Test temporal GT events: {len(test_gt_events)}")

    sweep_records: List[Dict[str, object]] = []

    total = len(thresholds) * len(smooth_windows) * len(min_event_lens) * len(merge_gaps) * len(boundary_radii) * 2
    idx = 0

    for threshold in thresholds:
        for smooth_window in smooth_windows:
            for min_event_len in min_event_lens:
                for merge_gap in merge_gaps:
                    for boundary_radius in boundary_radii:
                        for use_boundary_refine in [False, True]:
                            idx += 1
                            if idx % 100 == 0 or idx == total:
                                print(f"Sweep {idx}/{total}")

                            val_eval = evaluate_setting_on_split(
                                raw_list=val_raw,
                                gt_events=val_gt_events,
                                class_names=class_names,
                                class_to_idx=class_to_idx,
                                normal_idx=normal_idx,
                                topk_ratio=args.topk_ratio,
                                threshold=threshold,
                                smooth_window=smooth_window,
                                min_event_len=min_event_len,
                                merge_gap=merge_gap,
                                boundary_radius=boundary_radius,
                                use_boundary_refine=use_boundary_refine,
                                tiou_thresholds=tiou_thresholds,
                            )

                            val_loc = val_eval["localization"]
                            val_tiou = val_loc.get("tiou", {})
                            val_map_03 = val_tiou.get("0.3", {}).get("mAP") if isinstance(val_tiou, dict) else None
                            val_map_05 = val_tiou.get("0.5", {}).get("mAP") if isinstance(val_tiou, dict) else None

                            val_binary = val_eval["metrics"]["binary"]
                            val_results = val_eval["results"]

                            val_pos_videos = sum(int(r["binary_label"]) for r in val_results)
                            val_pred_events = int(sum(len(r.get("events_after_refine", [])) for r in val_results))
                            val_avg_evt = float(val_pred_events / max(val_pos_videos, 1))

                            sweep_records.append(
                                {
                                    "threshold": threshold,
                                    "smooth_window": smooth_window,
                                    "min_event_len": min_event_len,
                                    "merge_gap": merge_gap,
                                    "boundary_radius": boundary_radius,
                                    "boundary_refine": use_boundary_refine,
                                    "val_map_03": val_map_03,
                                    "val_map_05": val_map_05,
                                    "val_binary_auc": val_binary.get("auc"),
                                    "val_binary_ap": val_binary.get("ap"),
                                    "val_binary_f1": val_binary.get("f1"),
                                    "val_pred_events": val_pred_events,
                                    "val_gt_events": int(val_loc.get("gt_event_count", 0)),
                                    "val_avg_events_per_positive_video": val_avg_evt,
                                }
                            )

    sweep_records.sort(key=lambda r: setting_rank_key(r, has_val_temporal), reverse=True)

    top10 = sweep_records[:10]
    best = sweep_records[0]

    chosen_eval = evaluate_setting_on_split(
        raw_list=test_raw,
        gt_events=test_gt_events,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold=float(best["threshold"]),
        smooth_window=int(best["smooth_window"]),
        min_event_len=int(best["min_event_len"]),
        merge_gap=int(best["merge_gap"]),
        boundary_radius=int(best["boundary_radius"]),
        use_boundary_refine=bool(best["boundary_refine"]),
        tiou_thresholds=tiou_thresholds,
    )

    # boundary ablation with same selected numeric settings
    with_bnd = evaluate_setting_on_split(
        raw_list=test_raw,
        gt_events=test_gt_events,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold=float(best["threshold"]),
        smooth_window=int(best["smooth_window"]),
        min_event_len=int(best["min_event_len"]),
        merge_gap=int(best["merge_gap"]),
        boundary_radius=int(best["boundary_radius"]),
        use_boundary_refine=True,
        tiou_thresholds=tiou_thresholds,
    )
    without_bnd = evaluate_setting_on_split(
        raw_list=test_raw,
        gt_events=test_gt_events,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold=float(best["threshold"]),
        smooth_window=int(best["smooth_window"]),
        min_event_len=int(best["min_event_len"]),
        merge_gap=int(best["merge_gap"]),
        boundary_radius=int(best["boundary_radius"]),
        use_boundary_refine=False,
        tiou_thresholds=tiou_thresholds,
    )

    baseline_step7_setting = {
        "threshold": 0.20,
        "smooth_window": 5,
        "min_event_len": 1,
        "merge_gap": 0,
        "boundary_radius": 2,
        "boundary_refine": True,
    }
    baseline_eval = evaluate_setting_on_split(
        raw_list=test_raw,
        gt_events=test_gt_events,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold=baseline_step7_setting["threshold"],
        smooth_window=baseline_step7_setting["smooth_window"],
        min_event_len=baseline_step7_setting["min_event_len"],
        merge_gap=baseline_step7_setting["merge_gap"],
        boundary_radius=baseline_step7_setting["boundary_radius"],
        use_boundary_refine=baseline_step7_setting["boundary_refine"],
        tiou_thresholds=tiou_thresholds,
    )

    gt_by_video: Dict[str, List[Dict[str, object]]] = {}
    for g in test_gt_events:
        gt_by_video.setdefault(str(g["video_id"]), []).append(g)

    examples = select_examples(
        chosen_results=chosen_eval["results"],
        baseline_results=baseline_eval["results"],
        gt_by_video=gt_by_video,
        threshold=float(best["threshold"]),
    )

    # Save full sweep CSV
    csv_path = output_dir / "val_sweep_records.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_records[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_records)

    chosen_loc = chosen_eval["localization"]
    chosen_tiou = chosen_loc.get("tiou", {})

    def m(loc: Dict[str, object], t: str) -> Optional[float]:
        ti = loc.get("tiou", {})
        if not isinstance(ti, dict):
            return None
        return ti.get(t, {}).get("mAP")

    boundary_ablation = {
        "with_boundary_refinement": {
            "auc": with_bnd["metrics"]["binary"].get("auc"),
            "ap": with_bnd["metrics"]["binary"].get("ap"),
            "macro_f1": with_bnd["metrics"]["classification"].get("macro_f1"),
            "weighted_f1": with_bnd["metrics"]["classification"].get("weighted_f1"),
            "mAP@0.3": m(with_bnd["localization"], "0.3"),
            "mAP@0.5": m(with_bnd["localization"], "0.5"),
            "mAP@0.7": m(with_bnd["localization"], "0.7"),
            "pred_events": int(with_bnd["localization"].get("pred_event_count", 0)),
            "gt_events": int(with_bnd["localization"].get("gt_event_count", 0)),
        },
        "without_boundary_refinement": {
            "auc": without_bnd["metrics"]["binary"].get("auc"),
            "ap": without_bnd["metrics"]["binary"].get("ap"),
            "macro_f1": without_bnd["metrics"]["classification"].get("macro_f1"),
            "weighted_f1": without_bnd["metrics"]["classification"].get("weighted_f1"),
            "mAP@0.3": m(without_bnd["localization"], "0.3"),
            "mAP@0.5": m(without_bnd["localization"], "0.5"),
            "mAP@0.7": m(without_bnd["localization"], "0.7"),
            "pred_events": int(without_bnd["localization"].get("pred_event_count", 0)),
            "gt_events": int(without_bnd["localization"].get("gt_event_count", 0)),
        },
    }

    results = {
        "step": "step9_calibration_on_step7",
        "args": json_safe_args(args),
        "notes": {
            "val_temporal_available": has_val_temporal,
            "selection_rule_requested": "highest val mAP@0.5, then val mAP@0.3, then event-count closeness",
            "selection_rule_applied": (
                "requested rule"
                if has_val_temporal
                else "fallback: val binary F1 + anti-fragmentation (val has no temporal GT annotations in this dataset)"
            ),
        },
        "validation_sweep_summary": {
            "total_settings": len(sweep_records),
            "top10": top10,
            "csv_path": str(csv_path),
        },
        "best_setting": best,
        "test_metrics_chosen_setting": {
            "binary": chosen_eval["metrics"]["binary"],
            "classification": chosen_eval["metrics"]["classification"],
            "temporal_localization": chosen_eval["localization"],
            "mAP@0.3": chosen_tiou.get("0.3", {}).get("mAP") if isinstance(chosen_tiou, dict) else None,
            "mAP@0.5": chosen_tiou.get("0.5", {}).get("mAP") if isinstance(chosen_tiou, dict) else None,
            "mAP@0.7": chosen_tiou.get("0.7", {}).get("mAP") if isinstance(chosen_tiou, dict) else None,
            "pred_events": int(chosen_eval["localization"].get("pred_event_count", 0)),
            "gt_events": int(chosen_eval["localization"].get("gt_event_count", 0)),
        },
        "boundary_ablation": boundary_ablation,
        "qualitative_examples": examples,
        "baseline_step7_reference_setting": baseline_step7_setting,
    }

    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_video_results_chosen.json").write_text(json.dumps(chosen_eval["results"], indent=2) + "\n")
    (output_dir / "pred_events_test_chosen.json").write_text(json.dumps(chosen_eval["pred_events"], indent=2) + "\n")

    print("\nStep-9 complete")
    print(f"- val temporal gt available: {has_val_temporal}")
    print(f"- best setting: {best}")
    print(f"- test AUC/AP: {chosen_eval['metrics']['binary']['auc']:.6f} / {chosen_eval['metrics']['binary']['ap']:.6f}")
    print(f"- test macro/weighted F1: {chosen_eval['metrics']['classification']['macro_f1']:.6f} / {chosen_eval['metrics']['classification']['weighted_f1']:.6f}")
    print(f"- test mAP@0.3/0.5/0.7: {m(chosen_eval['localization'], '0.3'):.6f} / {m(chosen_eval['localization'], '0.5'):.6f} / {m(chosen_eval['localization'], '0.7'):.6f}")
    print(f"- pred vs gt events: {chosen_eval['localization']['pred_event_count']} vs {chosen_eval['localization']['gt_event_count']}")
    print(f"- saved: {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
