#!/usr/bin/env python3
"""Step-13: ShanghaiTech binary robustness evaluation (zero-shot, no retraining)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix

from evaluate_ablation_checkpoint import infer_and_postprocess_video, load_model
from train_rtfm_trn_boundary import (
    choose_device,
    parse_class_names,
    read_csv,
    resolve,
    safe_ap,
    safe_auc,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShanghaiTech robustness evaluation")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_features_i3d.csv"),
    )
    p.add_argument(
        "--master-manifest",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_master.csv"),
    )
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument(
        "--model-kind",
        choices=["step5_classifier", "step6_trn", "step7_boundary", "step8_progressive"],
        default="step7_boundary",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/step10_ablations/k1/train/checkpoints/best.pt"),
    )

    # Must match training architecture args.
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--target-segments", type=int, default=32)
    p.add_argument("--trn-layers", type=int, default=2)
    p.add_argument("--trn-heads", type=int, default=4)
    p.add_argument("--trn-ffn-mult", type=int, default=4)
    p.add_argument("--trn-dropout", type=float, default=0.1)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    # Locked decoding settings.
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

    p.add_argument(
        "--class-names",
        type=str,
        default="normal,fighting,shooting,explosion,robbery,abuse",
    )
    p.add_argument("--normal-class", type=str, default="normal")

    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/shanghaitech_robustness/results_summary.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pick_qualitative_examples(rows: List[Dict[str, object]], threshold: float) -> Dict[str, List[Dict[str, object]]]:
    successes: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for r in rows:
        gt_is_anom = int(r.get("binary_label", 0))
        score = as_float(r.get("video_anomaly_score", 0.0), 0.0)
        pred_is_anom = int(score >= threshold)
        gt_label = "anomaly" if gt_is_anom else "normal"
        pred_label = "anomaly" if pred_is_anom else "normal"

        if gt_is_anom and pred_is_anom:
            note = "anomalous clip correctly detected."
            rank = score
            bucket = successes
        elif (not gt_is_anom) and (not pred_is_anom):
            note = "normal clip correctly rejected."
            rank = 1.0 - score
            bucket = successes
        elif (not gt_is_anom) and pred_is_anom:
            note = "false positive; normal scene flagged as anomaly."
            rank = score
            bucket = failures
        else:
            note = "false negative; anomaly missed."
            rank = 1.0 - score
            bucket = failures

        bucket.append(
            {
                "video_id": r.get("video_id"),
                "gt_label": gt_label,
                "predicted_label": pred_label,
                "anomaly_score": float(score),
                "note": note,
                "_rank": float(rank),
            }
        )

    successes.sort(key=lambda x: x["_rank"], reverse=True)
    failures.sort(key=lambda x: x["_rank"], reverse=True)
    return {
        "successes": [{k: v for k, v in x.items() if k != "_rank"} for x in successes[:3]],
        "failures": [{k: v for k, v in x.items() if k != "_rank"} for x in failures[:2]],
    }


def main() -> None:
    args = parse_args()

    root = args.project_root.resolve()
    feature_manifest = resolve(root, args.feature_manifest)
    master_manifest = resolve(root, args.master_manifest)
    checkpoint = resolve(root, args.checkpoint)
    out_json = resolve(root, args.output_json)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list: {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    set_seed(args.seed)
    device = choose_device(args.device)

    # load_model helper reads args.checkpoint directly
    args.checkpoint = checkpoint
    model, _ = load_model(args, num_classes=len(class_names), device=device)

    feature_rows = [r for r in read_csv(feature_manifest) if r.get("status", "ok") == "ok"]
    if args.split != "all":
        feature_rows = [r for r in feature_rows if r.get("split") == args.split]

    master_rows = read_csv(master_manifest)
    master_by_id = {r["video_id"]: r for r in master_rows}

    results: List[Dict[str, object]] = []
    total = len(feature_rows)
    for i, r in enumerate(feature_rows, 1):
        if r.get("video_id") in master_by_id:
            mr = master_by_id[r["video_id"]]
            r = {
                **r,
                "binary_label": mr.get("binary_label", r.get("binary_label", "0")),
                "category_label": mr.get("category_label", r.get("category_label", "normal")),
                "split": mr.get("split", r.get("split", args.split)),
            }

        out = infer_and_postprocess_video(
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
        results.append(out)
        if i % 25 == 0 or i == total:
            print(f"  inference {i}/{total}")

    y_true = np.array([int(r.get("binary_label", 0)) for r in results], dtype=np.int64)
    y_score = np.array([as_float(r.get("video_anomaly_score", 0.0), 0.0) for r in results], dtype=np.float64)
    y_pred = (y_score >= args.threshold).astype(np.int64)

    auc = float(safe_auc(y_true, y_score))
    ap = float(safe_ap(y_true, y_score))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    normal_count = int((y_true == 0).sum())
    anomaly_count = int((y_true == 1).sum())

    seg_counts = [
        int(float(x.get("num_segments", "0")))
        for x in feature_rows
        if str(x.get("num_segments", "")).strip()
    ]
    failed_rows = [x for x in read_csv(feature_manifest) if x.get("status") != "ok"]

    qualitative = pick_qualitative_examples(results, threshold=args.threshold)

    # Lightweight error-profile heuristic
    if fp > fn * 1.5:
        failure_mode = "likely scene bias / domain shift (many false positives)"
    elif fn > fp * 1.5:
        failure_mode = "likely weak anomaly confidence / conservative thresholding (many false negatives)"
    else:
        failure_mode = "mixed errors: domain shift and confidence issues both present"

    summary = {
        "dataset_summary": {
            "total_videos_evaluated": int(len(results)),
            "normal_count": normal_count,
            "anomaly_count": anomaly_count,
            "split_used": args.split,
            "dataset": "shanghaitech",
        },
        "feature_extraction_summary": {
            "total_extracted": int(len(feature_rows)),
            "total_failed": int(len(failed_rows)),
            "segment_stats": {
                "min": int(min(seg_counts)) if seg_counts else -1,
                "max": int(max(seg_counts)) if seg_counts else -1,
                "mean": float(sum(seg_counts) / len(seg_counts)) if seg_counts else float("nan"),
            },
            "extractor_match_confirmation": {
                "frozen_i3d": True,
                "segment_length": 16,
                "overlap": 0,
                "tail_rule": "drop",
            },
        },
        "robustness_metrics": {
            "auc": auc,
            "ap": ap,
        },
        "error_profile": {
            "false_positives": fp,
            "false_negatives": fn,
            "confusion_matrix_labels": ["normal", "anomaly"],
            "confusion_matrix": cm,
            "failure_mode_guess": failure_mode,
        },
        "qualitative_examples": qualitative,
        "decode_settings_used": {
            "threshold": args.threshold,
            "smooth_window": args.smooth_window,
            "min_event_len": args.min_event_len,
            "merge_gap": args.merge_gap,
            "boundary_radius": args.boundary_radius,
            "boundary_refine": bool(args.boundary_refine),
            "zero_shot_no_shanghaitech_tuning": True,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nStep-13 ShanghaiTech robustness complete")
    print(f"- AUC/AP: {auc:.6f} / {ap:.6f}")
    print(f"- false positives / false negatives: {fp} / {fn}")
    print(f"- saved: {out_json}")


if __name__ == "__main__":
    main()

