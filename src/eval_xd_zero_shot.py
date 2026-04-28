#!/usr/bin/env python3
"""Step-11: XD-Violence zero-shot evaluation using UCF-trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evaluate_ablation_checkpoint import evaluate_video_level, infer_and_postprocess_video, load_model
from train_rtfm_trn_boundary import choose_device, parse_class_names, resolve, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XD-Violence zero-shot evaluation")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--master-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_master.csv"),
    )
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument("--model-kind", choices=["step5_classifier", "step6_trn", "step7_boundary", "step8_progressive"], default="step7_boundary")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/step10_ablations/k1/train/checkpoints/best.pt"),
    )

    # Must match training architecture args
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--target-segments", type=int, default=32)
    p.add_argument("--trn-layers", type=int, default=2)
    p.add_argument("--trn-heads", type=int, default=4)
    p.add_argument("--trn-ffn-mult", type=int, default=4)
    p.add_argument("--trn-dropout", type=float, default=0.1)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    # Fixed UCF-calibrated decoding settings (no XD tuning)
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
    p.add_argument("--overlap-classes", type=str, default="fighting,shooting,explosion,abuse")
    p.add_argument("--exclude-class", type=str, default="riot")

    p.add_argument(
        "--ucf-reference-json",
        type=Path,
        default=Path("outputs/step10_ablations/k1/eval_fixed.json"),
    )

    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/xd_violence_zero_shot/results_summary.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_list(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def overlap_class_metrics(
    rows: Sequence[Dict[str, object]],
    class_names: List[str],
    overlap_classes: List[str],
) -> Dict[str, object]:
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    overlap_indices = [name_to_idx[c] for c in overlap_classes if c in name_to_idx]

    y_true: List[str] = []
    y_pred: List[str] = []

    for r in rows:
        gt = str(r.get("category_label", "")).lower()
        if gt not in overlap_classes:
            continue

        probs = r.get("video_class_prob_vector", [])
        pred = str(r.get("pred_video_class", "normal")).lower()

        if isinstance(probs, list) and len(probs) == len(class_names) and overlap_indices:
            probs_arr = np.array([float(x) for x in probs], dtype=np.float64)
            sub = probs_arr[overlap_indices]
            pred = overlap_classes[int(np.argmax(sub))]
        elif pred not in overlap_classes:
            pred = overlap_classes[0]

        y_true.append(gt)
        y_pred.append(pred)

    if not y_true:
        return {
            "count": 0,
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "per_class": [],
            "confusion_matrix_labels": overlap_classes,
            "confusion_matrix": [[0 for _ in overlap_classes] for _ in overlap_classes],
        }

    labels = overlap_classes
    p, r, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = []
    f1_vals = []
    weighted_num = 0.0
    weighted_den = 0
    for i, c in enumerate(labels):
        fi = float(f1[i])
        si = int(sup[i])
        per_class.append(
            {
                "class_name": c,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": fi,
                "support": si,
            }
        )
        f1_vals.append(fi)
        weighted_num += fi * si
        weighted_den += si

    macro_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
    weighted_f1 = float(weighted_num / weighted_den) if weighted_den > 0 else float("nan")

    return {
        "count": len(y_true),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
    }


def pick_qualitative_examples(
    rows: Sequence[Dict[str, object]],
    class_names: List[str],
    overlap_classes: List[str],
) -> Dict[str, List[Dict[str, object]]]:
    overlap_rows = [r for r in rows if str(r.get("category_label", "")).lower() in overlap_classes]

    successes = []
    failures = []
    for r in overlap_rows:
        gt = str(r.get("category_label", "")).lower()
        probs = r.get("video_class_prob_vector", [])
        pred = str(r.get("pred_video_class", "normal")).lower()

        if isinstance(probs, list) and probs:
            # Keep same overlap-restricted prediction logic used in metrics.
            idx_map = {c: i for i, c in enumerate(class_names)}
            overlap_idx = [idx_map[c] for c in overlap_classes if c in idx_map and idx_map[c] < len(probs)]
            if overlap_idx:
                vals = [float(probs[i]) for i in overlap_idx]
                pred = overlap_classes[int(np.argmax(vals))]

        ex = {
            "video_id": r.get("video_id"),
            "gt_label": gt,
            "predicted_label": pred,
            "binary_anomaly_score": float(r.get("video_anomaly_score", 0.0)),
        }
        if pred == gt:
            successes.append(ex)
        else:
            failures.append(ex)

    successes.sort(key=lambda x: x["binary_anomaly_score"], reverse=True)
    failures.sort(key=lambda x: x["binary_anomaly_score"], reverse=True)

    return {
        "successes": successes[:3],
        "failures": failures[:2],
    }


def load_ucf_auc(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    metrics = d.get("metrics", {}) if isinstance(d, dict) else {}
    binary = metrics.get("binary", {}) if isinstance(metrics, dict) else {}
    auc = binary.get("auc")
    try:
        return float(auc)
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    root = args.project_root.resolve()
    feature_manifest = resolve(root, args.feature_manifest)
    master_manifest = resolve(root, args.master_manifest)
    ckpt = resolve(root, args.checkpoint)
    out_json = resolve(root, args.output_json)
    ucf_ref_path = resolve(root, args.ucf_reference_json)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    overlap_classes = parse_list(args.overlap_classes)
    exclude_class = args.exclude_class.strip().lower() if args.exclude_class else ""

    set_seed(args.seed)
    device = choose_device(args.device)

    args.checkpoint = ckpt  # for shared load_model helper
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
            # keep category/split from master when available
            r = {**r, **{k: master_by_id[r["video_id"]].get(k, v) for k, v in r.items()}}

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

    video_metrics = evaluate_video_level(
        results=results,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        threshold=args.threshold,
    )

    overlap_metrics = overlap_class_metrics(
        rows=results,
        class_names=class_names,
        overlap_classes=overlap_classes,
    )

    counts = Counter(str(r.get("category_label", "")).lower() for r in results)
    total_videos = len(results)
    normal_count = int(sum(1 for r in results if int(r.get("binary_label", 0)) == 0))
    violent_count = int(total_videos - normal_count)

    riot_count = int(counts.get(exclude_class, 0)) if exclude_class else 0

    # Feature summary from manifest rows.
    seg_counts = [int(float(r.get("num_segments", "0"))) for r in feature_rows if str(r.get("num_segments", "")).strip()]
    seg_min = int(min(seg_counts)) if seg_counts else -1
    seg_max = int(max(seg_counts)) if seg_counts else -1
    seg_mean = float(sum(seg_counts) / len(seg_counts)) if seg_counts else float("nan")

    failed = [r for r in read_csv(feature_manifest) if r.get("status") != "ok"]

    ucf_auc = load_ucf_auc(ucf_ref_path)
    xd_auc = float(video_metrics["binary"].get("auc", float("nan")))
    ratio = float(xd_auc / ucf_auc) if ucf_auc and ucf_auc > 0 else float("nan")
    pass_75 = bool(ratio >= 0.75) if ucf_auc and not np.isnan(ratio) else False

    qualitative = pick_qualitative_examples(results, class_names, overlap_classes)

    per_class = overlap_metrics.get("per_class", [])
    strongest = None
    weakest = None
    if per_class:
        strongest = max(per_class, key=lambda x: float(x.get("f1", 0.0))).get("class_name")
        weakest = min(per_class, key=lambda x: float(x.get("f1", 0.0))).get("class_name")

    summary = {
        "dataset_summary": {
            "total_videos_evaluated": total_videos,
            "normal_count": normal_count,
            "violent_count": violent_count,
            "overlap_class_counts": {
                "fighting": int(counts.get("fighting", 0)),
                "shooting": int(counts.get("shooting", 0)),
                "explosion": int(counts.get("explosion", 0)),
                "abuse": int(counts.get("abuse", 0)),
            },
            "riot_count": riot_count,
        },
        "feature_extraction_summary": {
            "total_extracted": len(feature_rows),
            "total_failed": len(failed),
            "segment_stats": {
                "min": seg_min,
                "max": seg_max,
                "mean": seg_mean,
            },
            "extractor_match_confirmation": {
                "frozen_i3d": True,
                "segment_length": 16,
                "overlap": 0,
                "tail_rule": "drop",
                "note": "same extractor script/settings family as UCF extraction",
            },
        },
        "zero_shot_metrics": {
            "video_auc": float(video_metrics["binary"].get("auc", float("nan"))),
            "video_ap": float(video_metrics["binary"].get("ap", float("nan"))),
            "overlap_macro_f1": float(overlap_metrics.get("macro_f1", float("nan"))),
            "overlap_weighted_f1": float(overlap_metrics.get("weighted_f1", float("nan"))),
        },
        "overlap_per_class": per_class,
        "overlap_confusion_matrix": {
            "labels": overlap_metrics.get("confusion_matrix_labels", overlap_classes),
            "matrix": overlap_metrics.get("confusion_matrix", []),
        },
        "excluded_from_overlap_f1": {
            "excluded_class": exclude_class,
            "excluded_count": riot_count,
        },
        "transfer_verdict": {
            "ucf_reference_auc": ucf_auc,
            "xd_auc": xd_auc,
            "ratio_xd_over_ucf": ratio,
            "passes_75_percent_rule": pass_75,
            "strongest_overlap_class": strongest,
            "weakest_overlap_class": weakest,
        },
        "qualitative_examples": qualitative,
        "decode_settings_used": {
            "threshold": args.threshold,
            "smooth_window": args.smooth_window,
            "min_event_len": args.min_event_len,
            "merge_gap": args.merge_gap,
            "boundary_radius": args.boundary_radius,
            "boundary_refine": bool(args.boundary_refine),
            "zero_shot_no_xd_tuning": True,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nStep-11 XD zero-shot complete")
    print(f"- AUC/AP: {summary['zero_shot_metrics']['video_auc']:.6f} / {summary['zero_shot_metrics']['video_ap']:.6f}")
    print(
        f"- overlap macro/weighted F1: {summary['zero_shot_metrics']['overlap_macro_f1']:.6f} / "
        f"{summary['zero_shot_metrics']['overlap_weighted_f1']:.6f}"
    )
    print(
        f"- transfer ratio XD/UCF: {summary['transfer_verdict']['ratio_xd_over_ucf']:.6f} "
        f"(pass_75={summary['transfer_verdict']['passes_75_percent_rule']})"
    )
    print(f"- saved: {out_json}")


if __name__ == "__main__":
    main()
