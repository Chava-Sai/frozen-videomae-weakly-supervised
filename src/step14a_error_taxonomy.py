#!/usr/bin/env python3
"""Step-14A: Cross-dataset error taxonomy + case-study bank.

Runs zero-shot inference on UCF/XD/RWF/Shanghai feature manifests with a locked
checkpoint and generates:
- unified per-video error table
- heuristic taxonomy counts
- 10-case study bank (5 successes, 5 failures; cross-dataset coverage)
- short cross-dataset insight summary
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evaluate_ablation_checkpoint import infer_and_postprocess_video, load_model
from train_rtfm_trn_boundary import choose_device, parse_class_names, read_csv, resolve, set_seed

DATASETS = ("ucf_crime", "xd_violence", "rwf_2000", "shanghaitech")
TAXONOMY_LABELS = [
    "occlusion",
    "camera motion",
    "crowd density",
    "low light",
    "similar normal motion",
    "clip-boundary / incomplete event",
    "other",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-14A error taxonomy and case studies")
    p.add_argument("--project-root", type=Path, default=Path.cwd())

    # Dataset manifests.
    p.add_argument("--ucf-feature-manifest", type=Path, default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"))
    p.add_argument("--ucf-master-manifest", type=Path, default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"))
    p.add_argument("--ucf-split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument("--xd-feature-manifest", type=Path, default=Path("data/xd_violence/manifests/xd_violence_features_i3d.csv"))
    p.add_argument("--xd-master-manifest", type=Path, default=Path("data/xd_violence/manifests/xd_violence_master.csv"))
    p.add_argument("--xd-split", choices=["train", "val", "test", "all"], default="test")

    p.add_argument("--rwf-feature-manifest", type=Path, default=Path("data/rwf_2000/manifests/rwf_2000_features_i3d.csv"))
    p.add_argument("--rwf-master-manifest", type=Path, default=Path("data/rwf_2000/manifests/rwf_2000_master.csv"))
    p.add_argument("--rwf-split", choices=["train", "val", "test", "all"], default="val")

    p.add_argument("--sh-feature-manifest", type=Path, default=Path("data/shanghaitech/manifests/shanghaitech_features_i3d.csv"))
    p.add_argument("--sh-master-manifest", type=Path, default=Path("data/shanghaitech/manifests/shanghaitech_master.csv"))
    p.add_argument("--sh-split", choices=["train", "val", "test", "all"], default="test")

    # Locked model settings.
    p.add_argument("--model-kind", choices=["step5_classifier", "step6_trn", "step7_boundary", "step8_progressive"], default="step7_boundary")
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
    p.add_argument(
        "--rwf-positive-mode",
        choices=["fight_label", "anomaly_threshold"],
        default="fight_label",
        help="How to compute binary status on RWF: use predicted fighting label (default) or anomaly threshold.",
    )

    p.add_argument("--out-dir", type=Path, default=Path("outputs/step14_interpretability/step14a"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def resolve_with_fallback(primary: Path, fallbacks: Sequence[Path]) -> Path:
    if primary.exists():
        return primary
    for p in fallbacks:
        if p.exists():
            return p
    return primary


def normalize_gt_label(dataset: str, raw: str, binary_label: int) -> str:
    r = (raw or "").strip().lower()
    if dataset == "rwf_2000":
        return "fighting" if binary_label == 1 else "normal"
    if dataset == "shanghaitech":
        return "anomaly" if binary_label == 1 else "normal"
    return r if r else ("anomaly" if binary_label == 1 else "normal")


def infer_dataset(
    dataset: str,
    split: str,
    feature_rows: Sequence[Dict[str, str]],
    master_by_id: Dict[str, Dict[str, str]],
    model,
    class_names: List[str],
    normal_idx: int,
    args: argparse.Namespace,
    root: Path,
    device: str,
) -> List[Dict[str, object]]:
    if split != "all":
        rows = [r for r in feature_rows if r.get("split") == split]
    else:
        rows = list(feature_rows)

    out: List[Dict[str, object]] = []
    total = len(rows)
    for i, r in enumerate(rows, 1):
        if r.get("status", "ok") != "ok":
            continue
        mr = master_by_id.get(r.get("video_id", ""), {})
        row = dict(r)
        for k in ("binary_label", "category_label", "split", "video_path"):
            if k in mr and mr[k] not in ("", None):
                row[k] = mr[k]

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

        binary_label = int(row.get("binary_label", 0))
        score = as_float(pred.get("video_anomaly_score", 0.0), 0.0)
        pred_video_class = str(pred.get("pred_video_class", "normal")).lower()
        if dataset == "rwf_2000" and args.rwf_positive_mode == "fight_label":
            pred_binary = 1 if pred_video_class == "fighting" else 0
        else:
            pred_binary = 1 if score >= args.threshold else 0

        if binary_label == 1 and pred_binary == 1:
            status = "TP"
        elif binary_label == 0 and pred_binary == 0:
            status = "TN"
        elif binary_label == 0 and pred_binary == 1:
            status = "FP"
        else:
            status = "FN"

        gt_label = normalize_gt_label(dataset, str(row.get("category_label", "")), binary_label)
        pred_label = pred_video_class
        if dataset == "shanghaitech":
            pred_label = "anomaly" if pred_binary == 1 else "normal"
        elif dataset == "rwf_2000":
            pred_label = "fighting" if pred_binary == 1 else "normal"

        out.append(
            {
                "video_id": row.get("video_id", ""),
                "dataset": dataset,
                "split": row.get("split", split),
                "video_path": row.get("video_path", ""),
                "gt_label": gt_label,
                "predicted_label": pred_label,
                "binary_label": binary_label,
                "pred_binary": pred_binary,
                "binary_status": status,
                "anomaly_score": score,
                "binary_confidence": score if pred_binary else (1.0 - score),
                "class_confidence": as_float(pred.get("pred_video_class_prob", 0.0), 0.0),
                "num_segments": int(pred.get("num_segments", 0)),
                "spans_after_refine": pred.get("spans_after_refine", []),
            }
        )
        if i % 50 == 0 or i == total:
            print(f"  {dataset}: inference {i}/{total}")
    return out


def suggest_taxonomy(row: Dict[str, object], threshold: float) -> Tuple[str, str]:
    status = str(row.get("binary_status", ""))
    dataset = str(row.get("dataset", ""))
    score = as_float(row.get("anomaly_score", 0.0), 0.0)
    nseg = int(row.get("num_segments", 0))
    spans = row.get("spans_after_refine", [])

    near_boundary = False
    if isinstance(spans, list) and nseg > 0:
        for s, e in spans:
            if int(s) <= 1 or int(e) >= nseg - 2:
                near_boundary = True
                break

    if status in {"TP", "TN"}:
        return "other", "Correct prediction."

    if status == "FN":
        if dataset == "shanghaitech":
            return "crowd density", "Likely domain shift to crowded scene anomalies; anomaly confidence collapsed."
        if dataset == "rwf_2000":
            return "similar normal motion", "Fight dynamics likely confused with normal human motion patterns."
        if near_boundary:
            return "clip-boundary / incomplete event", "Predicted evidence appears near clip edges; event may be truncated."
        if score < min(0.2, threshold * 0.5):
            return "low light", "Very low anomaly confidence on positive sample suggests weak visual saliency / low visibility."
        return "similar normal motion", "Anomalous action likely resembles learned normal motion pattern."

    # FP
    if near_boundary:
        return "clip-boundary / incomplete event", "False alarm concentrated near clip boundary."
    if dataset in {"xd_violence", "ucf_crime"} and score < threshold + 0.1:
        return "camera motion", "Likely transient motion burst around threshold."
    if dataset == "shanghaitech":
        return "camera motion", "Crowd/background dynamics likely misread as anomaly."
    return "other", "False positive with no strong heuristic signature."


def manual_note(row: Dict[str, object], taxonomy: str, note: str) -> str:
    gt = row["gt_label"]
    pred = row["predicted_label"]
    score = as_float(row["anomaly_score"])
    status = row["binary_status"]
    return (
        f"{status}: gt={gt}, pred={pred}, anomaly_score={score:.6f}. "
        f"Suggested failure type='{taxonomy}'. {note}"
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def rank_success(row: Dict[str, object]) -> float:
    s = row["binary_status"]
    sc = as_float(row["anomaly_score"])
    if s == "TP":
        return sc
    if s == "TN":
        return 1.0 - sc
    return -1.0


def rank_failure(row: Dict[str, object]) -> float:
    s = row["binary_status"]
    sc = as_float(row["anomaly_score"])
    if s == "FN":
        return 1.0 - sc
    if s == "FP":
        return sc
    return -1.0


def build_case_studies(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_dataset[str(r["dataset"])].append(r)

    # Required mix:
    # successes: UCF1, XD2, RWF1, SH1
    # failures : UCF1, XD1, RWF1, SH2
    success_need = {"ucf_crime": 1, "xd_violence": 2, "rwf_2000": 1, "shanghaitech": 1}
    failure_need = {"ucf_crime": 1, "xd_violence": 1, "rwf_2000": 1, "shanghaitech": 2}

    selected: List[Dict[str, object]] = []

    for ds, k in success_need.items():
        cand = [r for r in by_dataset.get(ds, []) if r["binary_status"] in {"TP", "TN"}]
        cand.sort(key=rank_success, reverse=True)
        for r in cand[:k]:
            rr = dict(r)
            rr["case_group"] = "success"
            selected.append(rr)

    for ds, k in failure_need.items():
        cand = [r for r in by_dataset.get(ds, []) if r["binary_status"] in {"FP", "FN"}]
        cand.sort(key=rank_failure, reverse=True)
        for r in cand[:k]:
            rr = dict(r)
            rr["case_group"] = "failure"
            selected.append(rr)

    # Ensure full 10-case bank even if some dataset slices are sparse/missing.
    selected_ids = {(r["dataset"], r["video_id"], r["case_group"]) for r in selected}
    current_s = [r for r in selected if r["case_group"] == "success"]
    current_f = [r for r in selected if r["case_group"] == "failure"]

    if len(current_s) < 5:
        pool_s = [r for r in rows if r["binary_status"] in {"TP", "TN"}]
        pool_s.sort(key=rank_success, reverse=True)
        for r in pool_s:
            key = (r["dataset"], r["video_id"], "success")
            if key in selected_ids:
                continue
            rr = dict(r)
            rr["case_group"] = "success"
            selected.append(rr)
            selected_ids.add(key)
            current_s.append(rr)
            if len(current_s) >= 5:
                break

    if len(current_f) < 5:
        pool_f = [r for r in rows if r["binary_status"] in {"FP", "FN"}]
        pool_f.sort(key=rank_failure, reverse=True)
        for r in pool_f:
            key = (r["dataset"], r["video_id"], "failure")
            if key in selected_ids:
                continue
            rr = dict(r)
            rr["case_group"] = "failure"
            selected.append(rr)
            selected_ids.add(key)
            current_f.append(rr)
            if len(current_f) >= 5:
                break

    # Trim deterministically to exactly 10 if oversized.
    succ = [r for r in selected if r["case_group"] == "success"]
    fail = [r for r in selected if r["case_group"] == "failure"]
    succ.sort(key=rank_success, reverse=True)
    fail.sort(key=rank_failure, reverse=True)
    selected = fail[:5] + succ[:5]

    # Stable ordering for report.
    selected.sort(key=lambda x: (x["case_group"], x["dataset"], x["video_id"]))
    return selected


def case_explanation(row: Dict[str, object]) -> str:
    ds = row["dataset"]
    status = row["binary_status"]
    score = as_float(row["anomaly_score"])
    tax = row["taxonomy_label"]
    gt = row["gt_label"]
    pred = row["predicted_label"]

    if status in {"TP", "TN"}:
        return (
            f"Prediction is correct ({status}) with anomaly score {score:.4f}. "
            f"This suggests stable transfer behavior for this sample on {ds}."
        )
    return (
        f"Prediction is incorrect ({status}); gt={gt} but pred={pred} at anomaly score {score:.4f}. "
        f"Heuristic failure type is '{tax}', indicating likely domain/visual mismatch."
    )


def short_insights(rows: Sequence[Dict[str, object]]) -> Dict[str, str]:
    by_ds: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    out: Dict[str, str] = {}
    for ds, ds_rows in by_ds.items():
        fn = sum(1 for r in ds_rows if r["binary_status"] == "FN")
        fp = sum(1 for r in ds_rows if r["binary_status"] == "FP")
        dom_tax = Counter(
            r["taxonomy_label"] for r in ds_rows if r["binary_status"] in {"FP", "FN"}
        ).most_common(2)
        dom = ", ".join([f"{k}({v})" for k, v in dom_tax]) if dom_tax else "none"
        out[ds] = f"FP={fp}, FN={fn}; dominant failure types: {dom}."
    return out


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' missing from class_names")
    normal_idx = class_to_idx[args.normal_class]

    ckpt = resolve(root, args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    set_seed(args.seed)
    device = choose_device(args.device)
    args.checkpoint = ckpt
    model, _ = load_model(args, num_classes=len(class_names), device=device)

    ucf_feat = resolve(root, args.ucf_feature_manifest)
    xd_feat = resolve(root, args.xd_feature_manifest)
    rwf_feat = resolve(root, args.rwf_feature_manifest)
    sh_feat = resolve(root, args.sh_feature_manifest)

    # Common fallback in this project where XD was generated as "*_testonly.csv".
    xd_feat = resolve_with_fallback(
        xd_feat,
        [resolve(root, Path("data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv"))],
    )

    configs = [
        ("ucf_crime", args.ucf_split, ucf_feat, resolve(root, args.ucf_master_manifest)),
        ("xd_violence", args.xd_split, xd_feat, resolve(root, args.xd_master_manifest)),
        ("rwf_2000", args.rwf_split, rwf_feat, resolve(root, args.rwf_master_manifest)),
        ("shanghaitech", args.sh_split, sh_feat, resolve(root, args.sh_master_manifest)),
    ]

    all_rows: List[Dict[str, object]] = []
    missing_inputs: List[str] = []

    for ds, split, feat_path, master_path in configs:
        if not feat_path.exists() or not master_path.exists():
            missing_inputs.append(f"{ds}: missing manifest(s): {feat_path} | {master_path}")
            continue

        feat_rows = [r for r in load_rows(feat_path) if r.get("status", "ok") == "ok"]
        master_rows = load_rows(master_path)
        master_by_id = {r.get("video_id", ""): r for r in master_rows}

        ds_rows = infer_dataset(
            dataset=ds,
            split=split,
            feature_rows=feat_rows,
            master_by_id=master_by_id,
            model=model,
            class_names=class_names,
            normal_idx=normal_idx,
            args=args,
            root=root,
            device=device,
        )
        for r in ds_rows:
            tax, note = suggest_taxonomy(r, threshold=args.threshold)
            r["taxonomy_label"] = tax
            r["taxonomy_note"] = note
            r["manual_note"] = manual_note(r, tax, note)
            all_rows.append(r)

    if missing_inputs:
        print("Warning: some datasets were skipped due to missing files:")
        for m in missing_inputs:
            print(" -", m)

    # Unified table.
    fields = [
        "video_id",
        "dataset",
        "split",
        "video_path",
        "gt_label",
        "predicted_label",
        "binary_label",
        "pred_binary",
        "binary_status",
        "anomaly_score",
        "binary_confidence",
        "class_confidence",
        "num_segments",
        "taxonomy_label",
        "taxonomy_note",
        "manual_note",
    ]
    error_csv = out_dir / "step14a_error_table.csv"
    write_csv(error_csv, all_rows, fields)

    # Taxonomy counts over failures.
    fail_rows = [r for r in all_rows if r["binary_status"] in {"FP", "FN"}]
    tax_counts = Counter(r["taxonomy_label"] for r in fail_rows)
    tax_payload = {k: int(tax_counts.get(k, 0)) for k in TAXONOMY_LABELS}
    tax_json = out_dir / "step14a_taxonomy_counts.json"
    tax_json.write_text(json.dumps(tax_payload, indent=2) + "\n")

    # Case study bank.
    cases = build_case_studies(all_rows)
    case_rows: List[Dict[str, object]] = []
    for r in cases:
        rr = dict(r)
        rr["explanation"] = case_explanation(r)
        case_rows.append(rr)
    cases_json = out_dir / "step14a_case_studies.json"
    cases_json.write_text(json.dumps(case_rows, indent=2) + "\n")

    case_fields = [
        "case_group",
        "video_id",
        "dataset",
        "gt_label",
        "predicted_label",
        "binary_status",
        "anomaly_score",
        "binary_confidence",
        "class_confidence",
        "taxonomy_label",
        "explanation",
    ]
    cases_csv = out_dir / "step14a_case_studies.csv"
    write_csv(cases_csv, case_rows, case_fields)

    # Cross-dataset short insights.
    insights = short_insights(all_rows)

    # Human-readable report.
    report = out_dir / "step14a_report.txt"
    lines: List[str] = []
    lines.append("1) Error taxonomy counts")
    for k in TAXONOMY_LABELS:
        lines.append(f"- {k}: {tax_payload.get(k, 0)}")

    lines.append("")
    lines.append("2) Ten selected case studies")
    for c in case_rows:
        lines.append(
            f"- [{c['case_group']}] {c['video_id']} | dataset={c['dataset']} | "
            f"gt={c['gt_label']} | pred={c['predicted_label']} | "
            f"anomaly_score={as_float(c['anomaly_score']):.6f} | "
            f"type={c['taxonomy_label']}"
        )
        lines.append(f"  explanation: {c['explanation']}")

    lines.append("")
    lines.append("3) Short cross-dataset insight summary")
    if missing_inputs:
        lines.append("- warning: some datasets were skipped due to missing manifests:")
        for m in missing_inputs:
            lines.append(f"  - {m}")
    for ds in DATASETS:
        if ds in insights:
            lines.append(f"- {ds}: {insights[ds]}")

    # Required bullets from advisor note.
    def ds_fail_stats(ds: str) -> Tuple[int, int]:
        rows = [r for r in all_rows if r["dataset"] == ds]
        fp = sum(1 for r in rows if r["binary_status"] == "FP")
        fn = sum(1 for r in rows if r["binary_status"] == "FN")
        return fp, fn

    ucf_fp, ucf_fn = ds_fail_stats("ucf_crime")
    xd_fp, xd_fn = ds_fail_stats("xd_violence")
    rwf_fp, rwf_fn = ds_fail_stats("rwf_2000")
    sh_fp, sh_fn = ds_fail_stats("shanghaitech")
    lines.append(f"- what kinds of errors dominate on UCF: FP={ucf_fp}, FN={ucf_fn}; see taxonomy counts in table.")
    lines.append(f"- what kinds dominate on XD: FP={xd_fp}, FN={xd_fn}; class/scene confusion appears in suggested taxonomy.")
    lines.append(f"- why RWF fight recall is low: FN-dominant pattern (FP={rwf_fp}, FN={rwf_fn}) consistent with conservative fight mapping.")
    lines.append(f"- why ShanghaiTech collapses: severe FN dominance (FP={sh_fp}, FN={sh_fn}) indicating strong domain-shift mismatch.")
    lines.append("")
    lines.append("Note: taxonomy labels above are heuristic suggestions and should be manually audited for final report figures.")
    report.write_text("\n".join(lines) + "\n")

    # Meta summary.
    meta = {
        "total_rows": len(all_rows),
        "total_failures": len(fail_rows),
        "datasets_included": sorted(set(r["dataset"] for r in all_rows)),
        "missing_inputs": missing_inputs,
        "outputs": {
            "error_table_csv": str(error_csv),
            "taxonomy_counts_json": str(tax_json),
            "case_studies_json": str(cases_json),
            "case_studies_csv": str(cases_csv),
            "report_txt": str(report),
        },
    }
    (out_dir / "step14a_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("Step-14A complete")
    print(f"- rows: {len(all_rows)}")
    print(f"- failures: {len(fail_rows)}")
    print(f"- report: {report}")
    print(f"- error table: {error_csv}")
    print(f"- cases: {cases_json}")


if __name__ == "__main__":
    main()
