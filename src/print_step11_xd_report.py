#!/usr/bin/env python3
"""Print advisor-style Step-11 XD zero-shot report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-11 XD zero-shot report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/xd_violence_zero_shot/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return "NA"
    return f"{v:.{digits}f}"


def print_per_class(rows: List[Dict[str, Any]]) -> None:
    target_order = ["fighting", "shooting", "explosion", "abuse"]
    by_name = {str(r.get("class_name", "")).lower(): r for r in rows}
    for c in target_order:
        r = by_name.get(c, {})
        print(
            f"- {c}: precision={fmt(r.get('precision'))} "
            f"recall={fmt(r.get('recall'))} f1={fmt(r.get('f1'))}"
        )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    res_path = resolve(root, args.results)
    d = json.loads(res_path.read_text())

    ds = d.get("dataset_summary", {})
    fx = d.get("feature_extraction_summary", {})
    zm = d.get("zero_shot_metrics", {})
    per_class = d.get("overlap_per_class", [])
    cm = d.get("overlap_confusion_matrix", {})
    ex = d.get("excluded_from_overlap_f1", {})
    tv = d.get("transfer_verdict", {})
    q = d.get("qualitative_examples", {})
    succ = q.get("successes", [])
    fail = q.get("failures", [])

    print("1) XD dataset summary")
    print(f"- total videos evaluated: {ds.get('total_videos_evaluated', 0)}")
    print(f"- normal count: {ds.get('normal_count', 0)}")
    print(f"- violent count: {ds.get('violent_count', 0)}")
    oc = ds.get("overlap_class_counts", {})
    print(
        "- overlap-class counts: "
        f"fighting={oc.get('fighting', 0)}, shooting={oc.get('shooting', 0)}, "
        f"explosion={oc.get('explosion', 0)}, abuse={oc.get('abuse', 0)}"
    )
    print(f"- riot count: {ds.get('riot_count', 0)}")

    print("\n2) Feature extraction summary")
    print(f"- total extracted: {fx.get('total_extracted', 0)}")
    print(f"- total failed: {fx.get('total_failed', 0)}")
    seg = fx.get("segment_stats", {})
    print(
        f"- segments min/max/mean: {seg.get('min', 'NA')} / "
        f"{seg.get('max', 'NA')} / {fmt(seg.get('mean'))}"
    )
    conf = fx.get("extractor_match_confirmation", {})
    print(
        "- extractor settings matched UCF: "
        f"frozen_i3d={conf.get('frozen_i3d')} segment_length={conf.get('segment_length')} "
        f"overlap={conf.get('overlap')} tail_rule={conf.get('tail_rule')}"
    )

    print("\n3) Zero-shot metrics")
    print(f"- AUC: {fmt(zm.get('video_auc'))}")
    print(f"- AP: {fmt(zm.get('video_ap'))}")
    print(f"- 4-class macro-F1: {fmt(zm.get('overlap_macro_f1'))}")
    print(f"- 4-class weighted-F1: {fmt(zm.get('overlap_weighted_f1'))}")

    print("\n4) Per-class overlap results")
    print_per_class(per_class)

    print("\n5) Confusion matrix (overlap classes only)")
    labels = cm.get("labels", [])
    print(f"- labels: {labels}")
    print(f"- matrix: {cm.get('matrix', [])}")
    print(
        f"- riot videos excluded from overlap F1: "
        f"{ex.get('excluded_count', 0)} (class={ex.get('excluded_class', 'NA')})"
    )

    print("\n6) Transfer verdict")
    print(
        f"- XD AUC / UCF AUC: {fmt(tv.get('xd_auc'))} / {fmt(tv.get('ucf_reference_auc'))} "
        f"(ratio={fmt(tv.get('ratio_xd_over_ucf'))})"
    )
    print(f"- meets 75% rule: {tv.get('passes_75_percent_rule', False)}")
    print(
        f"- strongest overlap class: {tv.get('strongest_overlap_class', 'NA')}, "
        f"weakest overlap class: {tv.get('weakest_overlap_class', 'NA')}"
    )

    print("\n7) Five qualitative examples")
    print("- strong successes (3):")
    for s in succ[:3]:
        print(
            f"  - {s.get('video_id')} | gt={s.get('gt_label')} "
            f"| pred={s.get('predicted_label')} "
            f"| anomaly_score={fmt(s.get('binary_anomaly_score'))}"
        )
    print("- failures (2):")
    for f in fail[:2]:
        print(
            f"  - {f.get('video_id')} | gt={f.get('gt_label')} "
            f"| pred={f.get('predicted_label')} "
            f"| anomaly_score={fmt(f.get('binary_anomaly_score'))}"
        )


if __name__ == "__main__":
    main()

