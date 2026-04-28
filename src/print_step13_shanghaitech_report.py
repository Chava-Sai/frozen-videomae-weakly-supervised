#!/usr/bin/env python3
"""Render Step-13 ShanghaiTech robustness report from JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-13 ShanghaiTech report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/shanghaitech_robustness/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def fmt(x: object, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    results_path = resolve(root, args.results)

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    d = json.loads(results_path.read_text())
    ds = d.get("dataset_summary", {})
    fs = d.get("feature_extraction_summary", {})
    seg = fs.get("segment_stats", {})
    rm = d.get("robustness_metrics", {})
    ep = d.get("error_profile", {})
    q = d.get("qualitative_examples", {})

    print("1) ShanghaiTech dataset summary")
    print(f"- total videos evaluated: {ds.get('total_videos_evaluated')}")
    print(f"- normal count: {ds.get('normal_count')}")
    print(f"- anomaly count: {ds.get('anomaly_count')}")
    print(f"- split used: {ds.get('split_used')}")

    print("\n2) Feature extraction summary")
    print(f"- total extracted: {fs.get('total_extracted')}")
    print(f"- total failed: {fs.get('total_failed')}")
    print(f"- segments min/max/mean: {seg.get('min')} / {seg.get('max')} / {fmt(seg.get('mean'))}")
    match = fs.get("extractor_match_confirmation", {})
    print(
        "- extractor settings matched UCF/XD/RWF: "
        f"frozen_i3d={match.get('frozen_i3d')} segment_length={match.get('segment_length')} "
        f"overlap={match.get('overlap')} tail_rule={match.get('tail_rule')}"
    )

    print("\n3) Robustness metrics")
    print(f"- AUC: {fmt(rm.get('auc'))}")
    print(f"- AP: {fmt(rm.get('ap'))}")

    print("\n4) Error profile")
    print(f"- false positives: {ep.get('false_positives')}")
    print(f"- false negatives: {ep.get('false_negatives')}")
    print(f"- confusion matrix labels: {ep.get('confusion_matrix_labels')}")
    print(f"- confusion matrix: {ep.get('confusion_matrix')}")
    print(f"- failure-mode guess: {ep.get('failure_mode_guess')}")

    print("\n5) Five qualitative examples")
    succ = q.get("successes", [])
    fail = q.get("failures", [])
    print("- strong successes (3):")
    for ex in succ[:3]:
        print(
            f"  - {ex.get('video_id')} | gt={ex.get('gt_label')} | pred={ex.get('predicted_label')} | "
            f"anomaly_score={fmt(ex.get('anomaly_score'))} | note={ex.get('note')}"
        )
    print("- failures (2):")
    for ex in fail[:2]:
        print(
            f"  - {ex.get('video_id')} | gt={ex.get('gt_label')} | pred={ex.get('predicted_label')} | "
            f"anomaly_score={fmt(ex.get('anomaly_score'))} | note={ex.get('note')}"
        )


if __name__ == "__main__":
    main()

