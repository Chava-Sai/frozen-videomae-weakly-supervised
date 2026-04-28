#!/usr/bin/env python3
"""Print advisor-format Step-8 progressive training summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-8 progressive report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/rtfm_progressive/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    d = json.loads(path.read_text())

    print("1) Progressive training schedule")
    for s in d.get("progressive_schedule", []):
        print(
            f"- stage {s.get('stage_id')} {s.get('stage_name')} | "
            f"epochs={s.get('epoch_range')} | trainable={s.get('trainable_modules')} | "
            f"use_classifier={s.get('use_classifier')} use_trn={s.get('use_trn')} use_boundary={s.get('use_boundary')}"
        )
    print()

    print("2) Validation-selected inference settings")
    cal = d.get("inference_calibration_val_only", {})
    sel = cal.get("selected_settings", {})
    print(f"- threshold: {sel.get('threshold')}")
    print(f"- smoothing window: {sel.get('smooth_window')}")
    print(f"- min event length: {sel.get('min_event_len')}")
    print(f"- merge gap: {sel.get('merge_gap')}")
    print(f"- boundary peak rule: {cal.get('boundary_peak_rule')}")
    print(f"- boundary radius: {sel.get('boundary_radius')}")
    print()

    print("3) Updated test metrics")
    upd = d.get("updated_metrics", {})
    bn = upd.get("binary", {})
    cls = upd.get("classification", {})
    loc = d.get("temporal_localization", {}).get("tiou", {})
    print(f"- AUC: {bn.get('auc')}")
    print(f"- AP: {bn.get('ap')}")
    print(f"- macro-F1: {cls.get('macro_f1')}")
    print(f"- weighted-F1: {cls.get('weighted_f1')}")
    print(f"- mAP@0.3: {loc.get('0.3', {}).get('mAP')}")
    print(f"- mAP@0.5: {loc.get('0.5', {}).get('mAP')}")
    print(f"- mAP@0.7: {loc.get('0.7', {}).get('mAP')}")
    print()

    print("4) GT event count vs predicted event count")
    ec = d.get("event_count_summary", {})
    print(f"- GT events: {ec.get('gt_event_count')}")
    print(f"- Predicted events: {ec.get('pred_event_count')}")
    print()

    print("5) Before/after table")
    print("Model | AUC | AP | Macro-F1 | Weighted-F1 | mAP@0.3 | mAP@0.5 | mAP@0.7")
    for r in d.get("before_after_table", []):
        print(
            f"{r.get('model')} | {r.get('auc')} | {r.get('ap')} | {r.get('macro_f1')} | "
            f"{r.get('weighted_f1')} | {r.get('mAP@0.3')} | {r.get('mAP@0.5')} | {r.get('mAP@0.7')}"
        )
    print()

    print("6) Qualitative localization")
    improved = d.get("qualitative_improved_videos", [])
    if improved:
        for i, r in enumerate(improved, 1):
            print(
                f"- improved {i}: {r.get('video_id')} | gt={r.get('ground_truth_class')} | "
                f"step7_iou={r.get('step7_best_iou')} -> step8_iou={r.get('step8_best_iou')} | "
                f"improvement={r.get('improvement')}"
            )
    else:
        print("- improved: none found (step7 comparison file unavailable or no IoU gains)")
    print(f"- one failure: {d.get('one_failure_case')}")


if __name__ == "__main__":
    main()
