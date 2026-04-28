#!/usr/bin/env python3
"""Print advisor-format Step-9 report from results_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-9 calibration report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/step9_step7_calibration/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    d = json.loads(path.read_text())

    notes = d.get("notes", {})
    sweep = d.get("validation_sweep_summary", {})
    top10 = sweep.get("top10", [])
    best = d.get("best_setting", {})
    test = d.get("test_metrics_chosen_setting", {})
    bnd = d.get("boundary_ablation", {})
    q = d.get("qualitative_examples", {})

    print("1) Validation sweep summary")
    print(f"- val temporal available: {notes.get('val_temporal_available')}")
    print(f"- selection rule applied: {notes.get('selection_rule_applied')}")
    print(f"- total settings: {sweep.get('total_settings')}")
    print("- top 10 settings:")
    for i, r in enumerate(top10, 1):
        print(
            f"  {i}. thr={r.get('threshold')} sw={r.get('smooth_window')} min_len={r.get('min_event_len')} "
            f"gap={r.get('merge_gap')} br={r.get('boundary_radius')} bnd={r.get('boundary_refine')} | "
            f"val mAP@0.5={r.get('val_map_05')} mAP@0.3={r.get('val_map_03')} "
            f"val_bin_f1={r.get('val_binary_f1')} pred_events={r.get('val_pred_events')}"
        )
    print()

    print("2) Best chosen setting")
    print(f"- threshold: {best.get('threshold')}")
    print(f"- smooth_window: {best.get('smooth_window')}")
    print(f"- min_event_len: {best.get('min_event_len')}")
    print(f"- merge_gap: {best.get('merge_gap')}")
    print(f"- boundary_radius: {best.get('boundary_radius')}")
    print(f"- boundary on/off: {best.get('boundary_refine')}")
    print()

    print("3) Test metrics with chosen setting")
    bn = test.get("binary", {})
    cls = test.get("classification", {})
    print(f"- AUC: {bn.get('auc')}")
    print(f"- AP: {bn.get('ap')}")
    print(f"- macro-F1: {cls.get('macro_f1')}")
    print(f"- weighted-F1: {cls.get('weighted_f1')}")
    print(f"- mAP@0.3: {test.get('mAP@0.3')}")
    print(f"- mAP@0.5: {test.get('mAP@0.5')}")
    print(f"- mAP@0.7: {test.get('mAP@0.7')}")
    print(f"- predicted events vs GT events: {test.get('pred_events')} vs {test.get('gt_events')}")
    print()

    print("4) Boundary ablation mini-result")
    print(f"- with boundary refinement: {bnd.get('with_boundary_refinement')}")
    print(f"- without boundary refinement: {bnd.get('without_boundary_refinement')}")
    print()

    print("5) Three videos")
    print(f"- strong success: {q.get('strong_success')}")
    print(f"- partial improvement: {q.get('partial_improvement')}")
    print(f"- failure: {q.get('failure')}")


if __name__ == "__main__":
    main()
