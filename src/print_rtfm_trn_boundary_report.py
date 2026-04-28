#!/usr/bin/env python3
"""Print advisor-format Step-7 summary from results_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-7 TRN+Boundary report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/rtfm_trn_boundary/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    d = json.loads(path.read_text())

    arch = d.get("architecture", {})
    bnd = arch.get("boundary_head", {})
    setup = d.get("training_setup", {})
    loc = d.get("temporal_localization", {})
    upd = d.get("updated_metrics", {})
    bn = upd.get("binary", {})
    cls = upd.get("classification", {})

    print("1) Boundary head design")
    print(f"- input dim: {bnd.get('input_dim')}")
    print(f"- input definition: {bnd.get('input_definition')}")
    print(f"- boundary head layers: {bnd.get('layers')}")
    print(f"- activation: {bnd.get('activation')}")
    print(f"- per-edge prediction: {bnd.get('prediction')}")
    print()

    print("2) Training setup")
    print(f"- initialization checkpoint: {setup.get('checkpoint_initialization')}")
    print(f"- full loss formula: {setup.get('loss_formula')}")
    print(f"- loss weights/details: {setup.get('losses')}")
    print(f"- epochs: {setup.get('epochs')}")
    print(f"- optimizer: {setup.get('optimizer')}")
    print(f"- learning rate: {setup.get('learning_rate')}")
    print(f"- checkpoint rule: {setup.get('checkpoint_rule')}")
    print()

    print("3) Temporal localization metrics")
    tiou = loc.get("tiou", {})
    print(f"- mAP@tIoU=0.3: {tiou.get('0.3', {}).get('mAP')}")
    print(f"- mAP@tIoU=0.5: {tiou.get('0.5', {}).get('mAP')}")
    print(f"- mAP@tIoU=0.7: {tiou.get('0.7', {}).get('mAP')}")
    print(f"- GT events: {loc.get('gt_event_count')}")
    print(f"- Pred events: {loc.get('pred_event_count')}")
    print()

    print("4) Updated classification/detection metrics")
    print(f"- binary AUC/AP: {bn.get('auc')} / {bn.get('ap')}")
    print(f"- macro-F1: {cls.get('macro_f1')}")
    print(f"- weighted-F1: {cls.get('weighted_f1')}")
    print()

    print("5) Boundary qualitative examples")
    for r in d.get("boundary_qualitative_examples", []):
        print(
            f"- {r.get('video_id')} | gt={r.get('ground_truth_class')} | "
            f"before={r.get('predicted_event_spans_before_refinement')} | "
            f"after={r.get('predicted_event_spans_after_refinement')} | "
            f"boundary_peaks={r.get('top_boundary_peaks')}"
        )
    print()

    print("6) One failure case")
    print(d.get("failure_case"))
    print()

    print("7) One success case")
    print(d.get("success_case"))


if __name__ == "__main__":
    main()
