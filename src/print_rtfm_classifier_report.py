#!/usr/bin/env python3
"""Print advisor-format Step-5 summary from results_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-5 RTFM+Classifier report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/rtfm_classifier/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    d = json.loads(path.read_text())

    print("1) Classifier design")
    arch = d.get("architecture", {})
    print(f"- input: {arch.get('input_shape')}")
    print(f"- classifier layers: {arch.get('classifier_head')}")
    print(f"- number of classes: {len(arch.get('taxonomy', {}).get('class_names', []))}")
    print(f"- classifier input source: {arch.get('classifier_input')}")
    print(f"- pseudo-label rule: {arch.get('pseudo_label_selection_rule')}")
    print()

    print("2) Training setup")
    setup = d.get("training_setup", {})
    print(f"- initialization source: {setup.get('initialization_source')}")
    print(f"- losses: {setup.get('losses')}")
    print(f"- optimizer: {setup.get('optimizer')}")
    print(f"- learning rate: {setup.get('learning_rate')}")
    print(f"- batch size: {setup.get('batch_size')}")
    print(f"- epochs: {setup.get('epochs')}")
    print(f"- checkpoint rule: {setup.get('checkpoint_selection_rule')}")
    print()

    print("3) Validation and test metrics")
    test = d.get("test", {})
    cls = test.get("classification", {})
    bn = test.get("binary", {})
    print(f"- test macro-F1: {cls.get('macro_f1')}")
    print(f"- test weighted-F1: {cls.get('weighted_f1')}")
    print(f"- per-class precision/recall/F1: {cls.get('per_class')}")
    print(f"- class confusion matrix: {cls.get('confusion_matrix')}")
    print(f"- binary AUC/AP: {bn.get('auc')} / {bn.get('ap')}")
    print(f"- binary confusion@{bn.get('confusion_matrix_threshold')}: {bn.get('confusion_matrix')}")
    print()

    print("4) Pseudo-label sanity (5 training anomalies)")
    for r in d.get("pseudo_label_sanity_train", []):
        print(
            f"- {r['video_id']} | class={r['class']} | idx={r['selected_topk_segment_indices']} | "
            f"scores={r['selected_topk_anomaly_scores']}"
        )
    print()

    print("5) Qualitative class predictions (5 test anomalies)")
    for r in d.get("qualitative_test_class_predictions", []):
        print(
            f"- {r['video_id']} | gt={r['ground_truth_class']} | pred={r['predicted_class']} | "
            f"p={r['top_predicted_probability']} | top_segments={r['top_anomaly_segments']}"
        )


if __name__ == "__main__":
    main()
