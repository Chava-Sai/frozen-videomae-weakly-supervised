#!/usr/bin/env python3
"""Print Step-4 report fields from results_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print compact Step-4 RTFM report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/rtfm_baseline/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    data = json.loads(path.read_text())

    arch = data.get("architecture", {})
    setup = data.get("training_setup", {})
    test = data.get("test", {})

    print("Model architecture summary")
    print(f"- input_shape: {arch.get('input_shape')}")
    print(f"- model: {arch.get('model')}")
    print(f"- feature_proj: {arch.get('feature_proj')}")
    print(f"- segment_head: {arch.get('segment_head')}")
    print(f"- temporal_sampling_rule: {arch.get('temporal_sampling_rule')}")
    print(f"- aggregation_rule: {arch.get('aggregation_rule')}")
    print()

    print("Training setup")
    print(f"- loss: {setup.get('loss')}")
    print(f"- optimizer: {setup.get('optimizer')}")
    print(f"- learning_rate: {setup.get('learning_rate')}")
    print(f"- batch_size: {setup.get('batch_size')}")
    print(f"- epochs: {setup.get('epochs')}")
    print(f"- checkpoint_selection: {setup.get('checkpoint_selection')}")
    print(f"- best_epoch: {data.get('best_epoch')}")
    print(f"- val_best: {data.get('val_best')}")
    print()

    print("Test results")
    print(f"- AUC: {test.get('auc')}")
    print(f"- AP: {test.get('ap')}")
    print(f"- confusion_matrix@{test.get('confusion_matrix_threshold')}: {test.get('confusion_matrix')}")
    print("- normal_score_examples:")
    for r in test.get("normal_score_examples", []):
        print(f"  - {r['video_id']} ({r['category_label']}): {r['score']:.6f}")
    print("- anomaly_score_examples:")
    for r in test.get("anomaly_score_examples", []):
        print(f"  - {r['video_id']} ({r['category_label']}): {r['score']:.6f}")
    print()

    print("Qualitative top segments")
    for r in data.get("qualitative_top_segments", []):
        print(
            f"- {r['video_id']} ({r['category_label']}), num_segments={r['num_segments']}, "
            f"top_indices={r['top_segment_indices']}, top_scores={r['top_segment_scores']}"
        )


if __name__ == "__main__":
    main()
