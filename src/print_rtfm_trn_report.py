#!/usr/bin/env python3
"""Print Step-6 advisor-format summary from results_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-6 TRN report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/rtfm_trn/results_summary.json"),
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    path = resolve(args.project_root.resolve(), args.results)
    d = json.loads(path.read_text())

    arch = d.get("architecture", {})
    setup = d.get("training_setup", {})
    test = d.get("test", {})
    cls = test.get("classification", {})
    bn = test.get("binary", {})

    print("1) TRN architecture")
    print(f"- projection dim: {arch.get('projection_dim')}")
    print(f"- positional encoding: {arch.get('positional_encoding')}")
    print(f"- encoder layers: {arch.get('trn_layers')}")
    print(f"- heads: {arch.get('trn_heads')}")
    print(f"- FFN size: {arch.get('trn_ffn_dim')}")
    print(f"- dropout: {arch.get('trn_dropout')}")
    print(f"- anomaly head attach: {arch.get('anomaly_head_attach')}")
    print(f"- classifier head attach: {arch.get('classifier_head_attach')}")
    print()

    print("2) Training setup")
    print(f"- checkpoint initialization: {setup.get('checkpoint_initialization')}")
    print(f"- losses: {setup.get('losses')}")
    print(f"- epochs: {setup.get('epochs')}")
    print(f"- optimizer: {setup.get('optimizer')}")
    print(f"- learning rate: {setup.get('learning_rate')}")
    print(f"- checkpoint rule: {setup.get('checkpoint_rule')}")
    print()

    print("3) Metrics")
    print(f"- test macro-F1: {cls.get('macro_f1')}")
    print(f"- test weighted-F1: {cls.get('weighted_f1')}")
    print(f"- test binary AUC/AP: {bn.get('auc')} / {bn.get('ap')}")
    print(f"- per-class precision/recall/F1: {cls.get('per_class')}")
    print(f"- confusion matrix: {cls.get('confusion_matrix')}")
    print()

    print("4) Comparison table")
    print("Model | AUC | AP | Macro-F1 | Weighted-F1")
    for r in d.get("comparison_table", []):
        print(f"{r.get('model')} | {r.get('auc')} | {r.get('ap')} | {r.get('macro_f1')} | {r.get('weighted_f1')}")
    print()

    print("5) Refined-score sanity")
    for r in d.get("refined_score_sanity", []):
        print(
            f"- {r['video_id']} ({r['class']}) | idx={r['top_refined_segment_indices']} | "
            f"scores={r['top_refined_scores']} | note={r['cluster_or_smooth_note']}"
        )
    print()

    print("6) Attention sanity")
    print(d.get("attention_sanity", {}))


if __name__ == "__main__":
    main()
