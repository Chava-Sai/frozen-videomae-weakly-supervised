#!/usr/bin/env python3
"""Sanity-check batch loading from cached I3D features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from feature_dataset import FeatureSequenceDataset, fixed_segments_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature loader sanity check")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-segments", type=int, default=32)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/ucf_crime/manifests/feature_loader_sanity.json"),
    )
    return parser.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    feature_manifest = resolve(root, args.feature_manifest)
    out_json = resolve(root, args.output_json)

    ds = FeatureSequenceDataset(
        manifest_csv=feature_manifest,
        project_root=root,
        split=args.split,
        target_segments=args.target_segments,
        require_ok_status=True,
    )

    if len(ds) == 0:
        raise RuntimeError(f"No rows available for split={args.split} in {feature_manifest}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=fixed_segments_collate,
    )

    batch = next(iter(loader))

    summary = {
        "split": args.split,
        "dataset_size": len(ds),
        "batch_size": int(batch["features"].shape[0]),
        "feature_batch_shape": list(batch["features"].shape),
        "binary_labels_shape": list(batch["binary_labels"].shape),
        "category_ids_shape": list(batch["category_ids"].shape),
        "category_labels_shape": [len(batch["category_labels"])],
        "video_ids_preview": batch["video_ids"][:5],
        "unique_category_labels_in_batch": sorted(set(batch["category_labels"])),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
