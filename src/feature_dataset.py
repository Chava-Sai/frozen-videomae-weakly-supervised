#!/usr/bin/env python3
"""Dataset utilities for cached I3D feature sequences."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureSample:
    features: torch.Tensor
    binary_label: torch.Tensor
    category_label: str
    category_id: torch.Tensor
    video_id: str
    split: str
    segment_start_frames: torch.Tensor
    segment_end_frames: torch.Tensor
    selected_segment_indices: torch.Tensor
    original_num_segments: torch.Tensor


def _uniform_indices(length: int, target: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be > 0")
    if target <= 0:
        raise ValueError("target must be > 0")

    if length >= target:
        return np.linspace(0, length - 1, num=target, dtype=np.int64)

    idx = np.arange(length, dtype=np.int64)
    pad = np.full((target - length,), fill_value=length - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


class FeatureSequenceDataset(Dataset):
    """Loads cached feature sequences from manifest rows.

    Collation strategy (Option B): fixed number of segments per video.
    - If T >= target_segments: uniform subsample to target_segments.
    - If T < target_segments: repeat last segment until target_segments.
    """

    def __init__(
        self,
        manifest_csv: Path,
        project_root: Path,
        split: Optional[str] = None,
        target_segments: int = 32,
        require_ok_status: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.target_segments = int(target_segments)

        rows = list(csv.DictReader(Path(manifest_csv).open()))
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if require_ok_status:
            rows = [r for r in rows if r.get("status") == "ok"]

        categories = sorted({r["category_label"] for r in rows})
        self.category_to_id: Dict[str, int] = {c: i for i, c in enumerate(categories)}

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> FeatureSample:
        row = self.rows[idx]
        feat_path = (self.project_root / row["feature_path"]).resolve()

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        with np.load(feat_path, allow_pickle=True) as data:
            feats = data["features"].astype(np.float32)
            starts = data["segment_start_frames"].astype(np.int64)
            ends = data["segment_end_frames"].astype(np.int64)

        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features for {row['video_id']}, got {feats.shape}")

        t = feats.shape[0]
        idxs = _uniform_indices(t, self.target_segments)

        feats_sel = feats[idxs]
        starts_sel = starts[idxs]
        ends_sel = ends[idxs]

        category = row["category_label"]
        cat_id = self.category_to_id[category]

        return FeatureSample(
            features=torch.from_numpy(feats_sel),
            binary_label=torch.tensor(int(row["binary_label"]), dtype=torch.long),
            category_label=category,
            category_id=torch.tensor(cat_id, dtype=torch.long),
            video_id=row["video_id"],
            split=row["split"],
            segment_start_frames=torch.from_numpy(starts_sel),
            segment_end_frames=torch.from_numpy(ends_sel),
            selected_segment_indices=torch.from_numpy(idxs.astype(np.int64)),
            original_num_segments=torch.tensor(int(row["num_segments"]), dtype=torch.long),
        )


def fixed_segments_collate(batch: Sequence[FeatureSample]) -> Dict[str, object]:
    features = torch.stack([x.features for x in batch], dim=0)
    binary_labels = torch.stack([x.binary_label for x in batch], dim=0)
    category_ids = torch.stack([x.category_id for x in batch], dim=0)

    out = {
        "features": features,
        "binary_labels": binary_labels,
        "category_ids": category_ids,
        "category_labels": [x.category_label for x in batch],
        "video_ids": [x.video_id for x in batch],
        "splits": [x.split for x in batch],
        "segment_start_frames": torch.stack([x.segment_start_frames for x in batch], dim=0),
        "segment_end_frames": torch.stack([x.segment_end_frames for x in batch], dim=0),
        "selected_segment_indices": torch.stack([x.selected_segment_indices for x in batch], dim=0),
        "original_num_segments": torch.stack([x.original_num_segments for x in batch], dim=0),
    }
    return out
