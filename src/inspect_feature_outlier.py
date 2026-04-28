#!/usr/bin/env python3
"""Inspect largest-segment feature entry for consistency checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect max-segment feature outlier")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--video-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    p.add_argument("--segment-len", type=int, default=16)
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    master_path = resolve(root, args.video_manifest)
    feature_path = resolve(root, args.feature_manifest)

    master_rows = read_csv(master_path)
    feat_rows = [r for r in read_csv(feature_path) if r.get("status") == "ok"]
    if not feat_rows:
        raise SystemExit("No successful feature rows found.")

    row = max(feat_rows, key=lambda r: int(float(r.get("num_segments", "0"))))
    video_id = row["video_id"]
    num_segments_manifest = int(float(row["num_segments"]))

    master_map = {r["video_id"]: r for r in master_rows}
    m = master_map.get(video_id)

    npz_path = resolve(root, Path(row["feature_path"]))
    with np.load(npz_path, allow_pickle=True) as data:
        feats = data["features"]
        starts = data["segment_start_frames"]
        ends = data["segment_end_frames"]

    segments_file, feature_dim = int(feats.shape[0]), int(feats.shape[1])

    out = {
        "video_id": video_id,
        "split": row.get("split"),
        "category_label": row.get("category_label"),
        "feature_path": row.get("feature_path"),
        "num_segments_manifest": num_segments_manifest,
        "num_segments_in_file": segments_file,
        "feature_dim": feature_dim,
        "segment_ranges_in_file": {
            "starts_len": int(starts.shape[0]),
            "ends_len": int(ends.shape[0]),
            "first_range": [int(starts[0]), int(ends[0])] if len(starts) > 0 else None,
            "last_range": [int(starts[-1]), int(ends[-1])] if len(starts) > 0 else None,
        },
    }

    if m is not None:
        num_frames = int(float(m["num_frames"]))
        fps = float(m["fps"])
        duration_sec = float(m["duration_sec"])
        expected_floor = num_frames // args.segment_len
        covered_frames = segments_file * args.segment_len
        covered_sec = covered_frames / fps if fps > 0 else None

        out["video_manifest"] = {
            "video_path": m.get("video_path"),
            "num_frames": num_frames,
            "fps": fps,
            "duration_sec": duration_sec,
        }
        out["segment_consistency"] = {
            "segment_len": args.segment_len,
            "expected_floor_segments": int(expected_floor),
            "manifest_matches_floor": bool(num_segments_manifest == expected_floor),
            "file_matches_manifest": bool(segments_file == num_segments_manifest),
            "covered_frames_by_segments": int(covered_frames),
            "covered_sec_by_segments": covered_sec,
            "duration_gap_sec": None if covered_sec is None else float(duration_sec - covered_sec),
            "approx_expected_sec_from_frames": float(num_frames / fps) if fps > 0 else None,
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
