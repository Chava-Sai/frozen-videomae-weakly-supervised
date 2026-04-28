#!/usr/bin/env python3
"""Summarize full I3D feature extraction outputs and integrity checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize extracted I3D features")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--video-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/ucf_crime/features/i3d_kinetics400_16f"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/ucf_crime/manifests/i3d_full_extraction_report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/ucf_crime/manifests/i3d_full_extraction_report.json"),
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=8,
        help="Mark videos with <= threshold segments as very small",
    )
    return parser.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": -1, "max": -1, "mean": -1, "median": -1}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
    }


def disk_usage_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.2f} {u}"
    return f"{x:.2f} PB"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    video_manifest = resolve(root, args.video_manifest)
    feature_manifest = resolve(root, args.feature_manifest)
    feature_root = resolve(root, args.feature_root)
    out_report = resolve(root, args.output_report)
    out_json = resolve(root, args.output_json)

    master_rows = read_csv(video_manifest)
    feat_rows = read_csv(feature_manifest)

    total_master = len(master_rows)
    total_feat_rows = len(feat_rows)

    by_video_master = {r["video_id"]: r for r in master_rows}
    by_video_feat = {r["video_id"]: r for r in feat_rows}

    missing_manifest_rows = sorted(set(by_video_master.keys()) - set(by_video_feat.keys()))

    success_rows = [r for r in feat_rows if r.get("status") == "ok"]
    fail_rows = [r for r in feat_rows if r.get("status") != "ok"]

    success_by_split = Counter(r["split"] for r in success_rows)
    success_by_class = Counter(r["category_label"] for r in success_rows)

    segment_counts_overall: List[int] = []
    segment_by_split: Dict[str, List[int]] = defaultdict(list)

    missing_feature_paths: List[str] = []
    bad_feature_dims: List[Tuple[str, int]] = []
    shape_mismatches: List[str] = []

    for r in success_rows:
        video_id = r["video_id"]
        expected_num_segments = int(float(r["num_segments"]))
        feature_dim = int(float(r["feature_dim"]))
        fp = resolve(root, Path(r["feature_path"]))

        if not fp.exists():
            missing_feature_paths.append(r["feature_path"])
            continue

        if feature_dim != 2048:
            bad_feature_dims.append((video_id, feature_dim))

        with np.load(fp, allow_pickle=True) as data:
            feats = data["features"]
            starts = data["segment_start_frames"]
            ends = data["segment_end_frames"]

        if feats.ndim != 2:
            shape_mismatches.append(f"{video_id}: features ndim {feats.ndim}")
            continue

        t, d = int(feats.shape[0]), int(feats.shape[1])
        if d != 2048:
            shape_mismatches.append(f"{video_id}: feature dim in file {d}")

        if t != expected_num_segments:
            shape_mismatches.append(
                f"{video_id}: manifest num_segments={expected_num_segments} file_segments={t}"
            )

        if starts.shape[0] != t or ends.shape[0] != t:
            shape_mismatches.append(
                f"{video_id}: ranges mismatch starts={starts.shape[0]} ends={ends.shape[0]} t={t}"
            )

        segment_counts_overall.append(t)
        segment_by_split[r["split"]].append(t)

    very_small = [
        r["video_id"]
        for r in success_rows
        if int(float(r.get("num_segments", "0"))) <= args.small_threshold
    ]

    stats_overall = _stats(segment_counts_overall)
    stats_train = _stats(segment_by_split.get("train", []))
    stats_val = _stats(segment_by_split.get("val", []))
    stats_test = _stats(segment_by_split.get("test", []))

    disk_bytes = disk_usage_bytes(feature_root)

    report_obj = {
        "total_manifest_videos": total_master,
        "total_feature_manifest_rows": total_feat_rows,
        "missing_feature_manifest_rows": len(missing_manifest_rows),
        "total_successful_features": len(success_rows),
        "total_failed_features": len(fail_rows),
        "success_by_split": dict(success_by_split),
        "success_by_class": dict(success_by_class),
        "segment_stats_overall": stats_overall,
        "segment_stats_by_split": {
            "train": stats_train,
            "val": stats_val,
            "test": stats_test,
        },
        "missing_feature_files": len(missing_feature_paths),
        "bad_feature_dim_rows": len(bad_feature_dims),
        "shape_mismatches": len(shape_mismatches),
        "very_small_segment_videos": len(very_small),
        "feature_disk_usage_bytes": disk_bytes,
        "feature_disk_usage_human": human_bytes(disk_bytes),
        "failures": [
            {
                "video_id": r["video_id"],
                "split": r["split"],
                "category_label": r["category_label"],
                "reason": r.get("failure_reason", ""),
            }
            for r in fail_rows
        ],
        "missing_manifest_video_ids": missing_manifest_rows,
        "missing_feature_paths": missing_feature_paths,
        "bad_feature_dims": bad_feature_dims,
        "shape_mismatch_details": shape_mismatches,
        "very_small_video_ids": very_small,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report_obj, indent=2) + "\n")

    lines = [
        "# Full I3D Extraction Report",
        "",
        "## Counts",
        f"- Total videos in master manifest: **{total_master}**",
        f"- Total rows in feature manifest: **{total_feat_rows}**",
        f"- Missing rows in feature manifest (vs master): **{len(missing_manifest_rows)}**",
        f"- Successful features: **{len(success_rows)}**",
        f"- Failed features: **{len(fail_rows)}**",
        "",
        "## Success by split",
        f"- train: {success_by_split.get('train', 0)}",
        f"- val: {success_by_split.get('val', 0)}",
        f"- test: {success_by_split.get('test', 0)}",
        "",
        "## Success by class",
    ]

    for c in ["fighting", "shooting", "explosion", "robbery", "abuse", "normal"]:
        lines.append(f"- {c}: {success_by_class.get(c, 0)}")

    lines.extend(
        [
            "",
            "## Segment statistics overall",
            f"- min: {stats_overall['min']}",
            f"- max: {stats_overall['max']}",
            f"- mean: {stats_overall['mean']:.3f}" if stats_overall["mean"] >= 0 else "- mean: -1",
            f"- median: {stats_overall['median']:.3f}" if stats_overall["median"] >= 0 else "- median: -1",
            "",
            "## Segment statistics by split",
            f"- train: min={stats_train['min']} max={stats_train['max']} mean={stats_train['mean']:.3f} median={stats_train['median']:.3f}"
            if stats_train["mean"] >= 0
            else "- train: no data",
            f"- val: min={stats_val['min']} max={stats_val['max']} mean={stats_val['mean']:.3f} median={stats_val['median']:.3f}"
            if stats_val["mean"] >= 0
            else "- val: no data",
            f"- test: min={stats_test['min']} max={stats_test['max']} mean={stats_test['mean']:.3f} median={stats_test['median']:.3f}"
            if stats_test["mean"] >= 0
            else "- test: no data",
            "",
            "## Integrity checks",
            f"- Missing feature files referenced by manifest: **{len(missing_feature_paths)}**",
            f"- Rows with feature_dim != 2048: **{len(bad_feature_dims)}**",
            f"- Feature shape mismatches: **{len(shape_mismatches)}**",
            f"- Videos with <= {args.small_threshold} segments: **{len(very_small)}**",
            f"- Feature directory size: **{human_bytes(disk_bytes)}**",
        ]
    )

    lines.extend(["", "## Failures"])
    if fail_rows:
        for r in fail_rows:
            lines.append(
                f"- {r['video_id']} ({r['category_label']}/{r['split']}): {r.get('failure_reason', '')}"
            )
    else:
        lines.append("- None")

    out_report.write_text("\n".join(lines) + "\n")

    print(json.dumps(report_obj, indent=2))


if __name__ == "__main__":
    main()
