#!/usr/bin/env python3
"""Build RWF-2000 manifest for Step-12 fight validation (evaluation only)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare RWF-2000 manifest")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--raw-root", type=Path, default=Path("data/rwf_2000/raw_videos"))
    p.add_argument(
        "--output-master",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_master.csv"),
    )
    p.add_argument(
        "--output-eval",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_eval.csv"),
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_manifest_report.json"),
    )
    p.add_argument("--dataset-name", type=str, default="rwf_2000")
    p.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="val")
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def list_videos(raw_root: Path) -> List[Path]:
    videos: List[Path] = []
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()
    return videos


def infer_split(path_parts_lower: List[str]) -> str:
    for token in path_parts_lower:
        if token in {"train", "val", "test"}:
            return token
    return "val"


def infer_category(path_parts_lower: List[str]) -> Optional[str]:
    for token in path_parts_lower:
        if token in {"nonfight", "non_fight", "non-fight", "normal"}:
            return "normal"
    for token in path_parts_lower:
        if token in {"fight", "fighting"}:
            return "fighting"
    return None


def safe_video_stats(video_path: Path) -> Tuple[bool, int, float, float, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0, 0.0, 0.0, "cv2_open_failed"

    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = float(num_frames / fps) if fps > 0 else 0.0
    finally:
        cap.release()

    if num_frames <= 0:
        return False, 0, fps, duration, "empty_or_bad_frame_count"
    return True, num_frames, fps, duration, ""


def make_unique_video_id(base_id: str, seen: Dict[str, int]) -> str:
    if base_id not in seen:
        seen[base_id] = 1
        return base_id
    seen[base_id] += 1
    return f"{base_id}__dup{seen[base_id]}"


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()

    root = args.project_root.resolve()
    raw_root = resolve(root, args.raw_root)
    out_master = resolve(root, args.output_master)
    out_eval = resolve(root, args.output_eval)
    out_report = resolve(root, args.output_report)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    videos = list_videos(raw_root)
    seen_ids: Dict[str, int] = {}

    rows: List[Dict[str, object]] = []
    unknown_paths: List[str] = []
    bad_rows: List[Dict[str, object]] = []

    for vp in videos:
        rel = vp.relative_to(root).as_posix() if str(vp).startswith(str(root)) else vp.as_posix()
        parts_lower = [x.lower() for x in vp.parts]
        split = infer_split(parts_lower)
        category = infer_category(parts_lower)
        if category is None:
            unknown_paths.append(rel)
            continue

        ok, num_frames, fps, duration, reason = safe_video_stats(vp)
        if not ok:
            bad_rows.append({"video_path": rel, "reason": reason})
            continue

        base_video_id = vp.stem
        video_id = make_unique_video_id(base_video_id, seen_ids)
        binary_label = 1 if category == "fighting" else 0

        rows.append(
            {
                "video_id": video_id,
                "video_path": rel,
                "split": split,
                "binary_label": binary_label,
                "category_label": category,
                "dataset": args.dataset_name,
                "num_frames": num_frames,
                "fps": f"{fps:.6f}",
                "duration_sec": f"{duration:.6f}",
                "has_temporal_annotation": False,
                "temporal_annotation_path": "",
            }
        )

    eval_rows = rows if args.eval_split == "all" else [r for r in rows if r["split"] == args.eval_split]
    if not eval_rows:
        eval_rows = list(rows)

    fieldnames = [
        "video_id",
        "video_path",
        "split",
        "binary_label",
        "category_label",
        "dataset",
        "num_frames",
        "fps",
        "duration_sec",
        "has_temporal_annotation",
        "temporal_annotation_path",
    ]
    write_csv(out_master, rows, fieldnames)
    write_csv(out_eval, eval_rows, fieldnames)

    class_counts = Counter(str(r["category_label"]) for r in rows)
    split_counts = Counter(str(r["split"]) for r in rows)

    report = {
        "raw_root": str(raw_root),
        "total_video_files_found": len(videos),
        "total_manifest_rows": len(rows),
        "total_eval_rows": len(eval_rows),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "unknown_paths_skipped": len(unknown_paths),
        "unknown_paths_preview": unknown_paths[:100],
        "invalid_or_unreadable_videos_count": len(bad_rows),
        "invalid_or_unreadable_videos_preview": bad_rows[:100],
        "output_master": str(out_master),
        "output_eval": str(out_eval),
        "eval_split": args.eval_split,
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")

    print("RWF manifest prep complete")
    print(f"- total files scanned: {len(videos)}")
    print(f"- manifest rows: {len(rows)}")
    print(f"- eval rows: {len(eval_rows)}")
    print(f"- unknown paths skipped: {len(unknown_paths)}")
    print(f"- invalid/corrupt skipped: {len(bad_rows)}")
    print(f"- class counts: {dict(class_counts)}")
    print(f"- saved: {out_master}")
    print(f"- report: {out_report}")


if __name__ == "__main__":
    main()

