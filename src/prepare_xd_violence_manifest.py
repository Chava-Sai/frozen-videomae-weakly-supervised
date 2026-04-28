#!/usr/bin/env python3
"""Build XD-Violence manifest for zero-shot evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare XD-Violence manifest")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--raw-root", type=Path, default=Path("data/xd_violence/raw_videos"))
    p.add_argument(
        "--output-master",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_master.csv"),
    )
    p.add_argument(
        "--output-eval",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_zero_shot_eval.csv"),
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_manifest_report.json"),
    )
    p.add_argument("--dataset-name", type=str, default="xd_violence")
    p.add_argument("--default-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--skip-unknown", action="store_true", default=True)
    p.add_argument("--include-unknown", dest="skip_unknown", action="store_false")
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


def infer_split(path_str: str, default_split: str) -> str:
    s = path_str.lower()
    # XD-Violence official structure has test videos under "test_videos"
    # and train videos under numbered buckets like "1-1004", "1005-2004", etc.
    if "test_videos" in s:
        return "test"
    if "/train/" in s or "_train" in s:
        return "train"
    if "/val/" in s or "/valid/" in s or "_val" in s or "_valid" in s:
        return "val"
    if "/test/" in s or "_test" in s:
        return "test"
    if re.search(r"/\d{1,4}-\d{1,4}/", s):
        return "train"
    return default_split


def infer_category(path_str: str) -> Optional[str]:
    s = path_str.lower()

    # Prefer official XD label code embedded in filename.
    # Mapping from authors:
    # A=normal, B1=fighting, B2=shooting, B4=riot, B5=abuse, B6=car_accident, G=explosion
    m = re.search(r"label_([a-z0-9_\-]+)", s)
    if m:
        raw_codes = m.group(1).upper()
        parts = [p for p in re.split(r"[_\-]+", raw_codes) if p]
        code_to_cat = {
            "A": "normal",
            "B1": "fighting",
            "B2": "shooting",
            "B4": "riot",
            "B5": "abuse",
            "B6": "car_accident",
            "G": "explosion",
        }
        for code in parts:
            if code in code_to_cat:
                return code_to_cat[code]

    # Prefer explicit normal/non-violence before generic matching.
    if re.search(r"non[-_ ]?violence", s):
        return "normal"
    if re.search(r"(^|[/_\- ])normal([/_\- ]|$)", s):
        return "normal"

    if "fighting" in s or re.search(r"(^|[/_\- ])fight([/_\- ]|$)", s):
        return "fighting"
    if "shooting" in s or re.search(r"(^|[/_\- ])shoot([/_\- ]|$)", s):
        return "shooting"
    if "explosion" in s or "explosive" in s or "blast" in s:
        return "explosion"
    if "abuse" in s:
        return "abuse"
    if "riot" in s:
        return "riot"

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
    unknown_rows: List[str] = []
    bad_rows: List[Dict[str, object]] = []

    for vp in videos:
        rel = vp.relative_to(root).as_posix() if str(vp).startswith(str(root)) else vp.as_posix()
        category = infer_category(rel)
        if category is None:
            unknown_rows.append(rel)
            if args.skip_unknown:
                continue
            category = "unknown"

        ok, num_frames, fps, duration, reason = safe_video_stats(vp)
        if not ok:
            bad_rows.append({"video_path": rel, "reason": reason})
            continue

        split = infer_split(rel, args.default_split)
        base_video_id = vp.stem
        video_id = make_unique_video_id(base_video_id, seen_ids)

        binary_label = 0 if category == "normal" else 1

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

    # If no explicit test split exists, use all rows as eval rows.
    eval_rows = [r for r in rows if r["split"] == "test"]
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
        "unknown_label_paths_count": len(unknown_rows),
        "unknown_label_paths_preview": unknown_rows[:100],
        "invalid_or_unreadable_videos_count": len(bad_rows),
        "invalid_or_unreadable_videos_preview": bad_rows[:100],
        "output_master": str(out_master),
        "output_eval": str(out_eval),
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")

    print("XD manifest prep complete")
    print(f"- total files scanned: {len(videos)}")
    print(f"- manifest rows: {len(rows)}")
    print(f"- eval rows: {len(eval_rows)}")
    print(f"- unknown labels skipped: {len(unknown_rows) if args.skip_unknown else 0}")
    print(f"- invalid/corrupt skipped: {len(bad_rows)}")
    print(f"- class counts: {dict(class_counts)}")
    print(f"- saved: {out_master}")
    print(f"- report: {out_report}")


if __name__ == "__main__":
    main()
