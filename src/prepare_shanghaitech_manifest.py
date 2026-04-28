#!/usr/bin/env python3
"""Build ShanghaiTech manifest for Step-13 binary robustness evaluation."""

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
    p = argparse.ArgumentParser(description="Prepare ShanghaiTech binary manifest")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--raw-root", type=Path, default=Path("data/shanghaitech/raw_videos"))
    p.add_argument(
        "--output-master",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_master.csv"),
    )
    p.add_argument(
        "--output-eval",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_eval.csv"),
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_manifest_report.json"),
    )
    p.add_argument("--dataset-name", type=str, default="shanghaitech")
    p.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--default-split", choices=["train", "val", "test"], default="test")
    p.add_argument(
        "--label-csv",
        type=Path,
        default=Path(""),
        help="Optional CSV with per-video binary labels (for datasets without label in path)",
    )
    p.add_argument(
        "--label-key",
        type=str,
        default="video_id",
        help="Column in label CSV used to match video stem",
    )
    p.add_argument(
        "--label-value",
        type=str,
        default="binary_label",
        help="Column in label CSV with {0,1} or {normal,anomaly}",
    )
    p.add_argument("--skip-unknown", action="store_true", default=True)
    p.add_argument("--include-unknown", dest="skip_unknown", action="store_false")
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def list_videos(raw_root: Path) -> List[Path]:
    vids: List[Path] = []
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    vids.sort()
    return vids


def infer_split(path_str: str, default_split: str) -> str:
    s = path_str.lower()
    if re.search(r"(^|[/_\- ])train(ing)?([/_\- ]|$)", s):
        return "train"
    if re.search(r"(^|[/_\- ])val(idation)?([/_\- ]|$)", s):
        return "val"
    if re.search(r"(^|[/_\- ])test(ing)?([/_\- ]|$)", s):
        return "test"
    return default_split


def infer_category(path_str: str) -> Optional[str]:
    s = path_str.lower()

    # Normal aliases.
    normal_patterns = [
        r"(^|[/_\- ])normal([/_\- ]|$)",
        r"non[-_ ]?violent",
        r"non[-_ ]?violence",
        r"(^|[/_\- ])background([/_\- ]|$)",
    ]
    for pat in normal_patterns:
        if re.search(pat, s):
            return "normal"

    # Anomaly / abnormal aliases.
    anomaly_patterns = [
        r"(^|[/_\- ])anomaly([/_\- ]|$)",
        r"(^|[/_\- ])abnormal([/_\- ]|$)",
        r"(^|[/_\- ])violent([/_\- ]|$)",
        r"(^|[/_\- ])violence([/_\- ]|$)",
        r"(^|[/_\- ])crime([/_\- ]|$)",
        r"(^|[/_\- ])fight(ing)?([/_\- ]|$)",
        r"(^|[/_\- ])shoot(ing)?([/_\- ]|$)",
        r"(^|[/_\- ])explosion([/_\- ]|$)",
        r"(^|[/_\- ])robbery([/_\- ]|$)",
        r"(^|[/_\- ])abuse([/_\- ]|$)",
    ]
    for pat in anomaly_patterns:
        if re.search(pat, s):
            return "anomaly"

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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_label_map(label_csv: Path, key_col: str, val_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not label_csv or not str(label_csv):
        return out
    if not label_csv.exists():
        return out
    with label_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        k = str(r.get(key_col, "")).strip()
        v = str(r.get(val_col, "")).strip()
        if not k or not v:
            continue
        out[k] = v
    return out


def normalize_label_value(v: str) -> Optional[str]:
    x = v.strip().lower()
    if x in {"0", "normal", "non_anomaly", "non-anomaly", "negative"}:
        return "normal"
    if x in {"1", "anomaly", "abnormal", "positive"}:
        return "anomaly"
    return None


def main() -> None:
    args = parse_args()

    root = args.project_root.resolve()
    raw_root = resolve(root, args.raw_root)
    out_master = resolve(root, args.output_master)
    out_eval = resolve(root, args.output_eval)
    out_report = resolve(root, args.output_report)
    label_csv = resolve(root, args.label_csv) if str(args.label_csv) else Path("")

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    videos = list_videos(raw_root)
    label_map = load_label_map(label_csv, args.label_key, args.label_value) if str(label_csv) else {}
    seen_ids: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []
    unknown_paths: List[str] = []
    label_override_used = 0
    bad_rows: List[Dict[str, object]] = []

    for vp in videos:
        rel = vp.relative_to(root).as_posix() if str(vp).startswith(str(root)) else vp.as_posix()
        split = infer_split(rel, args.default_split)
        category = None

        # Optional external label map wins when available.
        stem = vp.stem
        if label_map:
            raw_lab = label_map.get(stem, "")
            norm = normalize_label_value(raw_lab) if raw_lab else None
            if norm is not None:
                category = norm
                label_override_used += 1

        if category is None:
            category = infer_category(rel)
        if category is None:
            unknown_paths.append(rel)
            if args.skip_unknown:
                continue
            category = "unknown"

        ok, nframes, fps, duration, reason = safe_video_stats(vp)
        if not ok:
            bad_rows.append({"video_path": rel, "reason": reason})
            continue

        video_id = make_unique_video_id(vp.stem, seen_ids)
        binary_label = 0 if category == "normal" else 1

        rows.append(
            {
                "video_id": video_id,
                "video_path": rel,
                "split": split,
                "binary_label": binary_label,
                "category_label": category if category in {"normal", "anomaly"} else "anomaly",
                "dataset": args.dataset_name,
                "num_frames": nframes,
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
        "eval_split": args.eval_split,
        "unknown_label_paths_count": len(unknown_paths),
        "unknown_label_paths_preview": unknown_paths[:100],
        "label_csv": str(label_csv) if str(label_csv) else "",
        "label_override_used_count": label_override_used,
        "invalid_or_unreadable_videos_count": len(bad_rows),
        "invalid_or_unreadable_videos_preview": bad_rows[:100],
        "output_master": str(out_master),
        "output_eval": str(out_eval),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")

    print("ShanghaiTech manifest prep complete")
    print(f"- total files scanned: {len(videos)}")
    print(f"- manifest rows: {len(rows)}")
    print(f"- eval rows: {len(eval_rows)}")
    print(f"- unknown labels skipped: {len(unknown_paths) if args.skip_unknown else 0}")
    print(f"- invalid/corrupt skipped: {len(bad_rows)}")
    print(f"- class counts: {dict(class_counts)}")
    print(f"- saved: {out_master}")
    print(f"- report: {out_report}")


if __name__ == "__main__":
    main()
