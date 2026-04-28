#!/usr/bin/env python3
"""Prepare UCF-Crime violence subset manifests and split files.

This script builds a primary UCF-Crime subset with categories:
- fighting, shooting, explosion, robbery, abuse, normal

Outputs:
- data/ucf_crime/manifests/ucf_violence_master.csv
- data/ucf_crime/splits/train.csv
- data/ucf_crime/splits/val.csv
- data/ucf_crime/splits/test.csv
- data/ucf_crime/manifests/ucf_violence_summary.md
- data/ucf_crime/manifests/missing_videos.txt
- data/ucf_crime/manifests/unreadable_videos.txt
- data/ucf_crime/splits/class_counts_by_split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TARGET_CLASS_MAP = {
    "Abuse": "abuse",
    "Fighting": "fighting",
    "Shooting": "shooting",
    "Explosion": "explosion",
    "Robbery": "robbery",
    "Normal": "normal",
    "Training_Normal_Videos_Anomaly": "normal",
    "Testing_Normal_Videos_Anomaly": "normal",
}

CATEGORY_ORDER = ["fighting", "shooting", "explosion", "robbery", "abuse", "normal"]
SPLIT_ORDER = ["train", "val", "test"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
LOW_SAMPLE_THRESHOLD = 60


@dataclass
class VideoSpec:
    video_id: str
    filename: str
    category_label: str
    raw_class: str
    source_rel_hint: str
    split: str
    has_temporal_annotation: int


@dataclass
class ProbeResult:
    status: str  # ok | missing | unreadable
    num_frames: int
    fps: float
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UCF-Crime violence subset manifests")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current working directory)",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("data/ucf_crime/annotations/Anomaly_Train.txt"),
        help="Path to Anomaly_Train.txt (relative to project root if not absolute)",
    )
    parser.add_argument(
        "--temporal-annotation",
        type=Path,
        default=Path("data/ucf_crime/annotations/Temporal_Anomaly_Annotation.txt"),
        help="Path to Temporal_Anomaly_Annotation.txt (relative to project root if not absolute)",
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=Path("data/ucf_crime/raw_videos"),
        help="Root folder containing UCF-Crime raw videos",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation ratio taken from official train list (default: 0.10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    return parser.parse_args()


def resolve_path(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else project_root / maybe_relative


def normalize_category(raw_class: str, filename: str) -> Optional[str]:
    if raw_class in TARGET_CLASS_MAP:
        return TARGET_CLASS_MAP[raw_class]

    if filename.startswith("Normal_Videos"):
        return "normal"

    cleaned = raw_class.strip().replace(" ", "")
    if cleaned in TARGET_CLASS_MAP:
        return TARGET_CLASS_MAP[cleaned]

    return None


def parse_train_list(train_list_path: Path) -> List[VideoSpec]:
    specs: List[VideoSpec] = []
    for raw_line in train_list_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "/" in line:
            raw_class, filename = line.split("/", 1)
        else:
            raw_class, filename = "", line

        category = normalize_category(raw_class, filename)
        if category is None:
            continue

        video_id = Path(filename).stem
        specs.append(
            VideoSpec(
                video_id=video_id,
                filename=filename,
                category_label=category,
                raw_class=raw_class if raw_class else "Unknown",
                source_rel_hint=line,
                split="train",
                has_temporal_annotation=0,
            )
        )

    return specs


def parse_temporal_annotations(
    temporal_path: Path,
) -> Tuple[List[VideoSpec], Dict[str, List[Dict[str, int]]], Dict[str, str]]:
    specs: List[VideoSpec] = []
    temporal_segments: Dict[str, List[Dict[str, int]]] = {}
    raw_class_by_video_id: Dict[str, str] = {}

    for raw_line in temporal_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        filename, raw_class = parts[0], parts[1]

        category = normalize_category(raw_class, filename)
        if category is None:
            continue

        values: List[int] = []
        for item in parts[2:6]:
            try:
                values.append(int(float(item)))
            except ValueError:
                values.append(-1)

        segments: List[Dict[str, int]] = []
        for i in (0, 2):
            start_f, end_f = values[i], values[i + 1]
            if start_f >= 0 and end_f >= 0 and end_f >= start_f:
                segments.append({"start_frame": start_f, "end_frame": end_f})

        video_id = Path(filename).stem
        specs.append(
            VideoSpec(
                video_id=video_id,
                filename=filename,
                category_label=category,
                raw_class=raw_class,
                source_rel_hint=f"{raw_class}/{filename}",
                split="test",
                has_temporal_annotation=1,
            )
        )
        temporal_segments[video_id] = segments
        raw_class_by_video_id[video_id] = raw_class

    return specs, temporal_segments, raw_class_by_video_id


def stratified_train_val_split(
    train_specs: Iterable[VideoSpec], val_ratio: float, seed: int
) -> Tuple[List[VideoSpec], List[VideoSpec]]:
    by_class: Dict[str, List[VideoSpec]] = defaultdict(list)
    for spec in train_specs:
        by_class[spec.category_label].append(spec)

    rng = random.Random(seed)
    train_out: List[VideoSpec] = []
    val_out: List[VideoSpec] = []

    for category, items in by_class.items():
        items = list(items)
        rng.shuffle(items)

        if len(items) <= 1:
            val_count = 0
        else:
            val_count = max(1, int(round(len(items) * val_ratio)))
            val_count = min(val_count, len(items) - 1)

        val_items = items[:val_count]
        train_items = items[val_count:]

        for item in train_items:
            train_out.append(
                VideoSpec(
                    video_id=item.video_id,
                    filename=item.filename,
                    category_label=item.category_label,
                    raw_class=item.raw_class,
                    source_rel_hint=item.source_rel_hint,
                    split="train",
                    has_temporal_annotation=item.has_temporal_annotation,
                )
            )
        for item in val_items:
            val_out.append(
                VideoSpec(
                    video_id=item.video_id,
                    filename=item.filename,
                    category_label=item.category_label,
                    raw_class=item.raw_class,
                    source_rel_hint=item.source_rel_hint,
                    split="val",
                    has_temporal_annotation=item.has_temporal_annotation,
                )
            )

    return train_out, val_out


def discover_videos(raw_videos_dir: Path) -> Tuple[Dict[str, List[Path]], int]:
    basename_to_paths: Dict[str, List[Path]] = defaultdict(list)
    total_files = 0

    if not raw_videos_dir.exists():
        return basename_to_paths, total_files

    for path in raw_videos_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        basename_to_paths[path.name].append(path.resolve())
        total_files += 1

    return basename_to_paths, total_files


def choose_video_path(spec: VideoSpec, raw_videos_dir: Path, discovered: Dict[str, List[Path]]) -> Path:
    if spec.filename in discovered and discovered[spec.filename]:
        return sorted(discovered[spec.filename])[0]

    candidates = []
    if spec.source_rel_hint:
        candidates.append((raw_videos_dir / spec.source_rel_hint).resolve())
    if spec.raw_class:
        candidates.append((raw_videos_dir / spec.raw_class / spec.filename).resolve())
    if spec.category_label == "normal":
        candidates.append((raw_videos_dir / "Training_Normal_Videos_Anomaly" / spec.filename).resolve())
        candidates.append((raw_videos_dir / "Testing_Normal_Videos_Anomaly" / spec.filename).resolve())
    candidates.append((raw_videos_dir / spec.filename).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_ffprobe_fps(raw_fps: str) -> float:
    if not raw_fps or raw_fps == "0/0":
        return -1.0
    try:
        return float(Fraction(raw_fps))
    except Exception:
        return -1.0


def probe_video(video_path: Path) -> ProbeResult:
    if not video_path.exists():
        return ProbeResult(status="missing", num_frames=-1, fps=-1.0, duration_sec=-1.0)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ProbeResult(status="unreadable", num_frames=-1, fps=-1.0, duration_sec=-1.0)

    try:
        payload = json.loads(proc.stdout)
        streams = payload.get("streams", [])
        stream = streams[0] if streams else {}
        fmt = payload.get("format", {})

        fps = parse_ffprobe_fps(str(stream.get("avg_frame_rate", "")))

        duration_raw = stream.get("duration", fmt.get("duration", "-1"))
        try:
            duration_sec = float(duration_raw)
        except (TypeError, ValueError):
            duration_sec = -1.0

        nb_frames_raw = stream.get("nb_frames", "-1")
        try:
            num_frames = int(float(nb_frames_raw))
        except (TypeError, ValueError):
            num_frames = -1

        if num_frames <= 0 and fps > 0 and duration_sec > 0:
            num_frames = int(round(fps * duration_sec))

        if fps <= 0 or duration_sec <= 0 or num_frames <= 0:
            return ProbeResult(status="unreadable", num_frames=-1, fps=-1.0, duration_sec=-1.0)

        return ProbeResult(status="ok", num_frames=num_frames, fps=fps, duration_sec=duration_sec)

    except json.JSONDecodeError:
        return ProbeResult(status="unreadable", num_frames=-1, fps=-1.0, duration_sec=-1.0)


def write_temporal_files(
    temporal_segments: Dict[str, List[Dict[str, int]]],
    raw_class_by_video_id: Dict[str, str],
    annotations_dir: Path,
) -> Dict[str, Path]:
    out_dir = annotations_dir / "temporal_segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: Dict[str, Path] = {}
    for video_id, segments in temporal_segments.items():
        payload = {
            "video_id": video_id,
            "raw_class": raw_class_by_video_id.get(video_id, "Unknown"),
            "segments": segments,
        }
        out_path = out_dir / f"{video_id}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        out_paths[video_id] = out_path

    return out_paths


def format_rel(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    train_list_path = resolve_path(project_root, args.train_list)
    temporal_path = resolve_path(project_root, args.temporal_annotation)
    raw_videos_dir = resolve_path(project_root, args.raw_videos_dir)

    manifests_dir = project_root / "data/ucf_crime/manifests"
    splits_dir = project_root / "data/ucf_crime/splits"
    annotations_dir = project_root / "data/ucf_crime/annotations"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not train_list_path.exists():
        raise FileNotFoundError(f"Missing train list: {train_list_path}")
    if not temporal_path.exists():
        raise FileNotFoundError(f"Missing temporal annotation file: {temporal_path}")

    train_pool = parse_train_list(train_list_path)
    test_specs, temporal_segments, raw_class_by_video_id = parse_temporal_annotations(temporal_path)

    # Keep only requested UCF-Crime subset labels.
    selected_classes = set(CATEGORY_ORDER)
    train_pool = [x for x in train_pool if x.category_label in selected_classes]
    test_specs = [x for x in test_specs if x.category_label in selected_classes]

    train_specs, val_specs = stratified_train_val_split(train_pool, args.val_ratio, args.seed)

    # Enforce no split overlap by video_id.
    split_to_ids = {
        "train": {x.video_id for x in train_specs},
        "val": {x.video_id for x in val_specs},
        "test": {x.video_id for x in test_specs},
    }

    overlap_errors: List[str] = []
    for i, split_a in enumerate(SPLIT_ORDER):
        for split_b in SPLIT_ORDER[i + 1 :]:
            overlap = split_to_ids[split_a].intersection(split_to_ids[split_b])
            if overlap:
                overlap_errors.append(f"{split_a} vs {split_b}: {len(overlap)} overlaps")

    if overlap_errors:
        raise RuntimeError("Split overlap detected: " + "; ".join(overlap_errors))

    temporal_paths = write_temporal_files(temporal_segments, raw_class_by_video_id, annotations_dir)

    discovered, discovered_count = discover_videos(raw_videos_dir)

    rows: List[Dict[str, object]] = []
    missing_paths: List[str] = []
    unreadable_paths: List[str] = []

    all_specs = train_specs + val_specs + test_specs
    for spec in all_specs:
        video_path = choose_video_path(spec, raw_videos_dir, discovered)
        probe = probe_video(video_path)

        if probe.status == "missing":
            missing_paths.append(video_path.as_posix())
        elif probe.status == "unreadable":
            unreadable_paths.append(video_path.as_posix())

        binary_label = 0 if spec.category_label == "normal" else 1
        temporal_path_rel = ""
        if spec.split == "test" and spec.video_id in temporal_paths:
            temporal_path_rel = format_rel(temporal_paths[spec.video_id], project_root)

        rows.append(
            {
                "video_id": spec.video_id,
                "video_path": format_rel(video_path, project_root),
                "split": spec.split,
                "binary_label": binary_label,
                "category_label": spec.category_label,
                "dataset": "ucf_crime",
                "num_frames": probe.num_frames,
                "fps": round(probe.fps, 6) if probe.fps > 0 else -1,
                "duration_sec": round(probe.duration_sec, 3) if probe.duration_sec > 0 else -1,
                "has_temporal_annotation": spec.has_temporal_annotation,
                "temporal_annotation_path": temporal_path_rel,
            }
        )

    rows.sort(
        key=lambda r: (
            SPLIT_ORDER.index(str(r["split"])),
            CATEGORY_ORDER.index(str(r["category_label"])),
            str(r["video_id"]),
        )
    )

    master_csv = manifests_dir / "ucf_violence_master.csv"
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

    with master_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save split CSVs.
    split_rows = {split: [r for r in rows if r["split"] == split] for split in SPLIT_ORDER}
    for split, items in split_rows.items():
        out_path = splits_dir / f"{split}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)

    # Save class counts by split.
    class_counts_by_split = {
        split: Counter(str(r["category_label"]) for r in items) for split, items in split_rows.items()
    }
    counts_csv = splits_dir / "class_counts_by_split.csv"
    with counts_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "category_label", "count"])
        for split in SPLIT_ORDER:
            for category in CATEGORY_ORDER:
                writer.writerow([split, category, class_counts_by_split[split].get(category, 0)])

    # Save missing/unreadable lists.
    missing_txt = manifests_dir / "missing_videos.txt"
    unreadable_txt = manifests_dir / "unreadable_videos.txt"
    missing_txt.write_text("\n".join(sorted(set(missing_paths))) + ("\n" if missing_paths else ""))
    unreadable_txt.write_text("\n".join(sorted(set(unreadable_paths))) + ("\n" if unreadable_paths else ""))

    total_count = len(rows)
    overall_class_counts = Counter(str(r["category_label"]) for r in rows)
    split_counts = {split: len(items) for split, items in split_rows.items()}
    valid_count = sum(1 for r in rows if int(r["num_frames"]) > 0)

    low_sample_classes = [
        (category, count)
        for category, count in sorted(overall_class_counts.items())
        if count < LOW_SAMPLE_THRESHOLD
    ]

    test_with_temporal = sum(int(r["has_temporal_annotation"]) for r in split_rows["test"])

    summary_md = manifests_dir / "ucf_violence_summary.md"
    lines = [
        "# UCF-Crime Violence Subset Summary",
        "",
        "## Source files",
        f"- Train list: `{format_rel(train_list_path, project_root)}`",
        f"- Temporal annotations: `{format_rel(temporal_path, project_root)}`",
        "",
        "## Totals",
        f"- Total videos in manifest: **{total_count}**",
        f"- Videos with readable metadata: **{valid_count}**",
        f"- Missing videos: **{len(set(missing_paths))}**",
        f"- Unreadable/corrupt videos: **{len(set(unreadable_paths))}**",
        f"- Video files discovered under raw_videos: **{discovered_count}**",
        "",
        "## Count per class",
    ]

    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {overall_class_counts.get(category, 0)}")

    lines.extend(["", "## Count per split"])
    for split in SPLIT_ORDER:
        lines.append(f"- {split}: {split_counts.get(split, 0)}")

    lines.extend(["", "## Count per class per split"])
    for split in SPLIT_ORDER:
        lines.append(f"- {split}:")
        for category in CATEGORY_ORDER:
            lines.append(
                f"  - {category}: {class_counts_by_split[split].get(category, 0)}"
            )

    lines.extend(["", "## Low-sample classes"])
    if low_sample_classes:
        lines.append(
            f"- Threshold used: < {LOW_SAMPLE_THRESHOLD} videos in the manifest"
        )
        for category, count in low_sample_classes:
            lines.append(f"- {category}: {count}")
    else:
        lines.append(f"- None below threshold (< {LOW_SAMPLE_THRESHOLD})")

    lines.extend(["", "## Temporal annotation coverage"])
    lines.append(
        f"- Test videos with temporal annotation entries: {test_with_temporal}/{split_counts.get('test', 0)}"
    )

    summary_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote master manifest: {master_csv}")
    print(f"Wrote splits: {splits_dir / 'train.csv'}, {splits_dir / 'val.csv'}, {splits_dir / 'test.csv'}")
    print(f"Wrote summary: {summary_md}")
    print(f"Total videos: {total_count}")
    print(f"Missing: {len(set(missing_paths))}, unreadable: {len(set(unreadable_paths))}")


if __name__ == "__main__":
    main()
