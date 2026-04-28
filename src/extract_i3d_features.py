#!/usr/bin/env python3
"""Extract frozen I3D features for UCF-Crime videos from manifest rows.

Default mode runs the required 13-video pilot:
- normal: 3
- fighting: 2
- robbery: 2
- shooting: 2
- explosion: 2
- abuse: 2

Feature extraction policy:
- Segment length: 16 frames
- Overlap: 0 (non-overlapping)
- Tail handling: drop leftover frames (<16)
- Backbone: frozen I3D (Kinetics-400 pretrained)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

PILOT_QUOTA = {
    "normal": 3,
    "fighting": 2,
    "robbery": 2,
    "shooting": 2,
    "explosion": 2,
    "abuse": 2,
}

# Kinetics mean/std used for torchvision/pytorchvideo style I3D preprocessing.
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


@dataclass
class VideoResult:
    video_id: str
    split: str
    category_label: str
    binary_label: int
    num_frames: int
    fps: float
    duration_sec: float
    expected_segments: int
    saved_segments: int
    feature_dim: int
    feature_path: str
    status: str
    failure_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen I3D feature extraction")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
        help="Input video manifest CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ucf_crime/features/i3d_kinetics400_16f"),
        help="Feature output root",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
        help="Output feature manifest CSV",
    )
    parser.add_argument(
        "--sanity-report",
        type=Path,
        default=Path("data/ucf_crime/manifests/i3d_feature_sanity_report.md"),
        help="Output sanity report",
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="pilot=required 13-video subset, full=all manifest videos",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=16,
        help="Frames per segment",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Segments per inference batch",
    )
    parser.add_argument(
        "--resize-shorter-side",
        type=int,
        default=256,
        help="Shorter-side resize before center crop",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Center crop size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Write intermediate feature manifest every N processed videos",
    )
    return parser.parse_args()


def resolve(project_root: Path, path_arg: Path) -> Path:
    return path_arg if path_arg.is_absolute() else project_root / path_arg


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def select_pilot_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    used = set()
    for category, wanted in PILOT_QUOTA.items():
        candidates = [
            r
            for r in rows
            if r["category_label"] == category
            and r["split"] in {"train", "val", "test"}
            and parse_int(r.get("num_frames", "-1")) >= 16
        ]
        candidates.sort(
            key=lambda r: (
                parse_int(r.get("num_frames", "999999999")),
                r["split"],
                r["video_id"],
            )
        )

        picked = 0
        for row in candidates:
            if row["video_id"] in used:
                continue
            selected.append(row)
            used.add(row["video_id"])
            picked += 1
            if picked >= wanted:
                break

        if picked < wanted:
            raise RuntimeError(
                f"Pilot selection failed for class={category}: requested {wanted}, found {picked}"
            )

    return selected


def resize_and_center_crop(frame_rgb: np.ndarray, resize_short: int, crop: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame shape")

    scale = float(resize_short) / float(min(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = max(0, (new_h - crop) // 2)
    left = max(0, (new_w - crop) // 2)
    cropped = resized[top : top + crop, left : left + crop]

    if cropped.shape[0] != crop or cropped.shape[1] != crop:
        padded = np.zeros((crop, crop, 3), dtype=np.uint8)
        padded[: cropped.shape[0], : cropped.shape[1]] = cropped
        cropped = padded

    return cropped


def preprocess_segment(
    frames_rgb: Sequence[np.ndarray],
    resize_short: int,
    crop: int,
) -> torch.Tensor:
    processed: List[np.ndarray] = [
        resize_and_center_crop(frame, resize_short, crop) for frame in frames_rgb
    ]
    # T,H,W,C -> C,T,H,W
    arr = np.stack(processed).astype(np.float32) / 255.0
    arr = (arr - MEAN[None, None, None, :]) / STD[None, None, None, :]
    arr = np.transpose(arr, (3, 0, 1, 2))
    return torch.from_numpy(arr)


def iter_non_overlapping_segments(
    video_path: Path,
    segment_length: int,
) -> Iterable[Tuple[List[np.ndarray], int, int, int]]:
    """Yield (frames, start_frame_idx, end_frame_idx, total_decoded_frames_so_far)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open video: {video_path}")

    frame_buffer: List[np.ndarray] = []
    start_idx = 0
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if not frame_buffer:
                start_idx = frame_idx
            frame_buffer.append(frame_rgb)

            if len(frame_buffer) == segment_length:
                yield frame_buffer, start_idx, frame_idx, frame_idx + 1
                frame_buffer = []

            frame_idx += 1
    finally:
        cap.release()


def _forward_features(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        # Be defensive for implementations that return dict-like outputs.
        if "video" in out:
            out = out["video"]
        else:
            out = list(out.values())[0]
    if out.ndim > 2:
        out = out.flatten(1)
    return out


def load_frozen_i3d(device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    # Remove classification projection; keep pooled penultimate embedding.
    head = model.blocks[-1]
    if hasattr(head, "proj"):
        head.proj = torch.nn.Identity()
    if hasattr(head, "activation"):
        head.activation = torch.nn.Identity()
    if hasattr(head, "dropout"):
        head.dropout = torch.nn.Identity()

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def parse_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return -1.0


def parse_int(text: str) -> int:
    try:
        return int(float(text))
    except Exception:
        return -1


def extract_one_video(
    model: torch.nn.Module,
    device: str,
    row: Dict[str, str],
    project_root: Path,
    output_root: Path,
    segment_length: int,
    batch_size: int,
    resize_short: int,
    crop_size: int,
    overwrite: bool,
) -> VideoResult:
    video_id = row["video_id"]
    split = row["split"]
    category_label = row["category_label"]
    binary_label = int(row["binary_label"])

    video_path = (project_root / row["video_path"]).resolve()
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{video_id}.npz"

    num_frames = parse_int(row.get("num_frames", "-1"))
    fps = parse_float(row.get("fps", "-1"))
    duration_sec = parse_float(row.get("duration_sec", "-1"))

    expected_segments = max(0, num_frames // segment_length) if num_frames > 0 else -1

    if out_path.exists() and not overwrite:
        with np.load(out_path, allow_pickle=True) as d:
            feats = d["features"]
            starts = d["segment_start_frames"]
        return VideoResult(
            video_id=video_id,
            split=split,
            category_label=category_label,
            binary_label=binary_label,
            num_frames=num_frames,
            fps=fps,
            duration_sec=duration_sec,
            expected_segments=expected_segments,
            saved_segments=int(starts.shape[0]),
            feature_dim=int(feats.shape[1]) if feats.ndim == 2 and feats.shape[0] > 0 else 0,
            feature_path=out_path.relative_to(project_root).as_posix(),
            status="ok",
            failure_reason="",
        )

    if not video_path.exists():
        return VideoResult(
            video_id=video_id,
            split=split,
            category_label=category_label,
            binary_label=binary_label,
            num_frames=num_frames,
            fps=fps,
            duration_sec=duration_sec,
            expected_segments=expected_segments,
            saved_segments=0,
            feature_dim=0,
            feature_path="",
            status="failed",
            failure_reason=f"video_not_found:{video_path}",
        )

    batch_tensors: List[torch.Tensor] = []
    batch_ranges: List[Tuple[int, int]] = []
    all_features: List[np.ndarray] = []
    start_frames: List[int] = []
    end_frames: List[int] = []

    decoded_frames = 0

    try:
        with torch.no_grad():
            for frames, start_f, end_f, decoded in iter_non_overlapping_segments(video_path, segment_length):
                decoded_frames = decoded
                seg_tensor = preprocess_segment(frames, resize_short=resize_short, crop=crop_size)
                batch_tensors.append(seg_tensor)
                batch_ranges.append((start_f, end_f))

                if len(batch_tensors) >= batch_size:
                    batch = torch.stack(batch_tensors, dim=0).to(device)
                    feats = _forward_features(model, batch).cpu().numpy().astype(np.float32)
                    all_features.append(feats)
                    for s, e in batch_ranges:
                        start_frames.append(s)
                        end_frames.append(e)
                    batch_tensors = []
                    batch_ranges = []

            if batch_tensors:
                batch = torch.stack(batch_tensors, dim=0).to(device)
                feats = _forward_features(model, batch).cpu().numpy().astype(np.float32)
                all_features.append(feats)
                for s, e in batch_ranges:
                    start_frames.append(s)
                    end_frames.append(e)

    except Exception as exc:
        return VideoResult(
            video_id=video_id,
            split=split,
            category_label=category_label,
            binary_label=binary_label,
            num_frames=num_frames,
            fps=fps,
            duration_sec=duration_sec,
            expected_segments=expected_segments,
            saved_segments=0,
            feature_dim=0,
            feature_path="",
            status="failed",
            failure_reason=f"extract_error:{exc}",
        )

    if not all_features:
        return VideoResult(
            video_id=video_id,
            split=split,
            category_label=category_label,
            binary_label=binary_label,
            num_frames=num_frames if num_frames > 0 else decoded_frames,
            fps=fps,
            duration_sec=duration_sec,
            expected_segments=expected_segments,
            saved_segments=0,
            feature_dim=0,
            feature_path="",
            status="failed",
            failure_reason="no_full_16f_segments",
        )

    features = np.concatenate(all_features, axis=0)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)

    starts_arr = np.array(start_frames, dtype=np.int32)
    ends_arr = np.array(end_frames, dtype=np.int32)

    # If manifest metadata is stale, trust decode-derived values.
    effective_num_frames = num_frames if num_frames > 0 else decoded_frames
    effective_duration = duration_sec
    if effective_duration <= 0 and fps > 0 and effective_num_frames > 0:
        effective_duration = float(effective_num_frames) / fps

    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        segment_start_frames=starts_arr,
        segment_end_frames=ends_arr,
        fps=np.array(fps, dtype=np.float32),
        num_frames=np.array(effective_num_frames, dtype=np.int32),
        duration_sec=np.array(effective_duration, dtype=np.float32),
        video_id=np.array(video_id),
        category_label=np.array(category_label),
        binary_label=np.array(binary_label, dtype=np.int32),
        split=np.array(split),
        segment_length=np.array(segment_length, dtype=np.int32),
        overlap=np.array(0, dtype=np.int32),
        tail_policy=np.array("drop"),
    )

    return VideoResult(
        video_id=video_id,
        split=split,
        category_label=category_label,
        binary_label=binary_label,
        num_frames=effective_num_frames,
        fps=fps,
        duration_sec=effective_duration,
        expected_segments=max(0, effective_num_frames // segment_length),
        saved_segments=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feature_path=out_path.relative_to(project_root).as_posix(),
        status="ok",
        failure_reason="",
    )


def write_feature_manifest(path: Path, results: Sequence[VideoResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "split",
        "feature_path",
        "num_segments",
        "feature_dim",
        "binary_label",
        "category_label",
        "fps",
        "num_frames",
        "duration_sec",
        "status",
        "failure_reason",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "video_id": r.video_id,
                    "split": r.split,
                    "feature_path": r.feature_path,
                    "num_segments": r.saved_segments,
                    "feature_dim": r.feature_dim,
                    "binary_label": r.binary_label,
                    "category_label": r.category_label,
                    "fps": round(r.fps, 6) if r.fps > 0 else -1,
                    "num_frames": r.num_frames,
                    "duration_sec": round(r.duration_sec, 3) if r.duration_sec > 0 else -1,
                    "status": r.status,
                    "failure_reason": r.failure_reason,
                }
            )


def write_sanity_report(
    path: Path,
    mode: str,
    device: str,
    segment_length: int,
    batch_size: int,
    resize_short: int,
    crop_size: int,
    extractor_name: str,
    pretrained_source: str,
    results: Sequence[VideoResult],
    elapsed_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    seg_counts = [r.saved_segments for r in ok]
    feature_dims = sorted({r.feature_dim for r in ok if r.feature_dim > 0})

    by_split = Counter(r.split for r in ok)
    by_class = Counter(r.category_label for r in ok)

    lines = [
        "# I3D Feature Extraction Sanity Report",
        "",
        "## Extractor configuration",
        f"- Extractor: **{extractor_name}**",
        f"- Pretrained source: **{pretrained_source}**",
        "- Feature layer: **before final classification projection (frozen backbone embedding)**",
        f"- Device: **{device}**",
        f"- Input resolution: shorter-side resize {resize_short}, center crop {crop_size}",
        f"- Segment length: **{segment_length}**",
        "- Overlap: **0**",
        "- Tail rule: **drop leftover frames (<16)**",
        f"- Batch size (segments): **{batch_size}**",
        "",
        "## Run summary",
        f"- Mode: **{mode}**",
        f"- Videos attempted: **{len(results)}**",
        f"- Videos succeeded: **{len(ok)}**",
        f"- Videos failed: **{len(failed)}**",
        f"- Runtime: **{elapsed_sec:.2f}s**",
        "",
        "## Segment statistics (successful videos)",
    ]

    if seg_counts:
        lines.extend(
            [
                f"- Min segments: **{min(seg_counts)}**",
                f"- Max segments: **{max(seg_counts)}**",
                f"- Mean segments: **{(sum(seg_counts) / len(seg_counts)):.2f}**",
            ]
        )
    else:
        lines.append("- No successful videos")

    lines.extend(["", "## Feature dimensionality"])
    if feature_dims:
        lines.append(f"- Unique feature dimensions seen: **{feature_dims}**")
    else:
        lines.append("- No successful feature outputs")

    lines.extend(["", "## Success count by split"])
    for split in ["train", "val", "test"]:
        lines.append(f"- {split}: {by_split.get(split, 0)}")

    lines.extend(["", "## Success count by class"])
    for cls in ["normal", "fighting", "robbery", "shooting", "explosion", "abuse"]:
        lines.append(f"- {cls}: {by_class.get(cls, 0)}")

    lines.extend(["", "## Failures"])
    if failed:
        for r in failed:
            lines.append(f"- {r.video_id} ({r.category_label}/{r.split}): {r.failure_reason}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_path = resolve(project_root, args.manifest)
    output_root = resolve(project_root, args.output_root)
    feature_manifest_path = resolve(project_root, args.feature_manifest)
    sanity_report_path = resolve(project_root, args.sanity_report)

    rows = load_manifest(manifest_path)

    if args.mode == "pilot":
        target_rows = select_pilot_rows(rows)
    else:
        target_rows = rows

    device = choose_device(args.device)

    start = time.time()
    model = load_frozen_i3d(device)

    results: List[VideoResult] = []
    for idx, row in enumerate(target_rows, start=1):
        print(
            f"[{idx}/{len(target_rows)}] extracting {row['video_id']} "
            f"({row['category_label']}/{row['split']})"
        )
        res = extract_one_video(
            model=model,
            device=device,
            row=row,
            project_root=project_root,
            output_root=output_root,
            segment_length=args.segment_length,
            batch_size=args.batch_size,
            resize_short=args.resize_shorter_side,
            crop_size=args.crop_size,
            overwrite=args.overwrite,
        )
        results.append(res)
        if res.status != "ok":
            print(f"  FAILED: {res.failure_reason}")
        else:
            print(
                f"  OK: segments={res.saved_segments}, dim={res.feature_dim}, "
                f"feature={res.feature_path}"
            )

        if args.checkpoint_every > 0 and (
            idx % args.checkpoint_every == 0 or idx == len(target_rows)
        ):
            write_feature_manifest(feature_manifest_path, results)

    elapsed = time.time() - start

    write_feature_manifest(feature_manifest_path, results)
    write_sanity_report(
        path=sanity_report_path,
        mode=args.mode,
        device=device,
        segment_length=args.segment_length,
        batch_size=args.batch_size,
        resize_short=args.resize_shorter_side,
        crop_size=args.crop_size,
        extractor_name="pytorchvideo i3d_r50",
        pretrained_source="Kinetics-400 (torch hub: facebookresearch/pytorchvideo)",
        results=results,
        elapsed_sec=elapsed,
    )

    summary = Counter(r.status for r in results)
    print("\nExtraction complete")
    print(f"- manifest: {feature_manifest_path}")
    print(f"- report: {sanity_report_path}")
    print(f"- success: {summary.get('ok', 0)}")
    print(f"- failed: {sum(v for k, v in summary.items() if k != 'ok')}")


if __name__ == "__main__":
    main()
