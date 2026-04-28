#!/usr/bin/env python3
"""Extract frozen VideoMAE-B features for UCF-Crime (and transfer datasets).

Replaces frozen I3D pipeline. Same segment policy:
- Segment length : 16 frames
- Overlap        : 0 (non-overlapping)
- Tail handling  : drop leftover frames (<16)
- Backbone       : frozen VideoMAE-B pretrained on Kinetics-400 (MCG-NJU/videomae-base-finetuned-kinetics)
- Feature layer  : mean-pool of all encoder token representations (no CLS)
- Feature dim    : D = 768

Usage — pilot (13 videos, sanity check):
    python extract_videomae_features.py --mode pilot

Usage — full UCF extraction:
    python extract_videomae_features.py --mode full

Usage — transfer dataset (XD-Violence etc.):
    python extract_videomae_features.py --mode full \
        --manifest data/xd_violence/manifests/xd_violence_master.csv \
        --output-root data/xd_violence/features/videomae_kinetics400_16f \
        --feature-manifest data/xd_violence/manifests/xd_violence_features_videomae.csv \
        --sanity-report data/xd_violence/manifests/videomae_sanity_report.md
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

# ── lazy-import HuggingFace to give a clean error if not installed ──────────
try:
    # transformers >=4.39 renamed VideoMAEFeatureExtractor → VideoMAEImageProcessor
    try:
        from transformers import VideoMAEImageProcessor as _VideoMAEProcessor, VideoMAEModel  # type: ignore
    except ImportError:
        from transformers import VideoMAEFeatureExtractor as _VideoMAEProcessor, VideoMAEModel  # type: ignore  # noqa: F401
    VideoMAEProcessor = _VideoMAEProcessor
except ImportError:
    raise SystemExit(
        "transformers not found. Install it with:\n"
        "  pip install transformers>=4.36.0\n"
        "On BU SCC (inside venv):\n"
        "  pip install transformers>=4.36.0 accelerate"
    )

# ── VideoMAE pretrained checkpoint ──────────────────────────────────────────
VIDEOMAE_HF_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
FEATURE_DIM = 768         # VideoMAE-B hidden size
SEGMENT_LENGTH = 16       # frames per segment (same as I3D pipeline)

# ── pilot quota (same classes/counts as I3D pilot) ──────────────────────────
PILOT_QUOTA = {
    "normal":    3,
    "fighting":  2,
    "robbery":   2,
    "shooting":  2,
    "explosion": 2,
    "abuse":     2,
}


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen VideoMAE-B feature extraction")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ucf_crime/features/videomae_kinetics400_16f"),
    )
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_videomae.csv"),
    )
    p.add_argument(
        "--sanity-report",
        type=Path,
        default=Path("data/ucf_crime/manifests/videomae_feature_sanity_report.md"),
    )
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--segment-length", type=int, default=SEGMENT_LENGTH)
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Segments per GPU forward pass. Use 4 for 16GB GPU, 8 for 40GB.",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Frame crop size (VideoMAE expects 224x224)",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory (useful on SCC scratch)",
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def choose_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return -1.0


def parse_int(s: str) -> int:
    try:
        return int(float(s))
    except Exception:
        return -1


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def select_pilot_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    used: set = set()
    for category, wanted in PILOT_QUOTA.items():
        candidates = [
            r for r in rows
            if r["category_label"] == category
            and parse_int(r.get("num_frames", "-1")) >= SEGMENT_LENGTH
        ]
        candidates.sort(key=lambda r: (parse_int(r.get("num_frames", "999999")), r["video_id"]))
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
            raise RuntimeError(f"Pilot: class={category} needs {wanted}, found {picked}")
    return selected


# ────────────────────────────────────────────────────────────────────────────
# VideoMAE model loader
# ────────────────────────────────────────────────────────────────────────────

def load_frozen_videomae(
    device: str,
    hf_cache: str | None = None,
):
    """Load VideoMAE-B (Kinetics-400), freeze all weights."""
    kwargs = {}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    print(f"Loading VideoMAE from {VIDEOMAE_HF_NAME} ...")
    processor = VideoMAEProcessor.from_pretrained(VIDEOMAE_HF_NAME, **kwargs)
    model = VideoMAEModel.from_pretrained(VIDEOMAE_HF_NAME, **kwargs)

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VideoMAE loaded: {n_params/1e6:.1f}M params, frozen, device={device}")
    return model, processor


# ────────────────────────────────────────────────────────────────────────────
# Frame decoding
# ────────────────────────────────────────────────────────────────────────────

def center_crop_resize(frame_rgb: np.ndarray, crop: int) -> np.ndarray:
    """Resize shorter side to crop, then center crop to crop x crop."""
    h, w = frame_rgb.shape[:2]
    scale = float(crop) / float(min(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top  = max(0, (new_h - crop) // 2)
    left = max(0, (new_w - crop) // 2)
    out  = resized[top:top + crop, left:left + crop]
    if out.shape[0] != crop or out.shape[1] != crop:
        pad = np.zeros((crop, crop, 3), dtype=np.uint8)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out


def iter_segments(
    video_path: Path,
    segment_length: int,
    crop: int,
) -> Iterable[Tuple[List[np.ndarray], int, int]]:
    """Yield (list_of_cropped_rgb_frames, start_frame_idx, end_frame_idx)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    buf: List[np.ndarray] = []
    start_idx = 0
    frame_idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = center_crop_resize(rgb, crop)
            if not buf:
                start_idx = frame_idx
            buf.append(rgb)
            if len(buf) == segment_length:
                yield buf, start_idx, frame_idx
                buf = []
            frame_idx += 1
    finally:
        cap.release()


# ────────────────────────────────────────────────────────────────────────────
# Feature extraction — single video
# ────────────────────────────────────────────────────────────────────────────

def extract_one_video(
    model: VideoMAEModel,
    processor,
    device: str,
    row: Dict[str, str],
    project_root: Path,
    output_root: Path,
    segment_length: int,
    batch_size: int,
    crop_size: int,
    overwrite: bool,
) -> VideoResult:
    video_id       = row["video_id"]
    split          = row["split"]
    category_label = row["category_label"]
    binary_label   = int(row["binary_label"])
    num_frames     = parse_int(row.get("num_frames", "-1"))
    fps            = parse_float(row.get("fps", "-1"))
    duration_sec   = parse_float(row.get("duration_sec", "-1"))

    video_path     = (project_root / row["video_path"]).resolve()
    split_dir      = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path       = split_dir / f"{video_id}.npz"

    expected_segs  = max(0, num_frames // segment_length) if num_frames > 0 else -1

    # ── skip if already extracted ──────────────────────────────────────────
    if out_path.exists() and not overwrite:
        with np.load(out_path, allow_pickle=True) as d:
            feats  = d["features"]
            starts = d["segment_start_frames"]
        return VideoResult(
            video_id=video_id, split=split,
            category_label=category_label, binary_label=binary_label,
            num_frames=num_frames, fps=fps, duration_sec=duration_sec,
            expected_segments=expected_segs,
            saved_segments=int(starts.shape[0]),
            feature_dim=int(feats.shape[1]) if feats.ndim == 2 else 0,
            feature_path=out_path.relative_to(project_root).as_posix(),
            status="ok", failure_reason="",
        )

    if not video_path.exists():
        return VideoResult(
            video_id=video_id, split=split,
            category_label=category_label, binary_label=binary_label,
            num_frames=num_frames, fps=fps, duration_sec=duration_sec,
            expected_segments=expected_segs, saved_segments=0, feature_dim=0,
            feature_path="", status="failed",
            failure_reason=f"video_not_found:{video_path}",
        )

    # ── decode + batch ─────────────────────────────────────────────────────
    seg_frames_buf: List[List[np.ndarray]] = []
    seg_starts: List[int] = []
    seg_ends: List[int] = []
    all_features: List[np.ndarray] = []
    out_starts: List[int] = []
    out_ends: List[int] = []

    def _flush(frames_batch, starts_batch, ends_batch):
        """Run VideoMAE forward on a batch of segments."""
        # processor expects list of videos; each video = list of frames (H,W,C uint8 RGB)
        pixel_values_list = []
        for frames in frames_batch:
            inp = processor(frames, return_tensors="pt")
            pixel_values_list.append(inp["pixel_values"])  # [1, 16, 3, 224, 224]

        pixel_values = torch.cat(pixel_values_list, dim=0).to(device)  # [B, 16, 3, 224, 224]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # last_hidden_state: [B, num_patches, 768]  (1568 patches for 16-frame ViT-B/16)
            feats = outputs.last_hidden_state.mean(dim=1)  # [B, 768]  — mean-pool tokens
            feats = feats.cpu().float().numpy()

        all_features.append(feats)
        out_starts.extend(starts_batch)
        out_ends.extend(ends_batch)

    try:
        for frames, s, e in iter_segments(video_path, segment_length, crop_size):
            seg_frames_buf.append(frames)
            seg_starts.append(s)
            seg_ends.append(e)
            if len(seg_frames_buf) >= batch_size:
                _flush(seg_frames_buf, seg_starts, seg_ends)
                seg_frames_buf, seg_starts, seg_ends = [], [], []

        if seg_frames_buf:
            _flush(seg_frames_buf, seg_starts, seg_ends)

    except Exception as exc:
        return VideoResult(
            video_id=video_id, split=split,
            category_label=category_label, binary_label=binary_label,
            num_frames=num_frames, fps=fps, duration_sec=duration_sec,
            expected_segments=expected_segs, saved_segments=0, feature_dim=0,
            feature_path="", status="failed",
            failure_reason=f"extract_error:{exc}",
        )

    if not all_features:
        return VideoResult(
            video_id=video_id, split=split,
            category_label=category_label, binary_label=binary_label,
            num_frames=num_frames, fps=fps, duration_sec=duration_sec,
            expected_segments=expected_segs, saved_segments=0, feature_dim=0,
            feature_path="", status="failed",
            failure_reason="no_full_segments",
        )

    features   = np.concatenate(all_features, axis=0).astype(np.float32)  # [T, 768]
    starts_arr = np.array(out_starts, dtype=np.int32)
    ends_arr   = np.array(out_ends, dtype=np.int32)

    eff_nframes  = num_frames if num_frames > 0 else (int(ends_arr[-1]) + 1 if len(ends_arr) else 0)
    eff_duration = duration_sec if duration_sec > 0 else (eff_nframes / fps if fps > 0 else -1.0)

    np.savez_compressed(
        out_path,
        features=features,
        segment_start_frames=starts_arr,
        segment_end_frames=ends_arr,
        fps=np.float32(fps),
        num_frames=np.int32(eff_nframes),
        duration_sec=np.float32(eff_duration),
        video_id=np.array(video_id),
        category_label=np.array(category_label),
        binary_label=np.int32(binary_label),
        split=np.array(split),
        segment_length=np.int32(segment_length),
        overlap=np.int32(0),
        tail_policy=np.array("drop"),
        backbone=np.array("videomae-b-kinetics400"),
    )

    return VideoResult(
        video_id=video_id, split=split,
        category_label=category_label, binary_label=binary_label,
        num_frames=eff_nframes, fps=fps, duration_sec=eff_duration,
        expected_segments=max(0, eff_nframes // segment_length),
        saved_segments=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feature_path=out_path.relative_to(project_root).as_posix(),
        status="ok", failure_reason="",
    )


# ────────────────────────────────────────────────────────────────────────────
# Manifest + report writers (same schema as I3D version)
# ────────────────────────────────────────────────────────────────────────────

def write_feature_manifest(path: Path, results: Sequence[VideoResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id", "split", "feature_path", "num_segments", "feature_dim",
        "binary_label", "category_label", "fps", "num_frames", "duration_sec",
        "status", "failure_reason",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "video_id": r.video_id, "split": r.split,
                "feature_path": r.feature_path,
                "num_segments": r.saved_segments, "feature_dim": r.feature_dim,
                "binary_label": r.binary_label, "category_label": r.category_label,
                "fps": round(r.fps, 6) if r.fps > 0 else -1,
                "num_frames": r.num_frames,
                "duration_sec": round(r.duration_sec, 3) if r.duration_sec > 0 else -1,
                "status": r.status, "failure_reason": r.failure_reason,
            })


def write_sanity_report(
    path: Path,
    mode: str,
    device: str,
    results: Sequence[VideoResult],
    elapsed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok     = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    segs   = [r.saved_segments for r in ok]
    dims   = sorted({r.feature_dim for r in ok if r.feature_dim > 0})
    by_split = Counter(r.split for r in ok)
    by_cls   = Counter(r.category_label for r in ok)

    lines = [
        "# VideoMAE Feature Extraction Report",
        "",
        "## Extractor configuration",
        f"- Backbone: **{VIDEOMAE_HF_NAME}**",
        "- Feature layer: **mean-pool of all encoder tokens (no CLS token)**",
        f"- Feature dim: **{FEATURE_DIM}**",
        f"- Segment length: **{SEGMENT_LENGTH} frames**",
        "- Overlap: **0**",
        "- Tail rule: **drop leftover frames (<16)**",
        f"- Device: **{device}**",
        "",
        "## Run summary",
        f"- Mode: **{mode}**",
        f"- Attempted: **{len(results)}**",
        f"- Succeeded: **{len(ok)}**",
        f"- Failed: **{len(failed)}**",
        f"- Runtime: **{elapsed:.1f}s**",
        "",
        "## Segment statistics",
    ]
    if segs:
        lines += [
            f"- Min: **{min(segs)}**",
            f"- Max: **{max(segs)}**",
            f"- Mean: **{sum(segs)/len(segs):.2f}**",
        ]
    lines += ["", f"## Feature dims seen: {dims}", ""]
    lines += ["## Success by split"]
    for s in ["train", "val", "test"]:
        lines.append(f"- {s}: {by_split.get(s, 0)}")
    lines += ["", "## Success by class"]
    for c in ["normal", "fighting", "robbery", "shooting", "explosion", "abuse"]:
        lines.append(f"- {c}: {by_cls.get(c, 0)}")
    lines += ["", "## Failures"]
    if failed:
        for r in failed:
            lines.append(f"- {r.video_id}: {r.failure_reason}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    manifest_path  = resolve(root, args.manifest)
    output_root    = resolve(root, args.output_root)
    feat_manifest  = resolve(root, args.feature_manifest)
    sanity_report  = resolve(root, args.sanity_report)

    rows = load_manifest(manifest_path)
    target = select_pilot_rows(rows) if args.mode == "pilot" else rows

    device = choose_device(args.device)
    model, processor = load_frozen_videomae(device, hf_cache=args.hf_cache)

    results: List[VideoResult] = []
    t0 = time.time()

    for idx, row in enumerate(target, 1):
        print(f"[{idx}/{len(target)}] {row['video_id']} ({row['category_label']}/{row['split']})")
        res = extract_one_video(
            model=model, processor=processor, device=device,
            row=row, project_root=root, output_root=output_root,
            segment_length=args.segment_length, batch_size=args.batch_size,
            crop_size=args.crop_size, overwrite=args.overwrite,
        )
        results.append(res)
        if res.status == "ok":
            print(f"  OK  segments={res.saved_segments}  dim={res.feature_dim}")
        else:
            print(f"  FAIL  {res.failure_reason}")

        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0 or idx == len(target)):
            write_feature_manifest(feat_manifest, results)
            print(f"  [checkpoint] manifest saved ({idx} videos)")

    elapsed = time.time() - t0
    write_feature_manifest(feat_manifest, results)
    write_sanity_report(sanity_report, args.mode, device, results, elapsed)

    ok_count   = sum(1 for r in results if r.status == "ok")
    fail_count = len(results) - ok_count
    print(f"\nDone — success: {ok_count}  failed: {fail_count}  time: {elapsed:.1f}s")
    print(f"Manifest : {feat_manifest}")
    print(f"Report   : {sanity_report}")


if __name__ == "__main__":
    main()
