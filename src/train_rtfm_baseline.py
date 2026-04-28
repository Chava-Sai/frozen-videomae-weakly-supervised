#!/usr/bin/env python3
"""Train/evaluate an RTFM-style binary anomaly baseline on cached I3D features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from feature_dataset import FeatureSequenceDataset, fixed_segments_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTFM baseline training on cached features")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rtfm_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--target-segments", type=int, default=32)
    parser.add_argument("--topk-ratio", type=float, default=0.125)
    parser.add_argument("--rtfm-margin", type=float, default=5.0)
    parser.add_argument("--rtfm-lambda", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


class RTFMBaseline(nn.Module):
    """Simple RTFM-style snippet scorer with magnitude separation."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.segment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        z = self.feature_proj(x)  # [B, T, H]
        seg_logits = self.segment_head(z).squeeze(-1)  # [B, T]
        feat_magnitudes = torch.norm(z, p=2, dim=-1)  # [B, T]
        return seg_logits, feat_magnitudes


def topk_mean(values: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, int]:
    t = values.shape[1]
    k = max(1, int(math.ceil(t * ratio)))
    topk_vals = torch.topk(values, k=k, dim=1).values
    return topk_vals.mean(dim=1), k


def build_loader(
    ds: FeatureSequenceDataset,
    batch_size: int,
    num_workers: int,
    balanced_sampler: bool,
) -> DataLoader:
    if balanced_sampler:
        labels = np.array([int(r["binary_label"]) for r in ds.rows], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=fixed_segments_collate,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=fixed_segments_collate,
    )


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_bce_loss: float
    train_rtfm_loss: float
    val_auc: float
    val_ap: float


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def train_one_epoch(
    model: RTFMBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_fn: nn.Module,
    device: str,
    topk_ratio: float,
    rtfm_margin: float,
    rtfm_lambda: float,
) -> Tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_rtfm = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["binary_labels"].float().to(device)

        seg_logits, feat_mags = model(x)
        video_logits, _ = topk_mean(seg_logits, topk_ratio)
        bce_loss = bce_fn(video_logits, y)

        # RTFM magnitude separation: anomaly top-k magnitudes should exceed normal top-k.
        rtfm_loss = torch.tensor(0.0, device=device)
        anom_mask = y > 0.5
        norm_mask = y <= 0.5
        if anom_mask.any() and norm_mask.any():
            anom_topk, _ = topk_mean(feat_mags[anom_mask], topk_ratio)
            norm_topk, _ = topk_mean(feat_mags[norm_mask], topk_ratio)
            rtfm_loss = torch.relu(rtfm_margin - (anom_topk.mean() - norm_topk.mean()))

        loss = bce_loss + rtfm_lambda * rtfm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_rtfm += float(rtfm_loss.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    return total_loss / n_batches, total_bce / n_batches, total_rtfm / n_batches


def evaluate(
    model: RTFMBaseline,
    loader: DataLoader,
    device: str,
    topk_ratio: float,
    threshold: float,
    collect_segments: bool = False,
) -> Dict[str, object]:
    model.eval()

    y_true: List[int] = []
    y_score: List[float] = []
    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["binary_labels"].to(device)

            seg_logits, _ = model(x)
            video_logits, _ = topk_mean(seg_logits, topk_ratio)
            probs = torch.sigmoid(video_logits)

            for i in range(x.shape[0]):
                yi = int(y[i].item())
                pi = float(probs[i].item())
                y_true.append(yi)
                y_score.append(pi)

                rec = {
                    "video_id": batch["video_ids"][i],
                    "split": batch["splits"][i],
                    "category_label": batch["category_labels"][i],
                    "binary_label": yi,
                    "score": pi,
                    "original_num_segments": int(batch["original_num_segments"][i].item()),
                }

                if collect_segments:
                    seg_probs = torch.sigmoid(seg_logits[i]).detach().cpu().numpy().astype(float)
                    selected_idx = (
                        batch["selected_segment_indices"][i].detach().cpu().numpy().astype(int).tolist()
                    )
                    starts = batch["segment_start_frames"][i].detach().cpu().numpy().astype(int).tolist()
                    ends = batch["segment_end_frames"][i].detach().cpu().numpy().astype(int).tolist()
                    rec.update(
                        {
                            "segment_scores": seg_probs.tolist(),
                            "selected_segment_indices": selected_idx,
                            "segment_start_frames": starts,
                            "segment_end_frames": ends,
                        }
                    )

                records.append(rec)

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_score_arr = np.array(y_score, dtype=np.float64)

    auc = safe_auc(y_true_arr, y_score_arr)
    ap = safe_ap(y_true_arr, y_score_arr)

    if y_true_arr.size == 0:
        cm = [[0, 0], [0, 0]]
    else:
        preds = (y_score_arr >= threshold).astype(np.int64)
        cm = confusion_matrix(y_true_arr, preds, labels=[0, 1]).tolist()

    return {
        "auc": auc,
        "ap": ap,
        "confusion_matrix": cm,
        "num_samples": int(len(y_true_arr)),
        "records": records,
    }


def save_history(history: List[EpochResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bce_loss", "train_rtfm_loss", "val_auc", "val_ap"])
        for e in history:
            writer.writerow([e.epoch, e.train_loss, e.train_bce_loss, e.train_rtfm_loss, e.val_auc, e.val_ap])


def maybe_plot_curves(history: List[EpochResult], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [x.epoch for x in history]
    train_loss = [x.train_loss for x in history]
    val_auc = [x.val_auc for x in history]
    val_ap = [x.val_ap for x in history]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_auc, label="val_auc", color="tab:green")
    ax2.plot(epochs, val_ap, label="val_ap", color="tab:orange")
    ax2.set_ylabel("val metrics", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    feature_manifest = resolve(project_root, args.feature_manifest)
    output_dir = resolve(project_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)

    train_ds = FeatureSequenceDataset(
        manifest_csv=feature_manifest,
        project_root=project_root,
        split="train",
        target_segments=args.target_segments,
        require_ok_status=True,
    )
    val_ds = FeatureSequenceDataset(
        manifest_csv=feature_manifest,
        project_root=project_root,
        split="val",
        target_segments=args.target_segments,
        require_ok_status=True,
    )
    test_ds = FeatureSequenceDataset(
        manifest_csv=feature_manifest,
        project_root=project_root,
        split="test",
        target_segments=args.target_segments,
        require_ok_status=True,
    )

    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampler=args.balanced_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=fixed_segments_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=fixed_segments_collate,
    )

    model = RTFMBaseline(input_dim=2048, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    train_labels = np.array([int(r["binary_label"]) for r in train_ds.rows], dtype=np.int64)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    if args.balanced_sampler:
        # Avoid double re-weighting positives when class-balanced sampling is enabled.
        pos_weight = 1.0

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[EpochResult] = []
    best_key = (-float("inf"), -float("inf"))  # (val_auc, val_ap)
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bce, train_rtfm = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce_fn=bce_fn,
            device=device,
            topk_ratio=args.topk_ratio,
            rtfm_margin=args.rtfm_margin,
            rtfm_lambda=args.rtfm_lambda,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            topk_ratio=args.topk_ratio,
            threshold=args.threshold,
            collect_segments=False,
        )

        val_auc = float(val_metrics["auc"])
        val_ap = float(val_metrics["ap"])

        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_bce_loss=train_bce,
                train_rtfm_loss=train_rtfm,
                val_auc=val_auc,
                val_ap=val_ap,
            )
        )

        cur_key = (
            -1e9 if math.isnan(val_auc) else val_auc,
            -1e9 if math.isnan(val_ap) else val_ap,
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"(bce={train_bce:.5f}, rtfm={train_rtfm:.5f}) | "
            f"val_auc={val_auc:.5f} val_ap={val_ap:.5f}"
        )

        if cur_key > best_key:
            best_key = cur_key
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": _json_safe_args(args),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                },
                ckpt_dir / "best.pt",
            )

    save_history(history, output_dir / "train_history.csv")
    maybe_plot_curves(history, output_dir / "train_curves.png")

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        topk_ratio=args.topk_ratio,
        threshold=args.threshold,
        collect_segments=True,
    )

    test_records: List[Dict[str, object]] = test_metrics["records"]
    normal_scores = sorted(
        [r for r in test_records if int(r["binary_label"]) == 0],
        key=lambda x: str(x["video_id"]),
    )
    anomaly_scores = sorted(
        [r for r in test_records if int(r["binary_label"]) == 1],
        key=lambda x: str(x["video_id"]),
    )

    anomaly_for_qual = sorted(anomaly_scores, key=lambda x: float(x["score"]), reverse=True)[:3]
    qualitative = []
    for rec in anomaly_for_qual:
        seg_scores = np.array(rec["segment_scores"], dtype=np.float64)
        top_idx = np.argsort(seg_scores)[-5:][::-1]
        qualitative.append(
            {
                "video_id": rec["video_id"],
                "category_label": rec["category_label"],
                "num_segments": int(rec["original_num_segments"]),
                "top_segment_indices": top_idx.tolist(),
                "top_segment_scores": [float(seg_scores[i]) for i in top_idx],
                "top_segment_original_indices": [int(rec["selected_segment_indices"][i]) for i in top_idx],
                "top_segment_start_frames": [int(rec["segment_start_frames"][i]) for i in top_idx],
                "top_segment_end_frames": [int(rec["segment_end_frames"][i]) for i in top_idx],
            }
        )

    results = {
        "architecture": {
            "input_shape": ["B", args.target_segments, 2048],
            "model": "RTFMBaseline(feature_proj + segment_head)",
            "feature_proj": [2048, args.hidden_dim],
            "segment_head": [args.hidden_dim, args.hidden_dim // 2, 1],
            "temporal_sampling_rule": (
                "uniform sample to target_segments if longer; pad by repeating last segment if shorter"
            ),
            "aggregation_rule": f"video logit = mean(top-k segment logits), k=ceil(T*{args.topk_ratio})",
            "rtfm_magnitude_rule": (
                f"relu(margin - (mean_topk_mag_anomaly - mean_topk_mag_normal)); margin={args.rtfm_margin}"
            ),
        },
        "training_setup": {
            "loss": {
                "bce": "BCEWithLogitsLoss(video_logit, binary_label)",
                "rtfm": "magnitude separation loss",
                "total": f"bce + {args.rtfm_lambda} * rtfm",
            },
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "checkpoint_selection": "best val_auc, tie-break by val_ap",
            "balanced_sampler": bool(args.balanced_sampler),
            "pos_weight": pos_weight,
            "device": device,
            "seed": args.seed,
        },
        "training_curves": [
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss,
                "train_bce_loss": h.train_bce_loss,
                "train_rtfm_loss": h.train_rtfm_loss,
                "val_auc": h.val_auc,
                "val_ap": h.val_ap,
            }
            for h in history
        ],
        "best_epoch": best_epoch,
        "val_best": {
            "auc": best_ckpt.get("val_auc"),
            "ap": best_ckpt.get("val_ap"),
        },
        "test": {
            "auc": float(test_metrics["auc"]),
            "ap": float(test_metrics["ap"]),
            "confusion_matrix_threshold": args.threshold,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "normal_score_examples": [
                {"video_id": r["video_id"], "category_label": r["category_label"], "score": r["score"]}
                for r in normal_scores[:5]
            ],
            "anomaly_score_examples": [
                {"video_id": r["video_id"], "category_label": r["category_label"], "score": r["score"]}
                for r in anomaly_scores[:5]
            ],
        },
        "qualitative_top_segments": qualitative,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_records.json").write_text(json.dumps(test_records, indent=2) + "\n")

    print("\nFinal test metrics")
    print(f"- AUC: {results['test']['auc']:.6f}")
    print(f"- AP:  {results['test']['ap']:.6f}")
    print(f"- Confusion matrix @ {args.threshold}: {results['test']['confusion_matrix']}")
    print(f"- Saved: {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
