#!/usr/bin/env python3
"""Train/evaluate RTFM + Classifier + TRN + Boundary head (Step 7)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from feature_dataset import FeatureSequenceDataset, fixed_segments_collate

SEGMENT_LEN = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-7 TRN + Boundary training/evaluation")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    parser.add_argument(
        "--master-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("data/ucf_crime/annotations/temporal_segments"),
    )
    parser.add_argument(
        "--init-ckpt",
        type=Path,
        default=Path("outputs/rtfm_trn/checkpoints/best.pt"),
        help="Step-6 checkpoint used for initialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rtfm_trn_boundary"),
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
    parser.add_argument("--pseudo-topk", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--rtfm-margin", type=float, default=5.0)
    parser.add_argument("--rtfm-lambda", type=float, default=0.1)
    parser.add_argument("--cls-lambda", type=float, default=0.5)
    parser.add_argument("--bnd-lambda", type=float, default=0.3)
    parser.add_argument("--smooth-lambda", type=float, default=0.1)

    parser.add_argument("--trn-layers", type=int, default=2)
    parser.add_argument("--trn-heads", type=int, default=4)
    parser.add_argument("--trn-ffn-mult", type=int, default=4)
    parser.add_argument("--trn-dropout", type=float, default=0.1)
    parser.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    parser.add_argument("--infer-window", type=int, default=32)
    parser.add_argument("--infer-stride", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-event-len", type=int, default=1)
    parser.add_argument("--merge-gap", type=int, default=0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--threshold-candidates", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--localization-tiou", type=str, default="0.3,0.5,0.7")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument(
        "--class-names",
        type=str,
        default="normal,fighting,shooting,explosion,robbery,abuse",
    )
    parser.add_argument("--normal-class", type=str, default="normal")
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_macro_f1", "val_weighted_f1", "val_auc"],
        default="val_macro_f1",
    )
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


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def parse_class_names(raw: str) -> List[str]:
    classes = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not classes:
        raise ValueError("class list is empty")
    if len(classes) != len(set(classes)):
        raise ValueError(f"duplicate classes detected: {classes}")
    return classes


def parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for x in raw.split(","):
        s = x.strip()
        if s:
            vals.append(float(s))
    if not vals:
        raise ValueError(f"No numeric values parsed from: {raw}")
    return vals


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def topk_mean(values: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, int]:
    t = values.shape[1]
    k = max(1, int(math.ceil(t * ratio)))
    topk_vals = torch.topk(values, k=k, dim=1).values
    return topk_vals.mean(dim=1), k


def topk_mean_np(values: np.ndarray, ratio: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(values.shape[0] * ratio)))
    idx = np.argpartition(values, -k)[-k:]
    return float(values[idx].mean())


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def metric_key(metric: float) -> float:
    return -1e9 if math.isnan(metric) else metric


def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TRNEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.self_attn(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_w if return_attention else None


class RTFMTRNBoundary(nn.Module):
    """Feature projection -> TRN refinement -> anomaly/class/boundary heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        target_segments: int,
        trn_layers: int,
        trn_heads: int,
        trn_ffn_mult: int,
        trn_dropout: float,
        proj_dropout: float,
        pos_encoding: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_segments = target_segments
        self.pos_encoding_type = pos_encoding

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout),
        )

        if pos_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, target_segments, hidden_dim))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
            self.register_buffer("sinusoidal_pe", torch.empty(0), persistent=False)
        else:
            pe = sinusoidal_positional_encoding(target_segments, hidden_dim)
            self.register_buffer("sinusoidal_pe", pe.unsqueeze(0), persistent=False)
            self.pos_embedding = None

        self.trn_layers = nn.ModuleList(
            [
                TRNEncoderLayer(
                    dim=hidden_dim,
                    num_heads=trn_heads,
                    ffn_dim=hidden_dim * trn_ffn_mult,
                    dropout=trn_dropout,
                )
                for _ in range(trn_layers)
            ]
        )

        self.segment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if self.pos_encoding_type == "learned":
            if t <= self.target_segments:
                return x + self.pos_embedding[:, :t, :]
            # interpolate learned embeddings for longer sequences (rare in training; useful in fallback)
            pe = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=t,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            return x + pe
        return x + self.sinusoidal_pe[:, :t, :].to(dtype=x.dtype, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z = self.feature_proj(x)
        z = self.add_positional_encoding(z)

        attn_maps: List[torch.Tensor] = []
        for layer in self.trn_layers:
            z, attn_w = layer(z, return_attention=return_attention)
            if return_attention and attn_w is not None:
                attn_maps.append(attn_w)

        seg_anom_logits = self.segment_head(z).squeeze(-1)
        seg_class_logits = self.class_head(z)
        feat_magnitudes = torch.norm(z, p=2, dim=-1)

        if z.shape[1] > 1:
            s_prob = torch.sigmoid(seg_anom_logits)
            diff = torch.abs(s_prob[:, 1:] - s_prob[:, :-1]).unsqueeze(-1)
            edge_feat = torch.cat([z[:, :-1, :], z[:, 1:, :], diff], dim=-1)
            bnd_scores = self.boundary_head(edge_feat).squeeze(-1)
        else:
            bnd_scores = torch.zeros((z.shape[0], 0), device=z.device, dtype=z.dtype)

        return seg_anom_logits, seg_class_logits, feat_magnitudes, bnd_scores, z, attn_maps


def smoothness_loss(seg_logits: torch.Tensor) -> torch.Tensor:
    if seg_logits.shape[1] < 2:
        return torch.tensor(0.0, device=seg_logits.device)
    s = torch.sigmoid(seg_logits)
    return torch.mean((s[:, 1:] - s[:, :-1]) ** 2)


def boundary_loss(seg_logits: torch.Tensor, bnd_scores: torch.Tensor) -> torch.Tensor:
    if seg_logits.shape[1] < 2:
        return torch.tensor(0.0, device=seg_logits.device)
    target = torch.abs(torch.sigmoid(seg_logits[:, 1:]) - torch.sigmoid(seg_logits[:, :-1])).detach()
    return torch.mean((target - bnd_scores) ** 2)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_bce_loss: float
    train_rtfm_loss: float
    train_cls_loss: float
    train_bnd_loss: float
    train_smooth_loss: float
    val_auc: float
    val_ap: float
    val_macro_f1: float
    val_weighted_f1: float


def train_one_epoch(
    model: RTFMTRNBoundary,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_fn: nn.Module,
    device: str,
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    pseudo_topk: int,
    rtfm_margin: float,
    rtfm_lambda: float,
    cls_lambda: float,
    bnd_lambda: float,
    smooth_lambda: float,
) -> Tuple[float, float, float, float, float, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_rtfm = 0.0
    total_cls = 0.0
    total_bnd = 0.0
    total_smooth = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["features"].to(device)
        y_bin = batch["binary_labels"].float().to(device)

        seg_anom_logits, seg_class_logits, feat_mags, bnd_scores, _, _ = model(x, return_attention=False)

        video_logits, _ = topk_mean(seg_anom_logits, topk_ratio)
        bce_loss = bce_fn(video_logits, y_bin)

        rtfm_loss = torch.tensor(0.0, device=device)
        anom_mask = y_bin > 0.5
        norm_mask = y_bin <= 0.5
        if anom_mask.any() and norm_mask.any():
            anom_topk, _ = topk_mean(feat_mags[anom_mask], topk_ratio)
            norm_topk, _ = topk_mean(feat_mags[norm_mask], topk_ratio)
            rtfm_loss = torch.relu(rtfm_margin - (anom_topk.mean() - norm_topk.mean()))

        bsz, tlen, _ = seg_class_logits.shape
        cls_losses: List[torch.Tensor] = []
        for i in range(bsz):
            cat = str(batch["category_labels"][i]).lower()
            if cat not in class_to_idx:
                continue

            if int(y_bin[i].item()) == 1 and class_to_idx[cat] != normal_idx:
                k = min(pseudo_topk, tlen)
                top_idx = torch.topk(seg_anom_logits[i], k=k).indices
                seg_logits_sel = seg_class_logits[i, top_idx, :]
                targets = torch.full((k,), class_to_idx[cat], device=device, dtype=torch.long)
                cls_losses.append(F.cross_entropy(seg_logits_sel, targets))
            else:
                seg_logits_all = seg_class_logits[i]
                targets = torch.full((tlen,), normal_idx, device=device, dtype=torch.long)
                cls_losses.append(F.cross_entropy(seg_logits_all, targets))

        cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
        bnd_loss = boundary_loss(seg_anom_logits, bnd_scores)
        s_loss = smoothness_loss(seg_anom_logits)

        anomaly_loss = bce_loss + rtfm_lambda * rtfm_loss
        loss = anomaly_loss + cls_lambda * cls_loss + bnd_lambda * bnd_loss + smooth_lambda * s_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_rtfm += float(rtfm_loss.item())
        total_cls += float(cls_loss.item())
        total_bnd += float(bnd_loss.item())
        total_smooth += float(s_loss.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_bce / n_batches,
        total_rtfm / n_batches,
        total_cls / n_batches,
        total_bnd / n_batches,
        total_smooth / n_batches,
    )


def evaluate_sampled(
    model: RTFMTRNBoundary,
    loader: DataLoader,
    device: str,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    pseudo_topk: int,
    threshold: float,
) -> Dict[str, object]:
    model.eval()

    y_true_bin: List[int] = []
    y_score_bin: List[float] = []
    y_true_cls: List[int] = []
    y_pred_cls: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_bin = batch["binary_labels"].to(device)

            seg_anom_logits, seg_class_logits, _, _, _, _ = model(x, return_attention=False)
            video_logits, _ = topk_mean(seg_anom_logits, topk_ratio)
            video_anom_probs = torch.sigmoid(video_logits)

            bsz, tlen, _ = seg_class_logits.shape
            for i in range(bsz):
                gt_binary = int(y_bin[i].item())
                gt_label = str(batch["category_labels"][i]).lower()
                gt_class = class_to_idx.get(gt_label, normal_idx)

                anom_prob = float(video_anom_probs[i].item())
                y_true_bin.append(gt_binary)
                y_score_bin.append(anom_prob)

                k = min(pseudo_topk, tlen)
                top_idx = torch.topk(seg_anom_logits[i], k=k).indices
                seg_cls_probs = torch.softmax(seg_class_logits[i, top_idx, :], dim=-1)
                video_cls_probs = seg_cls_probs.mean(dim=0)

                raw_pred = int(torch.argmax(video_cls_probs).item())
                pred_class = normal_idx if anom_prob < threshold else raw_pred
                y_true_cls.append(gt_class)
                y_pred_cls.append(pred_class)

    y_true_bin_arr = np.array(y_true_bin, dtype=np.int64)
    y_score_bin_arr = np.array(y_score_bin, dtype=np.float64)
    binary_auc = safe_auc(y_true_bin_arr, y_score_bin_arr)
    binary_ap = safe_ap(y_true_bin_arr, y_score_bin_arr)

    labels = list(range(len(class_names)))
    if len(y_true_cls) == 0:
        macro_f1 = float("nan")
        weighted_f1 = float("nan")
    else:
        y_true_cls_arr = np.array(y_true_cls, dtype=np.int64)
        y_pred_cls_arr = np.array(y_pred_cls, dtype=np.int64)
        macro_f1 = float(f1_score(y_true_cls_arr, y_pred_cls_arr, labels=labels, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_true_cls_arr, y_pred_cls_arr, labels=labels, average="weighted", zero_division=0)
        )

    return {
        "binary": {"auc": binary_auc, "ap": binary_ap},
        "classification": {"macro_f1": macro_f1, "weighted_f1": weighted_f1},
    }


def sliding_window_starts(total_len: int, window: int, stride: int) -> List[int]:
    if total_len <= window:
        return [0]
    starts = list(range(0, total_len - window + 1, stride))
    last = total_len - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def infer_full_sequence_chunked(
    model: RTFMTRNBoundary,
    features: np.ndarray,
    device: str,
    num_classes: int,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = int(features.shape[0])
    if t <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, num_classes), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    sum_anom = np.zeros((t,), dtype=np.float64)
    cnt_anom = np.zeros((t,), dtype=np.float64)
    sum_cls = np.zeros((t, num_classes), dtype=np.float64)
    cnt_cls = np.zeros((t,), dtype=np.float64)

    if t > 1:
        sum_bnd = np.zeros((t - 1,), dtype=np.float64)
        cnt_bnd = np.zeros((t - 1,), dtype=np.float64)
    else:
        sum_bnd = np.zeros((0,), dtype=np.float64)
        cnt_bnd = np.zeros((0,), dtype=np.float64)

    starts = sliding_window_starts(t, window, stride)

    model.eval()
    with torch.no_grad():
        for s in starts:
            e = min(s + window, t)
            chunk = features[s:e]
            real_len = int(e - s)

            if real_len < window:
                pad = np.repeat(chunk[-1:, :], repeats=(window - real_len), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)

            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            seg_logits, seg_cls_logits, _, bnd_scores, _, _ = model(x, return_attention=False)
            seg_scores = torch.sigmoid(seg_logits)[0].detach().cpu().numpy().astype(np.float64)[:real_len]
            cls_probs = torch.softmax(seg_cls_logits, dim=-1)[0].detach().cpu().numpy().astype(np.float64)[:real_len]

            sum_anom[s:e] += seg_scores
            cnt_anom[s:e] += 1.0
            sum_cls[s:e, :] += cls_probs
            cnt_cls[s:e] += 1.0

            if real_len >= 2 and bnd_scores.shape[1] > 0:
                bnd = bnd_scores[0].detach().cpu().numpy().astype(np.float64)[: real_len - 1]
                sum_bnd[s : e - 1] += bnd
                cnt_bnd[s : e - 1] += 1.0

    anom = np.divide(sum_anom, np.maximum(cnt_anom, 1e-8), out=np.zeros_like(sum_anom), where=cnt_anom > 0)
    cls = np.divide(
        sum_cls,
        np.maximum(cnt_cls[:, None], 1e-8),
        out=np.zeros_like(sum_cls),
        where=cnt_cls[:, None] > 0,
    )

    if t > 1:
        bnd = np.divide(sum_bnd, np.maximum(cnt_bnd, 1e-8), out=np.zeros_like(sum_bnd), where=cnt_bnd > 0)
    else:
        bnd = np.zeros((0,), dtype=np.float64)

    return anom.astype(np.float32), cls.astype(np.float32), bnd.astype(np.float32)


def moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    if scores.size == 0:
        return scores
    if window <= 1:
        return scores.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    smooth = np.convolve(padded, kernel, mode="valid")
    return smooth.astype(np.float32)


def spans_from_scores(scores: np.ndarray, threshold: float, min_len: int, merge_gap: int) -> List[Tuple[int, int]]:
    if scores.size == 0:
        return []

    mask = scores >= threshold
    spans: List[Tuple[int, int]] = []

    i = 0
    n = int(mask.shape[0])
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        e = i
        spans.append((s, e))
        i += 1

    if merge_gap > 0 and spans:
        merged: List[Tuple[int, int]] = [spans[0]]
        for s, e in spans[1:]:
            ps, pe = merged[-1]
            if s - pe - 1 <= merge_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        spans = merged

    spans = [(s, e) for (s, e) in spans if (e - s + 1) >= min_len]
    return spans


def refine_spans_with_boundary(
    spans: Sequence[Tuple[int, int]],
    boundary_scores: np.ndarray,
    total_segments: int,
    radius: int,
    min_len: int,
    merge_gap: int,
) -> List[Tuple[int, int]]:
    if not spans:
        return []
    if boundary_scores.size == 0:
        return list(spans)

    refined: List[Tuple[int, int]] = []
    max_edge = int(boundary_scores.shape[0] - 1)

    for s, e in spans:
        start_center = max(0, s - 1)
        sl = max(0, start_center - radius)
        sr = min(max_edge, start_center + radius)
        start_edge = sl + int(np.argmax(boundary_scores[sl : sr + 1]))
        s_ref = int(start_edge + 1)

        end_center = min(max_edge, e)
        el = max(0, end_center - radius)
        er = min(max_edge, end_center + radius)
        end_edge = el + int(np.argmax(boundary_scores[el : er + 1]))
        e_ref = int(end_edge)

        s_ref = max(0, min(s_ref, total_segments - 1))
        e_ref = max(0, min(e_ref, total_segments - 1))
        if s_ref > e_ref:
            s_ref, e_ref = s, e

        refined.append((s_ref, e_ref))

    refined.sort(key=lambda x: x[0])
    if merge_gap > 0 and refined:
        merged: List[Tuple[int, int]] = [refined[0]]
        for s, e in refined[1:]:
            ps, pe = merged[-1]
            if s - pe - 1 <= merge_gap:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        refined = merged

    refined = [(s, e) for (s, e) in refined if (e - s + 1) >= min_len]
    return refined


def spans_to_events(
    spans: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    class_probs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    fps: float,
    class_names: List[str],
    normal_idx: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if scores.size == 0:
        return out

    anomaly_indices = [i for i in range(len(class_names)) if i != normal_idx]
    for s, e in spans:
        span_scores = scores[s : e + 1]
        span_cls = class_probs[s : e + 1]
        cls_mean = span_cls.mean(axis=0)

        if anomaly_indices:
            sub = cls_mean[anomaly_indices]
            j = int(np.argmax(sub))
            pred_idx = int(anomaly_indices[j])
        else:
            pred_idx = int(np.argmax(cls_mean))

        cls_conf = float(cls_mean[pred_idx])
        anom_conf = float(span_scores.mean())
        event_score = float(anom_conf * cls_conf)

        s_frame = int(segment_starts[s])
        e_frame = int(segment_ends[e])
        start_time = float(s_frame / fps) if fps > 0 else 0.0
        end_time = float(e_frame / fps) if fps > 0 else 0.0

        out.append(
            {
                "start_segment": int(s),
                "end_segment": int(e),
                "start_time": start_time,
                "end_time": end_time,
                "predicted_class": class_names[pred_idx],
                "predicted_class_idx": pred_idx,
                "event_score": event_score,
                "class_confidence": cls_conf,
                "anomaly_confidence": anom_conf,
            }
        )

    out.sort(key=lambda x: float(x["event_score"]), reverse=True)
    return out


def temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter <= 0:
        return 0.0
    la = a[1] - a[0] + 1
    lb = b[1] - b[0] + 1
    union = la + lb - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def detection_ap_at_tiou(
    pred_events: List[Dict[str, object]],
    gt_events_by_video: Dict[str, List[Dict[str, object]]],
    tiou: float,
) -> float:
    total_gt = sum(len(v) for v in gt_events_by_video.values())
    if total_gt <= 0:
        return float("nan")

    preds = sorted(pred_events, key=lambda x: float(x["event_score"]), reverse=True)
    if not preds:
        return 0.0

    matched: Dict[str, np.ndarray] = {
        vid: np.zeros((len(gts),), dtype=bool) for vid, gts in gt_events_by_video.items()
    }

    tp = np.zeros((len(preds),), dtype=np.float64)
    fp = np.zeros((len(preds),), dtype=np.float64)

    for i, p in enumerate(preds):
        vid = str(p["video_id"])
        gts = gt_events_by_video.get(vid, [])
        if not gts:
            fp[i] = 1.0
            continue

        best_iou = 0.0
        best_j = -1
        p_span = (int(p["start_segment"]), int(p["end_segment"]))
        for j, g in enumerate(gts):
            if matched[vid][j]:
                continue
            g_span = (int(g["start_segment"]), int(g["end_segment"]))
            iou = temporal_iou(p_span, g_span)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= tiou:
            tp[i] = 1.0
            matched[vid][best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / float(total_gt)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return compute_ap_from_pr(rec, prec)


def evaluate_localization_map(
    pred_events_all: List[Dict[str, object]],
    gt_events_all: List[Dict[str, object]],
    class_names: List[str],
    normal_idx: int,
    tiou_thresholds: List[float],
) -> Dict[str, object]:
    anomaly_classes = [c for i, c in enumerate(class_names) if i != normal_idx]

    gt_by_class_video: Dict[str, Dict[str, List[Dict[str, object]]]] = {c: {} for c in anomaly_classes}
    for g in gt_events_all:
        c = str(g["class_label"])
        vid = str(g["video_id"])
        gt_by_class_video[c].setdefault(vid, []).append(g)

    pred_by_class: Dict[str, List[Dict[str, object]]] = {c: [] for c in anomaly_classes}
    for p in pred_events_all:
        c = str(p["predicted_class"])
        if c in pred_by_class:
            pred_by_class[c].append(p)

    out = {
        "tiou": {},
        "gt_event_count": int(len(gt_events_all)),
        "pred_event_count": int(len(pred_events_all)),
        "classes": anomaly_classes,
    }

    for thr in tiou_thresholds:
        aps: List[float] = []
        per_class: Dict[str, float] = {}
        for c in anomaly_classes:
            ap = detection_ap_at_tiou(
                pred_events=pred_by_class[c],
                gt_events_by_video=gt_by_class_video[c],
                tiou=thr,
            )
            per_class[c] = ap
            if not math.isnan(ap):
                aps.append(ap)

        m_ap = float(np.mean(aps)) if aps else float("nan")
        out["tiou"][str(thr)] = {
            "mAP": m_ap,
            "per_class_ap": per_class,
        }

    return out


def infer_video_full(
    model: RTFMTRNBoundary,
    row: Dict[str, str],
    project_root: Path,
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    window: int,
    stride: int,
    smooth_window: int,
    threshold: float,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
    device: str,
) -> Dict[str, object]:
    feat_path = resolve(project_root, Path(row["feature_path"]))
    with np.load(feat_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        starts = data["segment_start_frames"].astype(np.int64)
        ends = data["segment_end_frames"].astype(np.int64)

    seg_scores, cls_probs, bnd_scores = infer_full_sequence_chunked(
        model=model,
        features=feats,
        device=device,
        num_classes=len(class_names),
        window=window,
        stride=stride,
    )

    seg_scores_smooth = moving_average(seg_scores, smooth_window)

    t = int(seg_scores_smooth.shape[0])
    spans_before = spans_from_scores(seg_scores_smooth, threshold=threshold, min_len=min_event_len, merge_gap=merge_gap)
    spans_after = refine_spans_with_boundary(
        spans=spans_before,
        boundary_scores=bnd_scores,
        total_segments=t,
        radius=boundary_radius,
        min_len=min_event_len,
        merge_gap=merge_gap,
    )

    fps = float(row["fps"])
    events_before = spans_to_events(
        spans=spans_before,
        scores=seg_scores_smooth,
        class_probs=cls_probs,
        segment_starts=starts,
        segment_ends=ends,
        fps=fps,
        class_names=class_names,
        normal_idx=normal_idx,
    )
    events_after = spans_to_events(
        spans=spans_after,
        scores=seg_scores_smooth,
        class_probs=cls_probs,
        segment_starts=starts,
        segment_ends=ends,
        fps=fps,
        class_names=class_names,
        normal_idx=normal_idx,
    )

    video_anomaly_score = topk_mean_np(seg_scores_smooth, topk_ratio)

    if video_anomaly_score < threshold or not events_after:
        pred_class_idx = normal_idx
        pred_class = class_names[pred_class_idx]
        pred_class_prob = 1.0 - video_anomaly_score
    else:
        best_event = events_after[0]
        pred_class = str(best_event["predicted_class"])
        pred_class_idx = int(best_event["predicted_class_idx"])
        pred_class_prob = float(best_event["class_confidence"])

    topk_bnd = []
    if bnd_scores.size > 0:
        k = int(min(5, bnd_scores.shape[0]))
        idx = np.argpartition(bnd_scores, -k)[-k:]
        idx = idx[np.argsort(-bnd_scores[idx])]
        topk_bnd = [
            {"edge_index": int(i), "value": float(bnd_scores[i])}
            for i in idx.tolist()
        ]

    out = {
        "video_id": row["video_id"],
        "split": row["split"],
        "binary_label": int(row["binary_label"]),
        "category_label": str(row["category_label"]),
        "num_segments": int(feats.shape[0]),
        "fps": fps,
        "duration_sec": float(row["duration_sec"]),
        "video_anomaly_score": float(video_anomaly_score),
        "pred_video_class": pred_class,
        "pred_video_class_idx": pred_class_idx,
        "pred_video_class_prob": float(pred_class_prob),
        "spans_before_refine": [[int(s), int(e)] for s, e in spans_before],
        "spans_after_refine": [[int(s), int(e)] for s, e in spans_after],
        "events_before_refine": events_before,
        "events_after_refine": events_after,
        "top_boundary_peaks": topk_bnd,
        "segment_scores_smooth": seg_scores_smooth.tolist(),
        "boundary_scores": bnd_scores.tolist(),
    }
    return out


def tune_threshold_on_val(video_results: List[Dict[str, object]], candidates: List[float]) -> Dict[str, object]:
    if not video_results:
        return {"best_threshold": 0.5, "best_f1": float("nan"), "num_videos": 0, "rows": []}

    y_true = np.array([int(r["binary_label"]) for r in video_results], dtype=np.int64)
    y_score = np.array([float(r["video_anomaly_score"]) for r in video_results], dtype=np.float64)

    rows = []
    best_thr = float(candidates[0])
    best_f1 = -1.0

    for thr in candidates:
        y_pred = (y_score >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append({"threshold": float(thr), "binary_f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return {
        "best_threshold": best_thr,
        "best_f1": best_f1,
        "num_videos": int(len(video_results)),
        "rows": rows,
    }


def load_gt_events_for_test(
    test_rows: List[Dict[str, str]],
    master_rows_by_video: Dict[str, Dict[str, str]],
    temporal_root: Path,
) -> List[Dict[str, object]]:
    gt_events: List[Dict[str, object]] = []

    for r in test_rows:
        vid = str(r["video_id"])
        if int(r["binary_label"]) == 0:
            continue

        m = master_rows_by_video.get(vid, {})
        ann_path_raw = str(m.get("temporal_annotation_path", "")).strip()
        ann_path = temporal_root / f"{vid}.json"
        if ann_path_raw:
            p = Path(ann_path_raw)
            ann_path = p if p.is_absolute() else temporal_root.parent.parent / p
        if not ann_path.exists():
            fallback = temporal_root / f"{vid}.json"
            if fallback.exists():
                ann_path = fallback
            else:
                continue

        d = json.loads(ann_path.read_text())
        segs = d.get("segments", []) if isinstance(d, dict) else []
        num_segments = int(float(r.get("num_segments", "0")))

        for seg in segs:
            sf = int(seg.get("start_frame", 0))
            ef = int(seg.get("end_frame", 0))
            s = max(0, sf // SEGMENT_LEN)
            e = max(s, ef // SEGMENT_LEN)
            if num_segments > 0:
                s = min(s, num_segments - 1)
                e = min(e, num_segments - 1)
            gt_events.append(
                {
                    "video_id": vid,
                    "class_label": str(r["category_label"]),
                    "start_segment": int(s),
                    "end_segment": int(e),
                    "start_frame": sf,
                    "end_frame": ef,
                }
            )

    return gt_events


def best_iou_against_gt(spans: Sequence[Tuple[int, int]], gt_events: Sequence[Dict[str, object]]) -> float:
    if not spans or not gt_events:
        return 0.0
    best = 0.0
    for s, e in spans:
        for g in gt_events:
            iou = temporal_iou((int(s), int(e)), (int(g["start_segment"]), int(g["end_segment"])))
            if iou > best:
                best = iou
    return best


def choose_failure_case(
    test_results: List[Dict[str, object]],
    gt_by_video: Dict[str, List[Dict[str, object]]],
    threshold: float,
) -> Optional[Dict[str, object]]:
    cands = []
    for r in test_results:
        if int(r["binary_label"]) == 0:
            continue
        vid = str(r["video_id"])
        gt = gt_by_video.get(vid, [])
        if not gt:
            continue
        before_spans = [tuple(x) for x in r.get("spans_before_refine", [])]
        after_spans = [tuple(x) for x in r.get("spans_after_refine", [])]
        iou_after = best_iou_against_gt(after_spans, gt)
        score = float(r.get("video_anomaly_score", 0.0))
        if score >= threshold and iou_after < 0.3:
            if len(after_spans) == 0:
                reason = "high anomaly score but no localized span after post-processing"
            elif len(after_spans) > len(gt):
                reason = "high anomaly score but fragmented/over-predicted spans"
            else:
                reason = "high anomaly score but boundaries do not align with GT interval"
            cands.append((score, {
                "video_id": vid,
                "ground_truth_class": r.get("category_label"),
                "video_anomaly_score": score,
                "gt_spans": [[int(g["start_segment"]), int(g["end_segment"])] for g in gt],
                "pred_spans_after_refine": [[int(a), int(b)] for (a, b) in after_spans],
                "reason_guess": reason,
            }))

    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def choose_success_case(
    test_results: List[Dict[str, object]],
    gt_by_video: Dict[str, List[Dict[str, object]]],
) -> Optional[Dict[str, object]]:
    cands = []
    for r in test_results:
        if int(r["binary_label"]) == 0:
            continue
        vid = str(r["video_id"])
        gt = gt_by_video.get(vid, [])
        if not gt:
            continue

        before_spans = [tuple(x) for x in r.get("spans_before_refine", [])]
        after_spans = [tuple(x) for x in r.get("spans_after_refine", [])]
        iou_before = best_iou_against_gt(before_spans, gt)
        iou_after = best_iou_against_gt(after_spans, gt)
        gain = iou_after - iou_before
        if gain > 1e-6:
            cands.append((gain, {
                "video_id": vid,
                "ground_truth_class": r.get("category_label"),
                "gt_spans": [[int(g["start_segment"]), int(g["end_segment"])] for g in gt],
                "pred_spans_before_refine": [[int(a), int(b)] for (a, b) in before_spans],
                "pred_spans_after_refine": [[int(a), int(b)] for (a, b) in after_spans],
                "best_iou_before": float(iou_before),
                "best_iou_after": float(iou_after),
                "improvement": float(gain),
            }))

    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def load_checkpoint_weights(model: RTFMTRNBoundary, ckpt_path: Path, device: str) -> Dict[str, object]:
    if not ckpt_path.exists():
        return {"loaded": False, "reason": f"checkpoint not found: {ckpt_path}"}

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    model_state = model.state_dict()
    copied = []
    updated = dict(model_state)
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            updated[k] = v
            copied.append(k)

    model.load_state_dict(updated)

    return {
        "loaded": True,
        "checkpoint": str(ckpt_path),
        "num_copied": len(copied),
        "copied_keys_preview": copied[:20],
        "num_model_params": len(model_state),
        "num_new_params_uninitialized": len(model_state) - len(copied),
    }


def load_json_if_exists(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_model_metrics(summary_obj: Optional[Dict[str, object]], fallback_name: str) -> Dict[str, object]:
    if summary_obj is None:
        return {
            "model": fallback_name,
            "auc": None,
            "ap": None,
            "macro_f1": None,
            "weighted_f1": None,
        }
    test = summary_obj.get("test", {})
    if "binary" in test and "classification" in test:
        return {
            "model": fallback_name,
            "auc": test["binary"].get("auc"),
            "ap": test["binary"].get("ap"),
            "macro_f1": test["classification"].get("macro_f1"),
            "weighted_f1": test["classification"].get("weighted_f1"),
        }
    return {
        "model": fallback_name,
        "auc": test.get("auc"),
        "ap": test.get("ap"),
        "macro_f1": None,
        "weighted_f1": None,
    }


def save_history(history: List[EpochResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_bce_loss",
                "train_rtfm_loss",
                "train_cls_loss",
                "train_bnd_loss",
                "train_smooth_loss",
                "val_auc",
                "val_ap",
                "val_macro_f1",
                "val_weighted_f1",
            ]
        )
        for h in history:
            writer.writerow(
                [
                    h.epoch,
                    h.train_loss,
                    h.train_bce_loss,
                    h.train_rtfm_loss,
                    h.train_cls_loss,
                    h.train_bnd_loss,
                    h.train_smooth_loss,
                    h.val_auc,
                    h.val_ap,
                    h.val_macro_f1,
                    h.val_weighted_f1,
                ]
            )


def maybe_plot_curves(history: List[EpochResult], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h.epoch for h in history]
    train_loss = [h.train_loss for h in history]
    val_auc = [h.val_auc for h in history]
    val_macro = [h.val_macro_f1 for h in history]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(epochs, train_loss, color="tab:blue", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_auc, color="tab:green", label="val_auc")
    ax2.plot(epochs, val_macro, color="tab:orange", label="val_macro_f1")
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
    master_manifest = resolve(project_root, args.master_manifest)
    temporal_root = resolve(project_root, args.temporal_root)
    init_ckpt = resolve(project_root, args.init_ckpt)
    output_dir = resolve(project_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    tiou_thresholds = parse_float_list(args.localization_tiou)
    threshold_candidates = parse_float_list(args.threshold_candidates)

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

    model = RTFMTRNBoundary(
        input_dim=2048,
        hidden_dim=args.hidden_dim,
        num_classes=len(class_names),
        target_segments=args.target_segments,
        trn_layers=args.trn_layers,
        trn_heads=args.trn_heads,
        trn_ffn_mult=args.trn_ffn_mult,
        trn_dropout=args.trn_dropout,
        proj_dropout=args.dropout,
        pos_encoding=args.pos_encoding,
    ).to(device)

    init_info = load_checkpoint_weights(model, init_ckpt, device)

    train_labels = np.array([int(r["binary_label"]) for r in train_ds.rows], dtype=np.int64)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    if args.balanced_sampler:
        pos_weight = 1.0

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[EpochResult] = []
    best_key = (-float("inf"), -float("inf"))
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bce, train_rtfm, train_cls, train_bnd, train_smooth = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce_fn=bce_fn,
            device=device,
            class_to_idx=class_to_idx,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            pseudo_topk=args.pseudo_topk,
            rtfm_margin=args.rtfm_margin,
            rtfm_lambda=args.rtfm_lambda,
            cls_lambda=args.cls_lambda,
            bnd_lambda=args.bnd_lambda,
            smooth_lambda=args.smooth_lambda,
        )

        val_metrics = evaluate_sampled(
            model=model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            class_to_idx=class_to_idx,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            pseudo_topk=args.pseudo_topk,
            threshold=args.threshold,
        )

        val_auc = float(val_metrics["binary"]["auc"])
        val_ap = float(val_metrics["binary"]["ap"])
        val_macro = float(val_metrics["classification"]["macro_f1"])
        val_weighted = float(val_metrics["classification"]["weighted_f1"])

        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_bce_loss=train_bce,
                train_rtfm_loss=train_rtfm,
                train_cls_loss=train_cls,
                train_bnd_loss=train_bnd,
                train_smooth_loss=train_smooth,
                val_auc=val_auc,
                val_ap=val_ap,
                val_macro_f1=val_macro,
                val_weighted_f1=val_weighted,
            )
        )

        if args.checkpoint_metric == "val_macro_f1":
            primary = metric_key(val_macro)
        elif args.checkpoint_metric == "val_weighted_f1":
            primary = metric_key(val_weighted)
        else:
            primary = metric_key(val_auc)

        cur_key = (primary, metric_key(val_auc))

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"(bce={train_bce:.5f}, rtfm={train_rtfm:.5f}, cls={train_cls:.5f}, bnd={train_bnd:.5f}, smooth={train_smooth:.5f}) | "
            f"val_auc={val_auc:.5f} val_ap={val_ap:.5f} "
            f"val_macro_f1={val_macro:.5f} val_weighted_f1={val_weighted:.5f}"
        )

        if cur_key > best_key:
            best_key = cur_key
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": json_safe_args(args),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "val_macro_f1": val_macro,
                    "val_weighted_f1": val_weighted,
                },
                ckpt_dir / "best.pt",
            )

    save_history(history, output_dir / "train_history.csv")
    maybe_plot_curves(history, output_dir / "train_curves.png")

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    feat_rows = [r for r in read_csv(feature_manifest) if r.get("status") == "ok"]
    val_rows = [r for r in feat_rows if r.get("split") == "val"]
    test_rows = [r for r in feat_rows if r.get("split") == "test"]

    master_rows = read_csv(master_manifest)
    master_by_video = {r["video_id"]: r for r in master_rows}

    print("\nCalibrating anomaly threshold on val split (full-sequence inference)...")
    val_full_results: List[Dict[str, object]] = []
    for i, r in enumerate(val_rows, 1):
        vr = infer_video_full(
            model=model,
            row=r,
            project_root=project_root,
            class_names=class_names,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            window=args.infer_window,
            stride=args.infer_stride,
            smooth_window=args.smooth_window,
            threshold=args.threshold,
            min_event_len=args.min_event_len,
            merge_gap=args.merge_gap,
            boundary_radius=args.boundary_radius,
            device=device,
        )
        val_full_results.append(vr)
        if i % 20 == 0 or i == len(val_rows):
            print(f"  val full inference {i}/{len(val_rows)}")

    threshold_info = tune_threshold_on_val(val_full_results, threshold_candidates)
    tuned_threshold = float(threshold_info["best_threshold"])
    print(f"Chosen threshold from val: {tuned_threshold:.4f} (best binary F1={threshold_info['best_f1']:.4f})")

    print("\nRunning test full-sequence inference with boundary refinement...")
    test_full_results: List[Dict[str, object]] = []
    for i, r in enumerate(test_rows, 1):
        tr = infer_video_full(
            model=model,
            row=r,
            project_root=project_root,
            class_names=class_names,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            window=args.infer_window,
            stride=args.infer_stride,
            smooth_window=args.smooth_window,
            threshold=tuned_threshold,
            min_event_len=args.min_event_len,
            merge_gap=args.merge_gap,
            boundary_radius=args.boundary_radius,
            device=device,
        )
        test_full_results.append(tr)
        if i % 25 == 0 or i == len(test_rows):
            print(f"  test full inference {i}/{len(test_rows)}")

    # Build GT events for localization
    gt_events = load_gt_events_for_test(
        test_rows=test_rows,
        master_rows_by_video=master_by_video,
        temporal_root=temporal_root,
    )
    gt_by_video: Dict[str, List[Dict[str, object]]] = {}
    for g in gt_events:
        gt_by_video.setdefault(str(g["video_id"]), []).append(g)

    # Build predicted events (after boundary refinement)
    pred_events_all: List[Dict[str, object]] = []
    for r in test_full_results:
        for ev in r.get("events_after_refine", []):
            pred_events_all.append(
                {
                    "video_id": r["video_id"],
                    "predicted_class": ev["predicted_class"],
                    "start_segment": int(ev["start_segment"]),
                    "end_segment": int(ev["end_segment"]),
                    "event_score": float(ev["event_score"]),
                }
            )

    localization = evaluate_localization_map(
        pred_events_all=pred_events_all,
        gt_events_all=gt_events,
        class_names=class_names,
        normal_idx=normal_idx,
        tiou_thresholds=tiou_thresholds,
    )

    # Updated video-level metrics from full-sequence inference
    y_true_bin = np.array([int(r["binary_label"]) for r in test_full_results], dtype=np.int64)
    y_score_bin = np.array([float(r["video_anomaly_score"]) for r in test_full_results], dtype=np.float64)
    y_pred_bin = (y_score_bin >= tuned_threshold).astype(np.int64)

    binary_auc = safe_auc(y_true_bin, y_score_bin)
    binary_ap = safe_ap(y_true_bin, y_score_bin)
    binary_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist() if y_true_bin.size > 0 else [[0, 0], [0, 0]]

    y_true_cls = np.array([class_to_idx.get(str(r["category_label"]), normal_idx) for r in test_full_results], dtype=np.int64)
    y_pred_cls = np.array([class_to_idx.get(str(r["pred_video_class"]), normal_idx) for r in test_full_results], dtype=np.int64)

    labels = list(range(len(class_names)))
    cls_cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_cls,
        y_pred_cls,
        labels=labels,
        zero_division=0,
    )
    per_class = [
        {
            "class_name": class_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    ]
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_cls, y_pred_cls, labels=labels, average="weighted", zero_division=0))

    # Boundary qualitative examples
    anomaly_test = [r for r in test_full_results if int(r["binary_label"]) == 1]
    anomaly_test = sorted(anomaly_test, key=lambda x: float(x["video_anomaly_score"]), reverse=True)
    boundary_qualitative = []
    for r in anomaly_test[:3]:
        boundary_qualitative.append(
            {
                "video_id": r["video_id"],
                "ground_truth_class": r["category_label"],
                "predicted_event_spans_before_refinement": r.get("spans_before_refine", []),
                "predicted_event_spans_after_refinement": r.get("spans_after_refine", []),
                "top_boundary_peaks": r.get("top_boundary_peaks", []),
            }
        )

    failure_case = choose_failure_case(test_full_results, gt_by_video, tuned_threshold)
    success_case = choose_success_case(test_full_results, gt_by_video)

    step4_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_baseline/results_summary.json")))
    step5_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_classifier/results_summary.json")))
    step6_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_trn/results_summary.json")))

    comparison_table = [
        extract_model_metrics(step4_summary, "Step 4 baseline"),
        extract_model_metrics(step5_summary, "Step 5 + classifier"),
        extract_model_metrics(step6_summary, "Step 6 + TRN"),
        {
            "model": "Step 7 + Boundary",
            "auc": binary_auc,
            "ap": binary_ap,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
    ]

    results = {
        "architecture": {
            "pipeline": "features -> projection -> TRN -> anomaly/classifier/boundary heads",
            "projection_dim": args.hidden_dim,
            "positional_encoding": args.pos_encoding,
            "trn_layers": args.trn_layers,
            "trn_heads": args.trn_heads,
            "trn_ffn_dim": args.hidden_dim * args.trn_ffn_mult,
            "trn_dropout": args.trn_dropout,
            "boundary_head": {
                "input_dim": args.hidden_dim * 2 + 1,
                "input_definition": "concat(h_t, h_t+1, |s_t - s_t+1|)",
                "layers": [args.hidden_dim * 2 + 1, args.hidden_dim // 2, 1],
                "activation": "sigmoid",
                "prediction": "per edge (t, t+1)",
            },
            "temporal_sampling_rule_train": "uniform sample to 32; pad by repeating last segment if shorter",
            "full_sequence_inference": {
                "mode": "chunked sliding-window over full cached sequence",
                "window": args.infer_window,
                "stride": args.infer_stride,
                "note": "no random 32-segment sampling at inference; all segments are processed",
            },
        },
        "training_setup": {
            "checkpoint_initialization": str(init_ckpt),
            "initialization_details": init_info,
            "loss_formula": "L = L_anomaly + 0.5*L_cls + 0.3*L_bnd + 0.1*L_smooth",
            "losses": {
                "anomaly": f"BCE + {args.rtfm_lambda}*RTFM",
                "classification": f"{args.cls_lambda}*CrossEntropy",
                "boundary": f"{args.bnd_lambda}*MSE(|s_t-s_t+1|, b_t)",
                "smoothness": f"{args.smooth_lambda}*Smoothness(refined anomaly scores)",
            },
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "checkpoint_rule": f"best {args.checkpoint_metric}, tie-break by val_auc",
            "device": device,
            "seed": args.seed,
            "balanced_sampler": bool(args.balanced_sampler),
            "pos_weight": pos_weight,
            "threshold_selection": threshold_info,
            "test_threshold": tuned_threshold,
        },
        "training_curves": [
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss,
                "train_bce_loss": h.train_bce_loss,
                "train_rtfm_loss": h.train_rtfm_loss,
                "train_cls_loss": h.train_cls_loss,
                "train_bnd_loss": h.train_bnd_loss,
                "train_smooth_loss": h.train_smooth_loss,
                "val_auc": h.val_auc,
                "val_ap": h.val_ap,
                "val_macro_f1": h.val_macro_f1,
                "val_weighted_f1": h.val_weighted_f1,
            }
            for h in history
        ],
        "best_epoch": best_epoch,
        "val_best": {
            "auc": best_ckpt.get("val_auc"),
            "ap": best_ckpt.get("val_ap"),
            "macro_f1": best_ckpt.get("val_macro_f1"),
            "weighted_f1": best_ckpt.get("val_weighted_f1"),
        },
        "temporal_localization": localization,
        "updated_metrics": {
            "binary": {
                "auc": binary_auc,
                "ap": binary_ap,
                "confusion_matrix_threshold": tuned_threshold,
                "confusion_matrix": binary_cm,
            },
            "classification": {
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "confusion_matrix": cls_cm,
                "class_names": class_names,
                "per_class": per_class,
            },
        },
        "boundary_qualitative_examples": boundary_qualitative,
        "failure_case": failure_case,
        "success_case": success_case,
        "comparison_table": comparison_table,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_video_results.json").write_text(json.dumps(test_full_results, indent=2) + "\n")
    (output_dir / "pred_events_test.json").write_text(json.dumps(pred_events_all, indent=2) + "\n")
    (output_dir / "gt_events_test.json").write_text(json.dumps(gt_events, indent=2) + "\n")

    print("\nFinal Step-7 metrics")
    print(f"- Binary AUC: {binary_auc:.6f}")
    print(f"- Binary AP:  {binary_ap:.6f}")
    print(f"- Macro-F1:   {macro_f1:.6f}")
    print(f"- Weighted-F1:{weighted_f1:.6f}")
    for thr in tiou_thresholds:
        k = str(thr)
        print(f"- mAP@{thr}: {results['temporal_localization']['tiou'][k]['mAP']:.6f}")
    print(f"- GT events: {results['temporal_localization']['gt_event_count']}")
    print(f"- Pred events: {results['temporal_localization']['pred_event_count']}")
    print(f"- Saved: {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
