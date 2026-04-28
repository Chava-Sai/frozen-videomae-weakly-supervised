#!/usr/bin/env python3
"""Train RTFM + Classifier + TRN + Boundary head on VideoMAE-B features (D=768).

Improvements over Step-7 (train_rtfm_trn_boundary.py):
  1. Feature dim: 768  (VideoMAE-B)  instead of 2048 (I3D)
  2. Class-weighted cross-entropy — down-weights the dominant 'normal' class
  3. Confidence-thresholded pseudo-labels — only assign category label when
     the top anomaly score exceeds --pseudo-conf-threshold (default 0.3)
  4. Cosine-annealing LR schedule (eta_min=1e-6)
  5. Multi-seed: pass --seed 42 / 123 / 456 and use --output-dir with seed tag

Usage — single seed:
    python src/train_videomae_full.py --seed 42

Usage — three seeds (run in separate terminals or sequentially):
    python src/train_videomae_full.py --seed 42  --output-dir outputs/videomae_rtfm/seed_42
    python src/train_videomae_full.py --seed 123 --output-dir outputs/videomae_rtfm/seed_123
    python src/train_videomae_full.py --seed 456 --output-dir outputs/videomae_rtfm/seed_456
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
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
    p = argparse.ArgumentParser(description="VideoMAE-B RTFM+TRN+Boundary training")
    p.add_argument("--project-root", type=Path, default=Path.cwd())

    # ── data paths ────────────────────────────────────────────────────────────
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_videomae.csv"),
        help="VideoMAE feature manifest CSV",
    )
    p.add_argument(
        "--master-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_master.csv"),
    )
    p.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("data/ucf_crime/annotations/temporal_segments"),
    )
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from (must also use D=768). "
             "Leave empty to train from scratch (recommended for VideoMAE).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/videomae_rtfm/seed_42"),
    )

    # ── architecture ──────────────────────────────────────────────────────────
    p.add_argument(
        "--feature-dim",
        type=int,
        default=768,
        help="Input feature dimensionality (768 for VideoMAE-B, 2048 for I3D)",
    )
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--target-segments", type=int, default=32)
    p.add_argument("--trn-layers", type=int, default=2)
    p.add_argument("--trn-heads", type=int, default=4)
    p.add_argument("--trn-ffn-mult", type=int, default=4)
    p.add_argument("--trn-dropout", type=float, default=0.1)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6, help="Cosine annealing eta_min")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable class-weighted CE (use uniform CE instead)")

    # ── loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--topk-ratio", type=float, default=0.125)
    p.add_argument("--pseudo-topk", type=int, default=4)
    p.add_argument(
        "--pseudo-conf-threshold",
        type=float,
        default=0.30,
        help="Only assign category pseudo-label when top segment anomaly "
             "score exceeds this threshold. Reduces noisy pseudo-labels.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rtfm-margin", type=float, default=5.0)
    p.add_argument("--rtfm-lambda", type=float, default=0.1)
    p.add_argument("--cls-lambda", type=float, default=0.5)
    p.add_argument("--bnd-lambda", type=float, default=0.3)
    p.add_argument("--smooth-lambda", type=float, default=0.1)

    # ── inference / threshold sweep ───────────────────────────────────────────
    p.add_argument("--infer-window", type=int, default=32)
    p.add_argument("--infer-stride", type=int, default=16)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--min-event-len", type=int, default=1)
    p.add_argument("--merge-gap", type=int, default=0)
    p.add_argument("--boundary-radius", type=int, default=2)
    p.add_argument(
        "--threshold-candidates",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )
    p.add_argument("--localization-tiou", type=str, default="0.3,0.5,0.7")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--class-names",
        type=str,
        default="normal,fighting,shooting,explosion,robbery,abuse",
    )
    p.add_argument("--normal-class", type=str, default="normal")
    p.add_argument(
        "--checkpoint-metric",
        choices=["val_macro_f1", "val_weighted_f1", "val_auc"],
        default="val_macro_f1",
    )
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


# ── utils ────────────────────────────────────────────────────────────────────

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def parse_class_names(raw: str) -> List[str]:
    classes = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if len(classes) != len(set(classes)):
        raise ValueError(f"duplicate classes: {classes}")
    return classes


def parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def topk_mean(values: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, int]:
    t = values.shape[1]
    k = max(1, int(math.ceil(t * ratio)))
    return torch.topk(values, k=k, dim=1).values.mean(dim=1), k


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


def metric_key(m: float) -> float:
    return -1e9 if math.isnan(m) else m


# ── class-weight computation ─────────────────────────────────────────────────

def compute_class_weights(
    rows: List[Dict[str, str]],
    class_names: List[str],
    device: str,
) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy loss."""
    counts = Counter(str(r.get("category_label", "normal")).lower() for r in rows)
    total = sum(counts.values())
    n_cls = len(class_names)
    weights = []
    for c in class_names:
        cnt = max(counts.get(c, 1), 1)
        weights.append(total / (n_cls * cnt))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    # normalise so mean weight = 1 (keeps loss scale stable)
    w = w / w.mean()
    return w


# ── model ────────────────────────────────────────────────────────────────────

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    pos = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TRNEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ffn_dim, dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.self_attn(x, x, x, need_weights=return_attention, average_attn_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x, attn_w if return_attention else None


class RTFMTRNBoundary(nn.Module):
    """Feature projection → TRN → anomaly / class / boundary heads.

    input_dim accepts any value; set to 768 for VideoMAE-B.
    """

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
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=proj_dropout)
        )

        if pos_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, target_segments, hidden_dim))
            nn.init.normal_(self.pos_embedding, std=0.02)
            self.register_buffer("sinusoidal_pe", torch.empty(0), persistent=False)
        else:
            pe = sinusoidal_positional_encoding(target_segments, hidden_dim)
            self.register_buffer("sinusoidal_pe", pe.unsqueeze(0), persistent=False)
            self.pos_embedding = None

        self.trn_layers = nn.ModuleList(
            [TRNEncoderLayer(hidden_dim, trn_heads, hidden_dim * trn_ffn_mult, trn_dropout) for _ in range(trn_layers)]
        )
        self.segment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout), nn.Linear(hidden_dim // 2, num_classes),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(p=proj_dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if self.pos_encoding_type == "learned":
            if t <= self.target_segments:
                return x + self.pos_embedding[:, :t, :]
            pe = F.interpolate(self.pos_embedding.transpose(1, 2), size=t, mode="linear", align_corners=False).transpose(1, 2)
            return x + pe
        return x + self.sinusoidal_pe[:, :t, :].to(dtype=x.dtype, device=x.device)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z = self.feature_proj(x)
        z = self.add_positional_encoding(z)
        attn_maps: List[torch.Tensor] = []
        for layer in self.trn_layers:
            z, aw = layer(z, return_attention=return_attention)
            if return_attention and aw is not None:
                attn_maps.append(aw)
        seg_anom = self.segment_head(z).squeeze(-1)
        seg_cls = self.class_head(z)
        feat_mags = torch.norm(z, p=2, dim=-1)
        if z.shape[1] > 1:
            s_prob = torch.sigmoid(seg_anom)
            diff = torch.abs(s_prob[:, 1:] - s_prob[:, :-1]).unsqueeze(-1)
            bnd = self.boundary_head(torch.cat([z[:, :-1], z[:, 1:], diff], dim=-1)).squeeze(-1)
        else:
            bnd = torch.zeros((z.shape[0], 0), device=z.device, dtype=z.dtype)
        return seg_anom, seg_cls, feat_mags, bnd, z, attn_maps


# ── losses ───────────────────────────────────────────────────────────────────

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


# ── data loader ───────────────────────────────────────────────────────────────

def build_loader(ds: FeatureSequenceDataset, batch_size: int, num_workers: int, balanced: bool) -> DataLoader:
    if balanced:
        labels = np.array([int(r["binary_label"]) for r in ds.rows], dtype=np.int64)
        counts = np.maximum(np.bincount(labels, minlength=2), 1)
        weights = torch.from_numpy(1.0 / counts[labels]).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=fixed_segments_collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=fixed_segments_collate)


# ── training epoch ────────────────────────────────────────────────────────────

@dataclass
class EpochResult:
    epoch: int
    lr: float
    train_loss: float
    train_bce: float
    train_rtfm: float
    train_cls: float
    train_bnd: float
    train_smooth: float
    val_auc: float
    val_ap: float
    val_macro_f1: float
    val_weighted_f1: float


def train_one_epoch(
    model: RTFMTRNBoundary,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_fn: nn.Module,
    cls_weight: Optional[torch.Tensor],
    device: str,
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    pseudo_topk: int,
    pseudo_conf_threshold: float,
    rtfm_margin: float,
    rtfm_lambda: float,
    cls_lambda: float,
    bnd_lambda: float,
    smooth_lambda: float,
) -> Tuple[float, float, float, float, float, float]:
    model.train()
    totals = [0.0] * 6
    n = 0

    for batch in loader:
        x = batch["features"].to(device)
        y_bin = batch["binary_labels"].float().to(device)

        seg_anom, seg_cls, feat_mags, bnd, _, _ = model(x)

        vid_logits, _ = topk_mean(seg_anom, topk_ratio)
        bce_loss = bce_fn(vid_logits, y_bin)

        # RTFM magnitude separation
        rtfm_loss = torch.tensor(0.0, device=device)
        anom_m = y_bin > 0.5
        norm_m = ~anom_m
        if anom_m.any() and norm_m.any():
            a_mag, _ = topk_mean(feat_mags[anom_m], topk_ratio)
            n_mag, _ = topk_mean(feat_mags[norm_m], topk_ratio)
            rtfm_loss = torch.relu(rtfm_margin - (a_mag.mean() - n_mag.mean()))

        # Class CE with confidence-thresholded pseudo-labels
        bsz, tlen, _ = seg_cls.shape
        cls_losses: List[torch.Tensor] = []
        for i in range(bsz):
            cat = str(batch["category_labels"][i]).lower()
            if cat not in class_to_idx:
                continue
            if int(y_bin[i].item()) == 1 and class_to_idx[cat] != normal_idx:
                # ── confidence threshold: only assign label if confident ──────
                max_score = float(torch.sigmoid(seg_anom[i]).max().item())
                if max_score < pseudo_conf_threshold:
                    continue  # skip noisy pseudo-label
                k = min(pseudo_topk, tlen)
                top_idx = torch.topk(seg_anom[i], k=k).indices
                tgt = torch.full((k,), class_to_idx[cat], device=device, dtype=torch.long)
                cls_losses.append(
                    F.cross_entropy(seg_cls[i, top_idx], tgt, weight=cls_weight)
                )
            else:
                tgt = torch.full((tlen,), normal_idx, device=device, dtype=torch.long)
                cls_losses.append(F.cross_entropy(seg_cls[i], tgt, weight=cls_weight))

        cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
        bnd_loss = boundary_loss(seg_anom, bnd)
        s_loss = smoothness_loss(seg_anom)

        loss = bce_loss + rtfm_lambda * rtfm_loss + cls_lambda * cls_loss + bnd_lambda * bnd_loss + smooth_lambda * s_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals[0] += loss.item()
        totals[1] += bce_loss.item()
        totals[2] += rtfm_loss.item()
        totals[3] += cls_loss.item()
        totals[4] += bnd_loss.item()
        totals[5] += s_loss.item()
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return tuple(t / n for t in totals)  # type: ignore[return-value]


# ── evaluation ────────────────────────────────────────────────────────────────

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
    y_true_bin, y_score_bin, y_true_cls, y_pred_cls = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_bin = batch["binary_labels"].to(device)
            seg_anom, seg_cls, _, _, _, _ = model(x)
            vid_probs = torch.sigmoid(topk_mean(seg_anom, topk_ratio)[0])
            bsz, tlen, _ = seg_cls.shape
            for i in range(bsz):
                gt_bin = int(y_bin[i].item())
                gt_cls = class_to_idx.get(str(batch["category_labels"][i]).lower(), normal_idx)
                anom_p = float(vid_probs[i].item())
                y_true_bin.append(gt_bin)
                y_score_bin.append(anom_p)
                k = min(pseudo_topk, tlen)
                top_idx = torch.topk(seg_anom[i], k=k).indices
                cls_probs = torch.softmax(seg_cls[i, top_idx], dim=-1).mean(dim=0)
                raw = int(torch.argmax(cls_probs).item())
                y_true_cls.append(gt_cls)
                y_pred_cls.append(normal_idx if anom_p < threshold else raw)

    y_tb = np.array(y_true_bin, dtype=np.int64)
    y_sb = np.array(y_score_bin, dtype=np.float64)
    labels = list(range(len(class_names)))
    y_tc = np.array(y_true_cls, dtype=np.int64)
    y_pc = np.array(y_pred_cls, dtype=np.int64)
    return {
        "binary": {"auc": safe_auc(y_tb, y_sb), "ap": safe_ap(y_tb, y_sb)},
        "classification": {
            "macro_f1": float(f1_score(y_tc, y_pc, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_tc, y_pc, labels=labels, average="weighted", zero_division=0)),
        },
    }


# ── full-sequence inference ───────────────────────────────────────────────────

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
        return np.zeros((0,), np.float32), np.zeros((0, num_classes), np.float32), np.zeros((0,), np.float32)

    sum_a = np.zeros(t, np.float64)
    cnt_a = np.zeros(t, np.float64)
    sum_c = np.zeros((t, num_classes), np.float64)
    cnt_c = np.zeros(t, np.float64)
    sum_b = np.zeros(max(t - 1, 0), np.float64)
    cnt_b = np.zeros(max(t - 1, 0), np.float64)

    model.eval()
    with torch.no_grad():
        for s in sliding_window_starts(t, window, stride):
            e = min(s + window, t)
            real = e - s
            chunk = features[s:e]
            if real < window:
                pad = np.repeat(chunk[-1:], window - real, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            sa, sc, _, bnd, _, _ = model(x)
            seg_s = torch.sigmoid(sa)[0].cpu().numpy()[:real]
            cls_p = torch.softmax(sc, -1)[0].cpu().numpy()[:real]
            sum_a[s:e] += seg_s; cnt_a[s:e] += 1
            sum_c[s:e] += cls_p; cnt_c[s:e] += 1
            if real >= 2 and bnd.shape[1] > 0:
                bv = bnd[0].cpu().numpy()[: real - 1]
                sum_b[s: e - 1] += bv; cnt_b[s: e - 1] += 1

    div = lambda a, c: np.divide(a, np.maximum(c, 1e-8))
    return div(sum_a, cnt_a).astype(np.float32), div(sum_c, cnt_c[:, None]).astype(np.float32), div(sum_b, cnt_b).astype(np.float32)


def moving_average(scores: np.ndarray, w: int) -> np.ndarray:
    if scores.size == 0 or w <= 1:
        return scores.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid").astype(np.float32)


def spans_from_scores(scores: np.ndarray, thr: float, min_len: int, merge_gap: int) -> List[Tuple[int, int]]:
    if scores.size == 0:
        return []
    mask = scores >= thr
    spans, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1; continue
        s = i
        while i + 1 < len(mask) and mask[i + 1]:
            i += 1
        spans.append((s, i)); i += 1
    if merge_gap > 0 and spans:
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s - merged[-1][1] - 1 <= merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        spans = merged
    return [(s, e) for s, e in spans if e - s + 1 >= min_len]


def refine_spans_with_boundary(
    spans: Sequence[Tuple[int, int]], bnd: np.ndarray, total: int, radius: int, min_len: int, merge_gap: int
) -> List[Tuple[int, int]]:
    if not spans:
        return []
    if bnd.size == 0:
        return list(spans)
    max_e = int(bnd.shape[0] - 1)
    refined = []
    for s, e in spans:
        sc = max(0, s - 1)
        sl, sr = max(0, sc - radius), min(max_e, sc + radius)
        se = sl + int(np.argmax(bnd[sl: sr + 1])) + 1
        ec = min(max_e, e)
        el, er = max(0, ec - radius), min(max_e, ec + radius)
        ee = el + int(np.argmax(bnd[el: er + 1]))
        se = max(0, min(se, total - 1)); ee = max(0, min(ee, total - 1))
        if se > ee:
            se, ee = s, e
        refined.append((se, ee))
    refined.sort(key=lambda x: x[0])
    if merge_gap > 0 and refined:
        merged = [refined[0]]
        for s, e in refined[1:]:
            if s - merged[-1][1] - 1 <= merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        refined = merged
    return [(s, e) for s, e in refined if e - s + 1 >= min_len]


def spans_to_events(spans, scores, cls_probs, starts, ends, fps, class_names, normal_idx):
    out = []
    anom_idx = [i for i in range(len(class_names)) if i != normal_idx]
    for s, e in spans:
        cm = cls_probs[s: e + 1].mean(0)
        if anom_idx:
            pi = anom_idx[int(np.argmax(cm[anom_idx]))]
        else:
            pi = int(np.argmax(cm))
        out.append({
            "start_segment": int(s), "end_segment": int(e),
            "start_time": float(starts[s] / fps) if fps > 0 else 0.0,
            "end_time": float(ends[e] / fps) if fps > 0 else 0.0,
            "predicted_class": class_names[pi], "predicted_class_idx": pi,
            "event_score": float(scores[s: e + 1].mean() * cm[pi]),
            "class_confidence": float(cm[pi]),
            "anomaly_confidence": float(scores[s: e + 1].mean()),
        })
    return sorted(out, key=lambda x: x["event_score"], reverse=True)


def infer_video_full(model, row, project_root, class_names, normal_idx, topk_ratio,
                     window, stride, smooth_window, threshold, min_event_len, merge_gap,
                     boundary_radius, device):
    feat_path = resolve(project_root, Path(row["feature_path"]))
    with np.load(feat_path, allow_pickle=True) as d:
        feats = d["features"].astype(np.float32)
        starts = d["segment_start_frames"].astype(np.int64)
        ends = d["segment_end_frames"].astype(np.int64)

    sa, cp, bnd = infer_full_sequence_chunked(model, feats, device, len(class_names), window, stride)
    sas = moving_average(sa, smooth_window)
    t = int(sas.shape[0])
    spans_b = spans_from_scores(sas, threshold, min_event_len, merge_gap)
    spans_a = refine_spans_with_boundary(spans_b, bnd, t, boundary_radius, min_event_len, merge_gap)
    fps = float(row.get("fps", 25.0))
    ev_b = spans_to_events(spans_b, sas, cp, starts, ends, fps, class_names, normal_idx)
    ev_a = spans_to_events(spans_a, sas, cp, starts, ends, fps, class_names, normal_idx)
    vid_score = topk_mean_np(sas, topk_ratio)
    if vid_score < threshold or not ev_a:
        pred_cls, pred_idx, pred_p = class_names[normal_idx], normal_idx, 1.0 - vid_score
    else:
        best = ev_a[0]
        pred_cls, pred_idx, pred_p = str(best["predicted_class"]), int(best["predicted_class_idx"]), float(best["class_confidence"])
    return {
        "video_id": row["video_id"], "split": row["split"],
        "binary_label": int(row["binary_label"]), "category_label": str(row["category_label"]),
        "num_segments": int(feats.shape[0]), "fps": fps, "duration_sec": float(row.get("duration_sec", -1)),
        "video_anomaly_score": float(vid_score),
        "pred_video_class": pred_cls, "pred_video_class_idx": pred_idx, "pred_video_class_prob": float(pred_p),
        "spans_before_refine": [[int(s), int(e)] for s, e in spans_b],
        "spans_after_refine": [[int(s), int(e)] for s, e in spans_a],
        "events_before_refine": ev_b, "events_after_refine": ev_a,
        "segment_scores_smooth": sas.tolist(), "boundary_scores": bnd.tolist(),
    }


# ── threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(results, candidates):
    if not results:
        return {"best_threshold": 0.5, "best_f1": float("nan"), "rows": []}
    y_t = np.array([int(r["binary_label"]) for r in results], dtype=np.int64)
    y_s = np.array([float(r["video_anomaly_score"]) for r in results], dtype=np.float64)
    rows, best_thr, best_f1 = [], candidates[0], -1.0
    for thr in candidates:
        f = float(f1_score(y_t, (y_s >= thr).astype(np.int64), zero_division=0))
        rows.append({"threshold": float(thr), "f1": f})
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    return {"best_threshold": best_thr, "best_f1": best_f1, "rows": rows}


# ── GT event loading (for localization eval) ──────────────────────────────────

def load_gt_events(test_rows, master_by_video, temporal_root):
    gt = []
    for r in test_rows:
        if int(r["binary_label"]) == 0:
            continue
        vid = str(r["video_id"])
        m = master_by_video.get(vid, {})
        ann = m.get("temporal_annotation_path", "")
        ap = Path(ann) if ann else temporal_root / f"{vid}.json"
        if not ap.is_absolute():
            ap = temporal_root.parent.parent / ap
        if not ap.exists():
            ap = temporal_root / f"{vid}.json"
        if not ap.exists():
            continue
        d = json.loads(ap.read_text())
        ns = int(float(r.get("num_segments", 0)))
        for seg in (d.get("segments", []) if isinstance(d, dict) else []):
            sf, ef = int(seg.get("start_frame", 0)), int(seg.get("end_frame", 0))
            s, e = max(0, sf // SEGMENT_LEN), max(0, ef // SEGMENT_LEN)
            if ns > 0:
                s, e = min(s, ns - 1), min(e, ns - 1)
            gt.append({"video_id": vid, "class_label": str(r["category_label"]),
                       "start_segment": s, "end_segment": e})
    return gt


# ── localization mAP ─────────────────────────────────────────────────────────

def temporal_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter <= 0:
        return 0.0
    return inter / (a[1] - a[0] + 1 + b[1] - b[0] + 1 - inter)


def compute_ap(rec, prec):
    mr = np.concatenate(([0.], rec, [1.]))
    mp = np.concatenate(([0.], prec, [0.]))
    for i in range(len(mp) - 1, 0, -1):
        mp[i - 1] = max(mp[i - 1], mp[i])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return float(np.sum((mr[idx + 1] - mr[idx]) * mp[idx + 1]))


def detection_ap(preds, gt_by_video, tiou):
    total_gt = sum(len(v) for v in gt_by_video.values())
    if total_gt <= 0:
        return float("nan")
    preds = sorted(preds, key=lambda x: x["event_score"], reverse=True)
    matched = {v: np.zeros(len(g), bool) for v, g in gt_by_video.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, p in enumerate(preds):
        vid = str(p["video_id"])
        gts = gt_by_video.get(vid, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[vid][j]:
                continue
            iou = temporal_iou((p["start_segment"], p["end_segment"]), (g["start_segment"], g["end_segment"]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= tiou:
            tp[i] = 1; matched[vid][best_j] = True
        else:
            fp[i] = 1
    tc = np.cumsum(tp); fc = np.cumsum(fp)
    return compute_ap(tc / total_gt, tc / np.maximum(tc + fc, 1e-12))


def evaluate_localization(preds, gts, class_names, normal_idx, thresholds):
    anom_cls = [c for i, c in enumerate(class_names) if i != normal_idx]
    gt_cv = {c: {} for c in anom_cls}
    for g in gts:
        c = str(g["class_label"])
        gt_cv.setdefault(c, {}).setdefault(str(g["video_id"]), []).append(g)
    pred_c = {c: [] for c in anom_cls}
    for p in preds:
        c = str(p["predicted_class"])
        if c in pred_c:
            pred_c[c].append(p)
    out = {"tiou": {}, "gt_event_count": len(gts), "pred_event_count": len(preds)}
    for thr in thresholds:
        aps = []
        pc = {}
        for c in anom_cls:
            ap = detection_ap(pred_c[c], gt_cv.get(c, {}), thr)
            pc[c] = ap
            if not math.isnan(ap):
                aps.append(ap)
        out["tiou"][str(thr)] = {"mAP": float(np.mean(aps)) if aps else float("nan"), "per_class_ap": pc}
    return out


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint_weights(model, ckpt_path, device):
    if ckpt_path is None or not ckpt_path.exists():
        return {"loaded": False}
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    ms = model.state_dict()
    up = dict(ms)
    copied = []
    for k, v in state.items():
        if k in ms and ms[k].shape == v.shape:
            up[k] = v; copied.append(k)
    model.load_state_dict(up)
    return {"loaded": True, "num_copied": len(copied)}


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "train_bce", "train_rtfm", "train_cls",
                    "train_bnd", "train_smooth", "val_auc", "val_ap", "val_macro_f1", "val_weighted_f1"])
        for h in history:
            w.writerow([h.epoch, h.lr, h.train_loss, h.train_bce, h.train_rtfm,
                        h.train_cls, h.train_bnd, h.train_smooth,
                        h.val_auc, h.val_ap, h.val_macro_f1, h.val_weighted_f1])


def maybe_plot(history, path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h.epoch for h in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, [h.train_loss for h in history], "tab:blue", label="train_loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h.val_auc for h in history], "tab:green", label="val_auc")
    ax2.plot(epochs, [h.val_macro_f1 for h in history], "tab:orange", label="val_macro_f1")
    ax2.set_ylabel("val metrics", color="tab:green")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    feat_manifest = resolve(root, args.feature_manifest)
    master_manifest = resolve(root, args.master_manifest)
    temporal_root = resolve(root, args.temporal_root)
    output_dir = resolve(root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    normal_idx = class_to_idx[args.normal_class]
    tiou_thresholds = parse_float_list(args.localization_tiou)
    thr_candidates = parse_float_list(args.threshold_candidates)

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device} | Seed: {args.seed} | Feature dim: {args.feature_dim}")

    # ── datasets ────────────────────────────────────────────────────────────
    train_ds = FeatureSequenceDataset(feat_manifest, root, "train", args.target_segments, require_ok_status=True)
    val_ds = FeatureSequenceDataset(feat_manifest, root, "val", args.target_segments, require_ok_status=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, args.balanced_sampler)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=fixed_segments_collate)

    # ── class weights ────────────────────────────────────────────────────────
    if args.no_class_weights:
        cls_weight = None
        print("Class weights: disabled (uniform CE)")
    else:
        cls_weight = compute_class_weights(train_ds.rows, class_names, device)
        print("Class weights:", {c: f"{cls_weight[i].item():.3f}" for i, c in enumerate(class_names)})

    # ── model ────────────────────────────────────────────────────────────────
    model = RTFMTRNBoundary(
        input_dim=args.feature_dim,          # 768 for VideoMAE-B
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

    init_info = load_checkpoint_weights(model, args.init_ckpt, device) if args.init_ckpt else {"loaded": False}
    print(f"Checkpoint init: {init_info}")

    # ── pos-weight BCE ───────────────────────────────────────────────────────
    train_labels = np.array([int(r["binary_label"]) for r in train_ds.rows])
    n_pos, n_neg = int((train_labels == 1).sum()), int((train_labels == 0).sum())
    pos_weight = 1.0 if args.balanced_sampler else float(n_neg / max(n_pos, 1))
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    # ── optimizer + cosine LR ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # ── training loop ─────────────────────────────────────────────────────────
    history: List[EpochResult] = []
    best_key = (-float("inf"), -float("inf"))
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        tl, tb, tr, tc, tbn, ts = train_one_epoch(
            model, train_loader, optimizer, bce_fn, cls_weight, device,
            class_to_idx, normal_idx, args.topk_ratio, args.pseudo_topk,
            args.pseudo_conf_threshold, args.rtfm_margin, args.rtfm_lambda,
            args.cls_lambda, args.bnd_lambda, args.smooth_lambda,
        )
        scheduler.step()

        vm = evaluate_sampled(model, val_loader, device, class_names, class_to_idx,
                               normal_idx, args.topk_ratio, args.pseudo_topk, args.threshold)
        vauc = float(vm["binary"]["auc"])
        vap = float(vm["binary"]["ap"])
        vmacro = float(vm["classification"]["macro_f1"])
        vwt = float(vm["classification"]["weighted_f1"])

        history.append(EpochResult(epoch, current_lr, tl, tb, tr, tc, tbn, ts, vauc, vap, vmacro, vwt))

        primary = metric_key(vmacro if args.checkpoint_metric == "val_macro_f1"
                             else (vwt if args.checkpoint_metric == "val_weighted_f1" else vauc))
        cur_key = (primary, metric_key(vauc))

        print(f"Ep {epoch:03d} lr={current_lr:.2e} | loss={tl:.5f} "
              f"(bce={tb:.4f} rtfm={tr:.4f} cls={tc:.4f} bnd={tbn:.4f} sm={ts:.4f}) | "
              f"val_auc={vauc:.5f} val_ap={vap:.5f} val_mF1={vmacro:.5f} val_wF1={vwt:.5f}")

        if cur_key > best_key:
            best_key = cur_key; best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "args": json_safe_args(args),
                        "val_auc": vauc, "val_ap": vap, "val_macro_f1": vmacro, "val_weighted_f1": vwt},
                       ckpt_dir / "best.pt")

    save_history(history, output_dir / "train_history.csv")
    maybe_plot(history, output_dir / "train_curves.png")
    print(f"\nBest epoch: {best_epoch}")

    # ── load best + full inference ───────────────────────────────────────────
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    feat_rows = [r for r in read_csv(feat_manifest) if r.get("status") == "ok"]
    val_rows = [r for r in feat_rows if r.get("split") == "val"]
    test_rows = [r for r in feat_rows if r.get("split") == "test"]
    master_by_vid = {r["video_id"]: r for r in read_csv(master_manifest)}

    print(f"\nCalibrating threshold on {len(val_rows)} val videos...")
    val_res = []
    for i, r in enumerate(val_rows, 1):
        val_res.append(infer_video_full(model, r, root, class_names, normal_idx, args.topk_ratio,
                                        args.infer_window, args.infer_stride, args.smooth_window,
                                        args.threshold, args.min_event_len, args.merge_gap,
                                        args.boundary_radius, device))
        if i % 20 == 0 or i == len(val_rows):
            print(f"  val {i}/{len(val_rows)}")

    thr_info = tune_threshold(val_res, thr_candidates)
    tuned_thr = float(thr_info["best_threshold"])
    print(f"Tuned threshold: {tuned_thr:.4f}  (F1={thr_info['best_f1']:.4f})")

    print(f"\nTest inference on {len(test_rows)} videos...")
    test_res = []
    for i, r in enumerate(test_rows, 1):
        test_res.append(infer_video_full(model, r, root, class_names, normal_idx, args.topk_ratio,
                                          args.infer_window, args.infer_stride, args.smooth_window,
                                          tuned_thr, args.min_event_len, args.merge_gap,
                                          args.boundary_radius, device))
        if i % 25 == 0 or i == len(test_rows):
            print(f"  test {i}/{len(test_rows)}")

    gt_events = load_gt_events(test_rows, master_by_vid, temporal_root)
    gt_by_vid: Dict[str, List] = {}
    for g in gt_events:
        gt_by_vid.setdefault(str(g["video_id"]), []).append(g)

    pred_events: List[Dict] = []
    for r in test_res:
        for ev in r.get("events_after_refine", []):
            pred_events.append({"video_id": r["video_id"],
                                 "predicted_class": ev["predicted_class"],
                                 "start_segment": int(ev["start_segment"]),
                                 "end_segment": int(ev["end_segment"]),
                                 "event_score": float(ev["event_score"])})

    localization = evaluate_localization(pred_events, gt_events, class_names, normal_idx, tiou_thresholds)

    y_tb = np.array([int(r["binary_label"]) for r in test_res], np.int64)
    y_sb = np.array([float(r["video_anomaly_score"]) for r in test_res], np.float64)
    y_pb = (y_sb >= tuned_thr).astype(np.int64)

    binary_auc = safe_auc(y_tb, y_sb)
    binary_ap = safe_ap(y_tb, y_sb)

    y_tc = np.array([class_to_idx.get(str(r["category_label"]), normal_idx) for r in test_res], np.int64)
    y_pc = np.array([class_to_idx.get(str(r["pred_video_class"]), normal_idx) for r in test_res], np.int64)
    labels = list(range(len(class_names)))
    macro_f1 = float(f1_score(y_tc, y_pc, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_tc, y_pc, labels=labels, average="weighted", zero_division=0))
    prec, rec, f1s, sup = precision_recall_fscore_support(y_tc, y_pc, labels=labels, zero_division=0)
    per_class = [{"class": class_names[i], "precision": float(prec[i]), "recall": float(rec[i]),
                  "f1": float(f1s[i]), "support": int(sup[i])} for i in range(len(class_names))]

    results = {
        "feature_backbone": "videomae-b-kinetics400",
        "feature_dim": args.feature_dim,
        "seed": args.seed,
        "improvements_vs_step7": [
            "feature_dim=768 (VideoMAE-B instead of I3D)",
            "class_weighted_CE" if not args.no_class_weights else "uniform_CE",
            f"confidence_thresholded_pseudo_labels (thr={args.pseudo_conf_threshold})",
            "cosine_annealing_LR",
        ],
        "training_setup": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "lr_min": args.lr_min,
            "weight_decay": args.weight_decay, "seed": args.seed, "device": device,
            "pos_weight_bce": pos_weight, "class_weights_ce": {c: float(cls_weight[i].item()) for i, c in enumerate(class_names)} if cls_weight is not None else None,
            "threshold_info": thr_info, "test_threshold": tuned_thr,
        },
        "best_epoch": best_epoch,
        "val_best": {"auc": best_ckpt.get("val_auc"), "ap": best_ckpt.get("val_ap"),
                     "macro_f1": best_ckpt.get("val_macro_f1"), "weighted_f1": best_ckpt.get("val_weighted_f1")},
        "test": {
            "binary": {"auc": binary_auc, "ap": binary_ap,
                       "confusion_matrix": confusion_matrix(y_tb, y_pb, labels=[0, 1]).tolist()},
            "classification": {"macro_f1": macro_f1, "weighted_f1": weighted_f1,
                                "per_class": per_class,
                                "confusion_matrix": confusion_matrix(y_tc, y_pc, labels=labels).tolist()},
        },
        "temporal_localization": localization,
        "training_curves": [{"epoch": h.epoch, "lr": h.lr, "train_loss": h.train_loss,
                              "val_auc": h.val_auc, "val_macro_f1": h.val_macro_f1} for h in history],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_video_results.json").write_text(json.dumps(test_res, indent=2) + "\n")
    (output_dir / "pred_events_test.json").write_text(json.dumps(pred_events, indent=2) + "\n")
    (output_dir / "gt_events_test.json").write_text(json.dumps(gt_events, indent=2) + "\n")

    print("\n" + "=" * 60)
    print(f"VideoMAE RTFM Results  (seed={args.seed})")
    print("=" * 60)
    print(f"  Binary AUC   : {binary_auc:.6f}")
    print(f"  Binary AP    : {binary_ap:.6f}")
    print(f"  Macro-F1     : {macro_f1:.6f}")
    print(f"  Weighted-F1  : {weighted_f1:.6f}")
    for thr in tiou_thresholds:
        print(f"  mAP@{thr}    : {results['temporal_localization']['tiou'][str(thr)]['mAP']:.6f}")
    print(f"  Saved → {output_dir / 'results_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
