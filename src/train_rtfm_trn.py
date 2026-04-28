#!/usr/bin/env python3
"""Train/evaluate RTFM+Classifier+TRN (without boundary head) on cached I3D features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTFM + Classifier + TRN training")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    parser.add_argument(
        "--init-ckpt",
        type=Path,
        default=Path("outputs/rtfm_classifier/checkpoints/best.pt"),
        help="Step-5 checkpoint used to initialize shared modules.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rtfm_trn"),
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
    parser.add_argument("--smooth-lambda", type=float, default=0.1)

    parser.add_argument("--trn-layers", type=int, default=2)
    parser.add_argument("--trn-heads", type=int, default=4)
    parser.add_argument("--trn-ffn-mult", type=int, default=4)
    parser.add_argument("--trn-dropout", type=float, default=0.1)
    parser.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument(
        "--class-names",
        type=str,
        default="normal,fighting,shooting,explosion,robbery,abuse",
        help="Comma-separated taxonomy used for this run.",
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


class RTFMTRNClassifier(nn.Module):
    """Feature projection -> TRN refinement -> anomaly/classifier heads."""

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

    def add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if self.pos_encoding_type == "learned":
            return x + self.pos_embedding[:, :t, :]
        return x + self.sinusoidal_pe[:, :t, :].to(dtype=x.dtype, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z = self.feature_proj(x)  # [B, T, H]
        z = self.add_positional_encoding(z)

        attn_maps: List[torch.Tensor] = []
        for layer in self.trn_layers:
            z, attn_w = layer(z, return_attention=return_attention)
            if return_attention and attn_w is not None:
                attn_maps.append(attn_w)

        seg_anom_logits = self.segment_head(z).squeeze(-1)
        seg_class_logits = self.class_head(z)
        feat_magnitudes = torch.norm(z, p=2, dim=-1)
        return seg_anom_logits, seg_class_logits, feat_magnitudes, z, attn_maps


def smoothness_loss(seg_logits: torch.Tensor) -> torch.Tensor:
    if seg_logits.shape[1] < 2:
        return torch.tensor(0.0, device=seg_logits.device)
    s = torch.sigmoid(seg_logits)
    return torch.mean((s[:, 1:] - s[:, :-1]) ** 2)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_bce_loss: float
    train_rtfm_loss: float
    train_cls_loss: float
    train_smooth_loss: float
    val_auc: float
    val_ap: float
    val_macro_f1: float
    val_weighted_f1: float


def train_one_epoch(
    model: RTFMTRNClassifier,
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
    smooth_lambda: float,
) -> Tuple[float, float, float, float, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_rtfm = 0.0
    total_cls = 0.0
    total_smooth = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["features"].to(device)
        y_bin = batch["binary_labels"].float().to(device)

        seg_anom_logits, seg_class_logits, feat_mags, _, _ = model(x, return_attention=False)

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
        s_loss = smoothness_loss(seg_anom_logits)

        anomaly_loss = bce_loss + rtfm_lambda * rtfm_loss
        loss = anomaly_loss + cls_lambda * cls_loss + smooth_lambda * s_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_rtfm += float(rtfm_loss.item())
        total_cls += float(cls_loss.item())
        total_smooth += float(s_loss.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_bce / n_batches,
        total_rtfm / n_batches,
        total_cls / n_batches,
        total_smooth / n_batches,
    )


def evaluate(
    model: RTFMTRNClassifier,
    loader: DataLoader,
    device: str,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    pseudo_topk: int,
    threshold: float,
    collect_records: bool,
) -> Dict[str, object]:
    model.eval()

    y_true_bin: List[int] = []
    y_score_bin: List[float] = []

    y_true_cls: List[int] = []
    y_pred_cls: List[int] = []

    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_bin = batch["binary_labels"].to(device)

            seg_anom_logits, seg_class_logits, _, _, _ = model(x, return_attention=False)
            video_logits, _ = topk_mean(seg_anom_logits, topk_ratio)
            video_anom_probs = torch.sigmoid(video_logits)
            seg_anom_probs = torch.sigmoid(seg_anom_logits)

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
                pred_prob = float(video_cls_probs[pred_class].item())

                y_true_cls.append(gt_class)
                y_pred_cls.append(pred_class)

                if collect_records:
                    top5 = torch.topk(seg_anom_probs[i], k=min(5, tlen))
                    top5_idx = top5.indices.detach().cpu().numpy().astype(int).tolist()
                    top5_scores = top5.values.detach().cpu().numpy().astype(float).tolist()

                    records.append(
                        {
                            "video_id": batch["video_ids"][i],
                            "split": batch["splits"][i],
                            "binary_label": gt_binary,
                            "category_label": gt_label,
                            "gt_class_id": gt_class,
                            "pred_class_id": pred_class,
                            "pred_class_label": class_names[pred_class],
                            "pred_class_prob": pred_prob,
                            "anomaly_score": anom_prob,
                            "top_anomaly_segment_indices": top5_idx,
                            "top_anomaly_segment_scores": top5_scores,
                            "selected_pseudo_segment_indices": top_idx.detach().cpu().numpy().astype(int).tolist(),
                            "selected_pseudo_segment_scores": (
                                seg_anom_probs[i, top_idx].detach().cpu().numpy().astype(float).tolist()
                            ),
                            "refined_segment_scores": seg_anom_probs[i].detach().cpu().numpy().astype(float).tolist(),
                            "original_num_segments": int(batch["original_num_segments"][i].item()),
                        }
                    )

    y_true_bin_arr = np.array(y_true_bin, dtype=np.int64)
    y_score_bin_arr = np.array(y_score_bin, dtype=np.float64)

    binary_auc = safe_auc(y_true_bin_arr, y_score_bin_arr)
    binary_ap = safe_ap(y_true_bin_arr, y_score_bin_arr)

    if y_true_bin_arr.size == 0:
        binary_cm = [[0, 0], [0, 0]]
    else:
        binary_pred = (y_score_bin_arr >= threshold).astype(np.int64)
        binary_cm = confusion_matrix(y_true_bin_arr, binary_pred, labels=[0, 1]).tolist()

    labels = list(range(len(class_names)))
    if len(y_true_cls) == 0:
        cls_cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        per_class = [
            {
                "class_name": name,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
            for name in class_names
        ]
        macro_f1 = float("nan")
        weighted_f1 = float("nan")
    else:
        y_true_cls_arr = np.array(y_true_cls, dtype=np.int64)
        y_pred_cls_arr = np.array(y_pred_cls, dtype=np.int64)

        cls_cm = confusion_matrix(y_true_cls_arr, y_pred_cls_arr, labels=labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_cls_arr,
            y_pred_cls_arr,
            labels=labels,
            zero_division=0,
        )
        per_class = []
        for i, cname in enumerate(class_names):
            per_class.append(
                {
                    "class_name": cname,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
            )

        macro_f1 = float(f1_score(y_true_cls_arr, y_pred_cls_arr, labels=labels, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_true_cls_arr, y_pred_cls_arr, labels=labels, average="weighted", zero_division=0)
        )

    return {
        "binary": {
            "auc": binary_auc,
            "ap": binary_ap,
            "confusion_matrix_threshold": threshold,
            "confusion_matrix": binary_cm,
            "num_samples": int(y_true_bin_arr.size),
        },
        "classification": {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cls_cm.tolist(),
            "class_names": class_names,
            "per_class": per_class,
            "num_samples": int(len(y_true_cls)),
        },
        "records": records,
    }


def collect_pseudo_label_sanity(
    model: RTFMTRNClassifier,
    loader: DataLoader,
    device: str,
    pseudo_topk: int,
    limit: int,
) -> List[Dict[str, object]]:
    model.eval()
    out: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_bin = batch["binary_labels"].to(device)
            seg_anom_logits, _, _, _, _ = model(x, return_attention=False)
            seg_anom_probs = torch.sigmoid(seg_anom_logits)

            bsz, tlen = seg_anom_logits.shape
            for i in range(bsz):
                if int(y_bin[i].item()) != 1:
                    continue

                k = min(pseudo_topk, tlen)
                top_idx = torch.topk(seg_anom_logits[i], k=k).indices
                out.append(
                    {
                        "video_id": batch["video_ids"][i],
                        "class": str(batch["category_labels"][i]).lower(),
                        "selected_topk_segment_indices": top_idx.detach().cpu().numpy().astype(int).tolist(),
                        "selected_topk_anomaly_scores": (
                            seg_anom_probs[i, top_idx].detach().cpu().numpy().astype(float).tolist()
                        ),
                    }
                )

    out = sorted(out, key=lambda x: str(x["video_id"]))
    return out[:limit]


def load_checkpoint_weights(model: RTFMTRNClassifier, ckpt_path: Path, device: str) -> Dict[str, object]:
    if not ckpt_path.exists():
        return {"loaded": False, "reason": f"checkpoint not found: {ckpt_path}"}

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    model_state = model.state_dict()
    copied_keys = []
    updated = dict(model_state)
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            updated[k] = v
            copied_keys.append(k)

    model.load_state_dict(updated)

    return {
        "loaded": True,
        "checkpoint": str(ckpt_path),
        "num_copied": len(copied_keys),
        "copied_keys_preview": copied_keys[:12],
        "num_model_params": len(model_state),
        "num_new_params_uninitialized": len(model_state) - len(copied_keys),
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

    # Step-4 format
    return {
        "model": fallback_name,
        "auc": test.get("auc"),
        "ap": test.get("ap"),
        "macro_f1": None,
        "weighted_f1": None,
    }


def compare_cluster_span(cur_idx: List[int], prev_idx: Optional[List[int]]) -> str:
    if not prev_idx:
        return "no_step5_reference"
    cur_span = max(cur_idx) - min(cur_idx) if cur_idx else 0
    prev_span = max(prev_idx) - min(prev_idx) if prev_idx else 0
    if cur_span < prev_span:
        return "more_clustered_than_step5"
    if cur_span > prev_span:
        return "less_clustered_than_step5"
    return "similar_cluster_span_to_step5"


def get_refined_score_sanity(
    test_records: List[Dict[str, object]],
    step5_records_by_video: Dict[str, Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    anomalies = [r for r in test_records if int(r.get("binary_label", 0)) == 1]
    anomalies = sorted(anomalies, key=lambda r: float(r.get("anomaly_score", 0.0)), reverse=True)[:limit]

    out = []
    for r in anomalies:
        scores = np.array(r.get("refined_segment_scores", []), dtype=np.float64)
        if scores.size == 0:
            continue
        top_idx = np.argsort(scores)[-5:][::-1]
        top_idx_list = top_idx.astype(int).tolist()
        top_scores = [float(scores[i]) for i in top_idx]

        prev = step5_records_by_video.get(str(r["video_id"]))
        prev_idx = prev.get("top_anomaly_segment_indices") if prev else None
        clustering_note = compare_cluster_span(top_idx_list, prev_idx)

        out.append(
            {
                "video_id": r["video_id"],
                "class": r["category_label"],
                "top_refined_segment_indices": top_idx_list,
                "top_refined_scores": top_scores,
                "cluster_or_smooth_note": clustering_note,
            }
        )
    return out


def get_attention_sanity(
    model: RTFMTRNClassifier,
    loader: DataLoader,
    device: str,
) -> Dict[str, object]:
    model.eval()

    with torch.no_grad():
        for batch in loader:
            labels = batch["binary_labels"]
            idx_candidates = (labels == 1).nonzero(as_tuple=False)
            if idx_candidates.numel() == 0:
                continue
            i = int(idx_candidates[0].item())

            x = batch["features"][i : i + 1].to(device)
            _, _, _, _, attn_maps = model(x, return_attention=True)
            if not attn_maps:
                return {
                    "video_id": batch["video_ids"][i],
                    "attention_tensor_shape": None,
                    "note": "no attention maps returned",
                }

            last = attn_maps[-1].detach().cpu()  # [1, H, T, T]
            shape = list(last.shape)
            row_max_mean = float(last.max(dim=-1).values.mean().item())
            t = max(1, shape[-1])
            uniform_level = 1.0 / float(t)
            note = "concentrated" if row_max_mean > (uniform_level * 3.0) else "diffuse"

            return {
                "video_id": batch["video_ids"][i],
                "category_label": batch["category_labels"][i],
                "attention_tensor_shape": shape,
                "row_max_mean": row_max_mean,
                "uniform_reference": uniform_level,
                "note": note,
            }

    return {
        "video_id": None,
        "attention_tensor_shape": None,
        "note": "no anomaly sample found for attention sanity",
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
    init_ckpt = resolve(project_root, args.init_ckpt)
    output_dir = resolve(project_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list {class_names}")
    normal_idx = class_to_idx[args.normal_class]

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
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=fixed_segments_collate,
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

    model = RTFMTRNClassifier(
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
        train_loss, train_bce, train_rtfm, train_cls, train_smooth = train_one_epoch(
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
            smooth_lambda=args.smooth_lambda,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            class_to_idx=class_to_idx,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            pseudo_topk=args.pseudo_topk,
            threshold=args.threshold,
            collect_records=False,
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
            f"(bce={train_bce:.5f}, rtfm={train_rtfm:.5f}, cls={train_cls:.5f}, smooth={train_smooth:.5f}) | "
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

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names,
        class_to_idx=class_to_idx,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        pseudo_topk=args.pseudo_topk,
        threshold=args.threshold,
        collect_records=True,
    )

    train_pseudo = collect_pseudo_label_sanity(
        model=model,
        loader=train_eval_loader,
        device=device,
        pseudo_topk=args.pseudo_topk,
        limit=5,
    )

    test_records: List[Dict[str, object]] = test_metrics["records"]
    normal_examples = sorted(
        [r for r in test_records if int(r["binary_label"]) == 0],
        key=lambda x: str(x["video_id"]),
    )[:5]
    anomaly_examples = sorted(
        [r for r in test_records if int(r["binary_label"]) == 1],
        key=lambda x: str(x["video_id"]),
    )[:5]

    step5_records_path = resolve(project_root, Path("outputs/rtfm_classifier/test_records.json"))
    step5_records = load_json_if_exists(step5_records_path) or []
    step5_by_video = {str(r.get("video_id")): r for r in step5_records if isinstance(r, dict)}

    refined_sanity = get_refined_score_sanity(
        test_records=test_records,
        step5_records_by_video=step5_by_video,
        limit=3,
    )

    attention_sanity = get_attention_sanity(
        model=model,
        loader=test_loader,
        device=device,
    )

    qualitative_test_class = []
    for r in anomaly_examples:
        qualitative_test_class.append(
            {
                "video_id": r["video_id"],
                "ground_truth_class": r["category_label"],
                "predicted_class": r["pred_class_label"],
                "top_predicted_probability": r["pred_class_prob"],
                "top_anomaly_segments": r["top_anomaly_segment_indices"],
                "top_anomaly_scores": r["top_anomaly_segment_scores"],
            }
        )

    step4_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_baseline/results_summary.json")))
    step5_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_classifier/results_summary.json")))

    comparison = [
        extract_model_metrics(step4_summary, "Step 4 baseline"),
        extract_model_metrics(step5_summary, "Step 5 + classifier"),
        {
            "model": "Step 6 + TRN",
            "auc": test_metrics["binary"]["auc"],
            "ap": test_metrics["binary"]["ap"],
            "macro_f1": test_metrics["classification"]["macro_f1"],
            "weighted_f1": test_metrics["classification"]["weighted_f1"],
        },
    ]

    results = {
        "architecture": {
            "pipeline": "sampled feature sequence -> projection -> TRN -> refined states -> anomaly/classifier heads",
            "projection_dim": args.hidden_dim,
            "positional_encoding": args.pos_encoding,
            "trn_layers": args.trn_layers,
            "trn_heads": args.trn_heads,
            "trn_ffn_dim": args.hidden_dim * args.trn_ffn_mult,
            "trn_dropout": args.trn_dropout,
            "anomaly_head_attach": "on refined states",
            "classifier_head_attach": "on refined states",
            "taxonomy": {
                "mode": "ucf_6_class",
                "class_names": class_names,
                "note": "Riot is absent in UCF-Crime and excluded from this run.",
            },
            "temporal_sampling_rule": "uniform sample to target_segments if longer; pad by repeating last segment if shorter",
            "pseudo_label_selection_rule": (
                f"anomalous videos: top-{args.pseudo_topk} segments by anomaly score -> video category label; "
                f"normal videos: all sampled {args.target_segments} segments -> normal label"
            ),
        },
        "training_setup": {
            "checkpoint_initialization": str(init_ckpt),
            "initialization_details": init_info,
            "losses": {
                "anomaly": f"BCE + {args.rtfm_lambda}*RTFM",
                "classification": f"{args.cls_lambda}*CrossEntropy",
                "smoothness": f"{args.smooth_lambda}*Smoothness(refined anomaly scores)",
                "total": "anomaly + cls + smoothness",
            },
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "checkpoint_rule": f"best {args.checkpoint_metric}, tie-break by val_auc",
            "device": device,
            "threshold": args.threshold,
            "seed": args.seed,
            "balanced_sampler": bool(args.balanced_sampler),
            "pos_weight": pos_weight,
        },
        "training_curves": [
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss,
                "train_bce_loss": h.train_bce_loss,
                "train_rtfm_loss": h.train_rtfm_loss,
                "train_cls_loss": h.train_cls_loss,
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
        "test": {
            "binary": test_metrics["binary"],
            "classification": test_metrics["classification"],
            "normal_score_examples": [
                {
                    "video_id": r["video_id"],
                    "category_label": r["category_label"],
                    "anomaly_score": r["anomaly_score"],
                }
                for r in normal_examples
            ],
            "anomaly_score_examples": [
                {
                    "video_id": r["video_id"],
                    "category_label": r["category_label"],
                    "anomaly_score": r["anomaly_score"],
                }
                for r in anomaly_examples
            ],
        },
        "pseudo_label_sanity_train": train_pseudo,
        "qualitative_test_class_predictions": qualitative_test_class,
        "refined_score_sanity": refined_sanity,
        "attention_sanity": attention_sanity,
        "comparison_table": comparison,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_records.json").write_text(json.dumps(test_records, indent=2) + "\n")

    print("\nFinal Step-6 test metrics")
    print(f"- Binary AUC: {results['test']['binary']['auc']:.6f}")
    print(f"- Binary AP:  {results['test']['binary']['ap']:.6f}")
    print(f"- Binary confusion @ {args.threshold}: {results['test']['binary']['confusion_matrix']}")
    print(f"- Macro-F1:   {results['test']['classification']['macro_f1']:.6f}")
    print(f"- Weighted-F1:{results['test']['classification']['weighted_f1']:.6f}")
    print(f"- Saved: {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
