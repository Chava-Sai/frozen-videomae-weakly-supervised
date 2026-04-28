#!/usr/bin/env python3
"""Step-8: Progressive 3-stage training + val-only temporal inference calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from feature_dataset import FeatureSequenceDataset, fixed_segments_collate
from train_rtfm_trn_boundary import (
    RTFMTRNBoundary,
    best_iou_against_gt,
    boundary_loss,
    build_loader,
    choose_device,
    choose_failure_case,
    evaluate_localization_map,
    extract_model_metrics,
    load_gt_events_for_test,
    load_json_if_exists,
    metric_key,
    moving_average,
    parse_class_names,
    parse_float_list,
    read_csv,
    refine_spans_with_boundary,
    resolve,
    safe_ap,
    safe_auc,
    set_seed,
    smoothness_loss,
    spans_from_scores,
    spans_to_events,
    topk_mean,
    topk_mean_np,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-8 progressive training + temporal calibration")
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
        type=str,
        default="",
        help="Optional checkpoint to initialize overlapping weights before progressive training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rtfm_progressive"),
    )

    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage3-epochs", type=int, default=40)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--target-segments", type=int, default=32)

    parser.add_argument("--topk-ratio", type=float, default=0.125)
    parser.add_argument("--pseudo-topk", type=int, default=4)

    parser.add_argument("--rtfm-margin", type=float, default=5.0)
    parser.add_argument("--rtfm-lambda", type=float, default=0.1)
    parser.add_argument("--stage2-cls-lambda", type=float, default=0.5)
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

    parser.add_argument("--threshold-candidates", type=str, default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--smooth-window-candidates", type=str, default="3,5,7")
    parser.add_argument("--min-event-len-candidates", type=str, default="1,2,3")
    parser.add_argument("--merge-gap-candidates", type=str, default="0,1,2")
    parser.add_argument("--boundary-radius-candidates", type=str, default="1,2,3")
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
        "--checkpoint-stage",
        choices=["all", "stage3"],
        default="stage3",
        help="Choose best checkpoint over all epochs or only stage-3 epochs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for x in raw.split(","):
        s = x.strip()
        if s:
            vals.append(int(s))
    if not vals:
        raise ValueError(f"No integer values parsed from: {raw}")
    return vals


class ProgressiveRTFM(RTFMTRNBoundary):
    """Same architecture as Step-7, with optional TRN/boundary application per stage."""

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        apply_trn: bool = True,
        apply_boundary: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z = self.feature_proj(x)

        attn_maps: List[torch.Tensor] = []
        if apply_trn:
            z = self.add_positional_encoding(z)
            for layer in self.trn_layers:
                z, attn_w = layer(z, return_attention=return_attention)
                if return_attention and attn_w is not None:
                    attn_maps.append(attn_w)

        seg_anom_logits = self.segment_head(z).squeeze(-1)
        seg_class_logits = self.class_head(z)
        feat_magnitudes = torch.norm(z, p=2, dim=-1)

        if apply_boundary and z.shape[1] > 1:
            s_prob = torch.sigmoid(seg_anom_logits)
            diff = torch.abs(s_prob[:, 1:] - s_prob[:, :-1]).unsqueeze(-1)
            edge_feat = torch.cat([z[:, :-1, :], z[:, 1:, :], diff], dim=-1)
            bnd_scores = self.boundary_head(edge_feat).squeeze(-1)
        else:
            bnd_scores = torch.zeros((z.shape[0], max(0, z.shape[1] - 1)), device=z.device, dtype=z.dtype)

        return seg_anom_logits, seg_class_logits, feat_magnitudes, bnd_scores, z, attn_maps


@dataclass
class StageConfig:
    stage_id: int
    stage_name: str
    start_epoch: int
    end_epoch: int
    use_classifier: bool
    use_trn: bool
    use_boundary: bool
    cls_lambda: float
    bnd_lambda: float
    smooth_lambda: float
    trainable_modules: List[str]


@dataclass
class EpochResult:
    epoch: int
    stage_id: int
    stage_name: str
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


def build_stage_schedule(args: argparse.Namespace) -> List[StageConfig]:
    s1_end = args.stage1_epochs
    s2_end = s1_end + args.stage2_epochs
    s3_end = s2_end + args.stage3_epochs

    return [
        StageConfig(
            stage_id=1,
            stage_name="stage1_mil_only",
            start_epoch=1,
            end_epoch=s1_end,
            use_classifier=False,
            use_trn=False,
            use_boundary=False,
            cls_lambda=0.0,
            bnd_lambda=0.0,
            smooth_lambda=0.0,
            trainable_modules=["feature_proj", "segment_head"],
        ),
        StageConfig(
            stage_id=2,
            stage_name="stage2_mil_classifier",
            start_epoch=s1_end + 1,
            end_epoch=s2_end,
            use_classifier=True,
            use_trn=False,
            use_boundary=False,
            cls_lambda=args.stage2_cls_lambda,
            bnd_lambda=0.0,
            smooth_lambda=0.0,
            trainable_modules=["feature_proj", "segment_head", "class_head"],
        ),
        StageConfig(
            stage_id=3,
            stage_name="stage3_full_trn_boundary",
            start_epoch=s2_end + 1,
            end_epoch=s3_end,
            use_classifier=True,
            use_trn=True,
            use_boundary=True,
            cls_lambda=args.cls_lambda,
            bnd_lambda=args.bnd_lambda,
            smooth_lambda=args.smooth_lambda,
            trainable_modules=["all"],
        ),
    ]


def stage_for_epoch(stages: Sequence[StageConfig], epoch: int) -> StageConfig:
    for s in stages:
        if s.start_epoch <= epoch <= s.end_epoch:
            return s
    raise ValueError(f"Epoch {epoch} not covered by stage schedule")


def set_trainable_for_stage(model: ProgressiveRTFM, stage: StageConfig) -> Dict[str, object]:
    for p in model.parameters():
        p.requires_grad = False

    if "all" in stage.trainable_modules:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for mod_name in stage.trainable_modules:
            module = getattr(model, mod_name)
            for p in module.parameters():
                p.requires_grad = True

    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_params = sum(int(p.numel()) for p in model.parameters())
    return {
        "stage_id": stage.stage_id,
        "stage_name": stage.stage_name,
        "trainable_modules": stage.trainable_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def make_optimizer(model: ProgressiveRTFM, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for current stage")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def load_checkpoint_weights(model: ProgressiveRTFM, ckpt_path: Optional[Path], device: str) -> Dict[str, object]:
    if ckpt_path is None:
        return {"loaded": False, "reason": "no checkpoint provided"}
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


def train_one_epoch_stage(
    model: ProgressiveRTFM,
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
    stage: StageConfig,
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

        seg_anom_logits, seg_class_logits, feat_mags, bnd_scores, _, _ = model(
            x,
            return_attention=False,
            apply_trn=stage.use_trn,
            apply_boundary=stage.use_boundary,
        )

        video_logits, _ = topk_mean(seg_anom_logits, topk_ratio)
        bce_loss = bce_fn(video_logits, y_bin)

        rtfm_loss = torch.tensor(0.0, device=device)
        anom_mask = y_bin > 0.5
        norm_mask = y_bin <= 0.5
        if anom_mask.any() and norm_mask.any():
            anom_topk, _ = topk_mean(feat_mags[anom_mask], topk_ratio)
            norm_topk, _ = topk_mean(feat_mags[norm_mask], topk_ratio)
            rtfm_loss = torch.relu(rtfm_margin - (anom_topk.mean() - norm_topk.mean()))

        cls_loss = torch.tensor(0.0, device=device)
        if stage.use_classifier:
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

            if cls_losses:
                cls_loss = torch.stack(cls_losses).mean()

        bnd_loss = boundary_loss(seg_anom_logits, bnd_scores) if stage.use_boundary else torch.tensor(0.0, device=device)
        s_loss = smoothness_loss(seg_anom_logits) if stage.use_boundary else torch.tensor(0.0, device=device)

        anomaly_loss = bce_loss + rtfm_lambda * rtfm_loss
        loss = anomaly_loss + stage.cls_lambda * cls_loss + stage.bnd_lambda * bnd_loss + stage.smooth_lambda * s_loss

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


def evaluate_sampled_stage(
    model: ProgressiveRTFM,
    loader: DataLoader,
    device: str,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    normal_idx: int,
    topk_ratio: float,
    pseudo_topk: int,
    threshold: float,
    stage: StageConfig,
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

            seg_anom_logits, seg_class_logits, _, _, _, _ = model(
                x,
                return_attention=False,
                apply_trn=stage.use_trn,
                apply_boundary=False,
            )
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

                if stage.use_classifier:
                    k = min(pseudo_topk, tlen)
                    top_idx = torch.topk(seg_anom_logits[i], k=k).indices
                    seg_cls_probs = torch.softmax(seg_class_logits[i, top_idx, :], dim=-1)
                    video_cls_probs = seg_cls_probs.mean(dim=0)
                    raw_pred = int(torch.argmax(video_cls_probs).item())
                    pred_class = normal_idx if anom_prob < threshold else raw_pred
                else:
                    pred_class = normal_idx

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


def infer_full_sequence_chunked_stage3(
    model: ProgressiveRTFM,
    features: np.ndarray,
    device: str,
    num_classes: int,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = int(features.shape[0])
    if t <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, num_classes), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

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
            seg_logits, seg_cls_logits, _, bnd_scores, _, _ = model(
                x,
                return_attention=False,
                apply_trn=True,
                apply_boundary=True,
            )
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


def infer_video_raw(
    model: ProgressiveRTFM,
    row: Dict[str, str],
    project_root: Path,
    device: str,
    num_classes: int,
    infer_window: int,
    infer_stride: int,
) -> Dict[str, object]:
    feat_path = resolve(project_root, Path(row["feature_path"]))
    with np.load(feat_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        starts = data["segment_start_frames"].astype(np.int64)
        ends = data["segment_end_frames"].astype(np.int64)

    seg_scores, cls_probs, bnd_scores = infer_full_sequence_chunked_stage3(
        model=model,
        features=feats,
        device=device,
        num_classes=num_classes,
        window=infer_window,
        stride=infer_stride,
    )

    return {
        "video_id": row["video_id"],
        "split": row["split"],
        "binary_label": int(row["binary_label"]),
        "category_label": str(row["category_label"]),
        "fps": float(row["fps"]),
        "duration_sec": float(row["duration_sec"]),
        "num_segments": int(feats.shape[0]),
        "segment_start_frames": starts,
        "segment_end_frames": ends,
        "segment_scores_raw": seg_scores,
        "segment_class_probs_raw": cls_probs,
        "boundary_scores_raw": bnd_scores,
    }


def postprocess_video_raw(
    raw: Dict[str, object],
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
) -> Dict[str, object]:
    seg_scores_raw = np.asarray(raw["segment_scores_raw"], dtype=np.float32)
    cls_probs_raw = np.asarray(raw["segment_class_probs_raw"], dtype=np.float32)
    bnd_scores_raw = np.asarray(raw["boundary_scores_raw"], dtype=np.float32)
    starts = np.asarray(raw["segment_start_frames"], dtype=np.int64)
    ends = np.asarray(raw["segment_end_frames"], dtype=np.int64)

    seg_scores_smooth = moving_average(seg_scores_raw, smooth_window)

    spans_before = spans_from_scores(seg_scores_smooth, threshold=threshold, min_len=min_event_len, merge_gap=merge_gap)
    spans_after = refine_spans_with_boundary(
        spans=spans_before,
        boundary_scores=bnd_scores_raw,
        total_segments=int(seg_scores_smooth.shape[0]),
        radius=boundary_radius,
        min_len=min_event_len,
        merge_gap=merge_gap,
    )

    fps = float(raw["fps"])
    events_before = spans_to_events(
        spans=spans_before,
        scores=seg_scores_smooth,
        class_probs=cls_probs_raw,
        segment_starts=starts,
        segment_ends=ends,
        fps=fps,
        class_names=class_names,
        normal_idx=normal_idx,
    )
    events_after = spans_to_events(
        spans=spans_after,
        scores=seg_scores_smooth,
        class_probs=cls_probs_raw,
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
    if bnd_scores_raw.size > 0:
        k = int(min(5, bnd_scores_raw.shape[0]))
        idx = np.argpartition(bnd_scores_raw, -k)[-k:]
        idx = idx[np.argsort(-bnd_scores_raw[idx])]
        topk_bnd = [{"edge_index": int(i), "value": float(bnd_scores_raw[i])} for i in idx.tolist()]

    return {
        "video_id": raw["video_id"],
        "split": raw["split"],
        "binary_label": int(raw["binary_label"]),
        "category_label": raw["category_label"],
        "num_segments": int(raw["num_segments"]),
        "fps": float(raw["fps"]),
        "duration_sec": float(raw["duration_sec"]),
        "video_anomaly_score": float(video_anomaly_score),
        "pred_video_class": pred_class,
        "pred_video_class_idx": pred_class_idx,
        "pred_video_class_prob": float(pred_class_prob),
        "spans_before_refine": [[int(s), int(e)] for s, e in spans_before],
        "spans_after_refine": [[int(s), int(e)] for s, e in spans_after],
        "events_before_refine": events_before,
        "events_after_refine": events_after,
        "top_boundary_peaks": topk_bnd,
    }


def process_all_videos_with_setting(
    raw_list: Sequence[Dict[str, object]],
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    threshold: float,
    smooth_window: int,
    min_event_len: int,
    merge_gap: int,
    boundary_radius: int,
) -> List[Dict[str, object]]:
    out = []
    for raw in raw_list:
        out.append(
            postprocess_video_raw(
                raw=raw,
                class_names=class_names,
                normal_idx=normal_idx,
                topk_ratio=topk_ratio,
                threshold=threshold,
                smooth_window=smooth_window,
                min_event_len=min_event_len,
                merge_gap=merge_gap,
                boundary_radius=boundary_radius,
            )
        )
    return out


def summarize_val_setting(processed: Sequence[Dict[str, object]], threshold: float) -> Dict[str, object]:
    if not processed:
        return {
            "binary_f1": float("nan"),
            "binary_precision": float("nan"),
            "binary_recall": float("nan"),
            "pred_event_count": 0,
            "avg_events_per_positive_video": float("nan"),
            "num_positive_videos": 0,
        }

    y_true = np.array([int(r["binary_label"]) for r in processed], dtype=np.int64)
    y_score = np.array([float(r["video_anomaly_score"]) for r in processed], dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    binary_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    pred_pos = int((y_pred == 1).sum())
    true_pos = int((y_true == 1).sum())
    prec = float(tp / pred_pos) if pred_pos > 0 else 0.0
    rec = float(tp / true_pos) if true_pos > 0 else 0.0

    pred_event_count = int(sum(len(r.get("events_after_refine", [])) for r in processed))
    num_positive_videos = int((y_true == 1).sum())
    avg_events = float(pred_event_count / max(num_positive_videos, 1))

    return {
        "binary_f1": binary_f1,
        "binary_precision": prec,
        "binary_recall": rec,
        "pred_event_count": pred_event_count,
        "avg_events_per_positive_video": avg_events,
        "num_positive_videos": num_positive_videos,
    }


def val_setting_key(stats: Dict[str, object]) -> Tuple[float, float, float, float]:
    f1 = float(stats["binary_f1"])
    avg_events = float(stats["avg_events_per_positive_video"])
    pred_events = int(stats["pred_event_count"])
    # Primary: binary F1. Tie-breakers bias toward event count near 1/event-positive-video and fewer events.
    return (
        f1,
        -abs(avg_events - 1.0),
        -float(max(0.0, avg_events - 1.5)),
        -float(pred_events),
    )


def calibrate_on_val(
    val_raw: Sequence[Dict[str, object]],
    class_names: List[str],
    normal_idx: int,
    topk_ratio: float,
    threshold_candidates: List[float],
    smooth_candidates: List[int],
    min_len_candidates: List[int],
    merge_gap_candidates: List[int],
    boundary_radius_candidates: List[int],
) -> Dict[str, object]:
    settings = {
        "threshold": float(threshold_candidates[0]),
        "smooth_window": int(smooth_candidates[0]),
        "min_event_len": int(min_len_candidates[0]),
        "merge_gap": int(merge_gap_candidates[0]),
        "boundary_radius": int(boundary_radius_candidates[0]),
    }

    tuning_log: List[Dict[str, object]] = []

    # Sequential val-only calibration.
    tuning_steps = [
        ("threshold", [float(x) for x in threshold_candidates]),
        ("smooth_window", [int(x) for x in smooth_candidates]),
        ("min_event_len", [int(x) for x in min_len_candidates]),
        ("merge_gap", [int(x) for x in merge_gap_candidates]),
        ("boundary_radius", [int(x) for x in boundary_radius_candidates]),
    ]

    for param_name, candidates in tuning_steps:
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_setting_value = settings[param_name]
        best_stats: Optional[Dict[str, object]] = None

        for candidate in candidates:
            trial = dict(settings)
            trial[param_name] = candidate

            processed = process_all_videos_with_setting(
                raw_list=val_raw,
                class_names=class_names,
                normal_idx=normal_idx,
                topk_ratio=topk_ratio,
                threshold=float(trial["threshold"]),
                smooth_window=int(trial["smooth_window"]),
                min_event_len=int(trial["min_event_len"]),
                merge_gap=int(trial["merge_gap"]),
                boundary_radius=int(trial["boundary_radius"]),
            )
            stats = summarize_val_setting(processed, threshold=float(trial["threshold"]))
            key = val_setting_key(stats)

            tuning_log.append(
                {
                    "step": param_name,
                    "candidate": candidate,
                    "trial_settings": trial,
                    "stats": stats,
                    "selection_key": list(key),
                }
            )

            if best_key is None or key > best_key:
                best_key = key
                best_setting_value = candidate
                best_stats = stats

        settings[param_name] = best_setting_value
        tuning_log.append(
            {
                "step_result": param_name,
                "selected_value": best_setting_value,
                "selected_settings": dict(settings),
                "selected_stats": best_stats,
            }
        )

    final_processed = process_all_videos_with_setting(
        raw_list=val_raw,
        class_names=class_names,
        normal_idx=normal_idx,
        topk_ratio=topk_ratio,
        threshold=float(settings["threshold"]),
        smooth_window=int(settings["smooth_window"]),
        min_event_len=int(settings["min_event_len"]),
        merge_gap=int(settings["merge_gap"]),
        boundary_radius=int(settings["boundary_radius"]),
    )
    final_stats = summarize_val_setting(final_processed, threshold=float(settings["threshold"]))

    return {
        "selected_settings": settings,
        "final_val_stats": final_stats,
        "tuning_log": tuning_log,
        "boundary_peak_rule": "for each preliminary boundary, search +/- boundary_radius edges and snap to local max b_t",
    }


def build_improvements_vs_step7(
    step8_test_results: Sequence[Dict[str, object]],
    step7_test_results: Optional[Sequence[Dict[str, object]]],
    gt_by_video: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    if step7_test_results is None:
        return []

    step7_by_video = {str(r.get("video_id")): r for r in step7_test_results}
    improved: List[Dict[str, object]] = []

    for r8 in step8_test_results:
        vid = str(r8.get("video_id"))
        if int(r8.get("binary_label", 0)) == 0:
            continue
        if vid not in step7_by_video:
            continue
        gt = gt_by_video.get(vid, [])
        if not gt:
            continue

        r7 = step7_by_video[vid]
        spans7 = [tuple(x) for x in r7.get("spans_after_refine", [])]
        spans8 = [tuple(x) for x in r8.get("spans_after_refine", [])]
        iou7 = best_iou_against_gt(spans7, gt)
        iou8 = best_iou_against_gt(spans8, gt)

        if iou8 > iou7 + 1e-8:
            improved.append(
                {
                    "video_id": vid,
                    "ground_truth_class": r8.get("category_label"),
                    "gt_spans": [[int(g["start_segment"]), int(g["end_segment"])] for g in gt],
                    "step7_spans_after_refine": [[int(a), int(b)] for (a, b) in spans7],
                    "step8_spans_after_refine": [[int(a), int(b)] for (a, b) in spans8],
                    "step7_best_iou": float(iou7),
                    "step8_best_iou": float(iou8),
                    "improvement": float(iou8 - iou7),
                }
            )

    improved.sort(key=lambda x: float(x["improvement"]), reverse=True)
    return improved[:3]


def save_history(history: List[EpochResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_id",
                "stage_name",
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
                    h.stage_id,
                    h.stage_name,
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

    fig, ax1 = plt.subplots(figsize=(9.0, 5.0))
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
    output_dir = resolve(project_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' not found in class list {class_names}")
    normal_idx = class_to_idx[args.normal_class]

    threshold_candidates = parse_float_list(args.threshold_candidates)
    smooth_window_candidates = parse_int_list(args.smooth_window_candidates)
    min_event_len_candidates = parse_int_list(args.min_event_len_candidates)
    merge_gap_candidates = parse_int_list(args.merge_gap_candidates)
    boundary_radius_candidates = parse_int_list(args.boundary_radius_candidates)
    tiou_thresholds = parse_float_list(args.localization_tiou)

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

    model = ProgressiveRTFM(
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

    init_ckpt: Optional[Path] = None
    if args.init_ckpt.strip():
        p = Path(args.init_ckpt.strip())
        init_ckpt = p if p.is_absolute() else (project_root / p)
    init_info = load_checkpoint_weights(model, init_ckpt, device)

    train_labels = np.array([int(r["binary_label"]) for r in train_ds.rows], dtype=np.int64)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    if args.balanced_sampler:
        pos_weight = 1.0

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    stages = build_stage_schedule(args)
    total_epochs = stages[-1].end_epoch

    history: List[EpochResult] = []
    stage_freeze_log: List[Dict[str, object]] = []
    best_key = (-float("inf"), -float("inf"))
    best_epoch = -1

    optimizer: Optional[torch.optim.Optimizer] = None
    current_stage_id: Optional[int] = None

    for epoch in range(1, total_epochs + 1):
        stage = stage_for_epoch(stages, epoch)

        if current_stage_id != stage.stage_id:
            freeze_info = set_trainable_for_stage(model, stage)
            stage_freeze_log.append(freeze_info)
            optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
            current_stage_id = stage.stage_id

            print(
                f"\n[Stage {stage.stage_id}] {stage.stage_name} | epochs {stage.start_epoch}-{stage.end_epoch} | "
                f"trainable={freeze_info['trainable_modules']} ({freeze_info['trainable_params']}/{freeze_info['total_params']} params)"
            )

        assert optimizer is not None

        train_loss, train_bce, train_rtfm, train_cls, train_bnd, train_smooth = train_one_epoch_stage(
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
            stage=stage,
        )

        val_metrics = evaluate_sampled_stage(
            model=model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            class_to_idx=class_to_idx,
            normal_idx=normal_idx,
            topk_ratio=args.topk_ratio,
            pseudo_topk=args.pseudo_topk,
            threshold=float(threshold_candidates[0]),
            stage=stage,
        )

        val_auc = float(val_metrics["binary"]["auc"])
        val_ap = float(val_metrics["binary"]["ap"])
        val_macro = float(val_metrics["classification"]["macro_f1"])
        val_weighted = float(val_metrics["classification"]["weighted_f1"])

        history.append(
            EpochResult(
                epoch=epoch,
                stage_id=stage.stage_id,
                stage_name=stage.stage_name,
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
            f"Epoch {epoch:03d} [{stage.stage_name}] | train_loss={train_loss:.5f} "
            f"(bce={train_bce:.5f}, rtfm={train_rtfm:.5f}, cls={train_cls:.5f}, bnd={train_bnd:.5f}, smooth={train_smooth:.5f}) | "
            f"val_auc={val_auc:.5f} val_ap={val_ap:.5f} val_macro_f1={val_macro:.5f} val_weighted_f1={val_weighted:.5f}"
        )

        can_checkpoint = args.checkpoint_stage == "all" or stage.stage_id == 3
        if can_checkpoint and cur_key > best_key:
            best_key = cur_key
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "stage_id": stage.stage_id,
                    "stage_name": stage.stage_name,
                    "model_state": model.state_dict(),
                    "args": json_safe_args(args),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "val_macro_f1": val_macro,
                    "val_weighted_f1": val_weighted,
                },
                ckpt_dir / "best.pt",
            )

        if epoch == stage.end_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "stage_id": stage.stage_id,
                    "stage_name": stage.stage_name,
                    "model_state": model.state_dict(),
                    "args": json_safe_args(args),
                },
                ckpt_dir / f"stage{stage.stage_id}_end.pt",
            )

    save_history(history, output_dir / "train_history.csv")
    maybe_plot_curves(history, output_dir / "train_curves.png")

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        raise RuntimeError("No best checkpoint saved. Check stage configuration/checkpoint-stage settings.")

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    feat_rows = [r for r in read_csv(feature_manifest) if r.get("status") == "ok"]
    val_rows = [r for r in feat_rows if r.get("split") == "val"]
    test_rows = [r for r in feat_rows if r.get("split") == "test"]

    master_rows = read_csv(master_manifest)
    master_by_video = {r["video_id"]: r for r in master_rows}

    print("\nRunning full-sequence model inference cache on val split...")
    val_raw: List[Dict[str, object]] = []
    for i, r in enumerate(val_rows, 1):
        val_raw.append(
            infer_video_raw(
                model=model,
                row=r,
                project_root=project_root,
                device=device,
                num_classes=len(class_names),
                infer_window=args.infer_window,
                infer_stride=args.infer_stride,
            )
        )
        if i % 20 == 0 or i == len(val_rows):
            print(f"  val raw inference {i}/{len(val_rows)}")

    print("\nCalibrating temporal inference settings on val split only...")
    calibration = calibrate_on_val(
        val_raw=val_raw,
        class_names=class_names,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold_candidates=threshold_candidates,
        smooth_candidates=smooth_window_candidates,
        min_len_candidates=min_event_len_candidates,
        merge_gap_candidates=merge_gap_candidates,
        boundary_radius_candidates=boundary_radius_candidates,
    )
    selected = calibration["selected_settings"]
    print(
        "Selected val settings: "
        f"threshold={selected['threshold']}, smooth_window={selected['smooth_window']}, "
        f"min_event_len={selected['min_event_len']}, merge_gap={selected['merge_gap']}, "
        f"boundary_radius={selected['boundary_radius']}"
    )

    print("\nRunning full-sequence model inference cache on test split...")
    test_raw: List[Dict[str, object]] = []
    for i, r in enumerate(test_rows, 1):
        test_raw.append(
            infer_video_raw(
                model=model,
                row=r,
                project_root=project_root,
                device=device,
                num_classes=len(class_names),
                infer_window=args.infer_window,
                infer_stride=args.infer_stride,
            )
        )
        if i % 25 == 0 or i == len(test_rows):
            print(f"  test raw inference {i}/{len(test_rows)}")

    test_results = process_all_videos_with_setting(
        raw_list=test_raw,
        class_names=class_names,
        normal_idx=normal_idx,
        topk_ratio=args.topk_ratio,
        threshold=float(selected["threshold"]),
        smooth_window=int(selected["smooth_window"]),
        min_event_len=int(selected["min_event_len"]),
        merge_gap=int(selected["merge_gap"]),
        boundary_radius=int(selected["boundary_radius"]),
    )

    gt_events = load_gt_events_for_test(
        test_rows=test_rows,
        master_rows_by_video=master_by_video,
        temporal_root=temporal_root,
    )
    gt_by_video: Dict[str, List[Dict[str, object]]] = {}
    for g in gt_events:
        gt_by_video.setdefault(str(g["video_id"]), []).append(g)

    pred_events_all: List[Dict[str, object]] = []
    for r in test_results:
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

    y_true_bin = np.array([int(r["binary_label"]) for r in test_results], dtype=np.int64)
    y_score_bin = np.array([float(r["video_anomaly_score"]) for r in test_results], dtype=np.float64)
    y_pred_bin = (y_score_bin >= float(selected["threshold"])).astype(np.int64)

    binary_auc = safe_auc(y_true_bin, y_score_bin)
    binary_ap = safe_ap(y_true_bin, y_score_bin)
    binary_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist() if y_true_bin.size > 0 else [[0, 0], [0, 0]]

    y_true_cls = np.array([class_to_idx.get(str(r["category_label"]), normal_idx) for r in test_results], dtype=np.int64)
    y_pred_cls = np.array([class_to_idx.get(str(r["pred_video_class"]), normal_idx) for r in test_results], dtype=np.int64)

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

    failure_case = choose_failure_case(test_results, gt_by_video, float(selected["threshold"]))

    step7_summary = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_trn_boundary/results_summary.json")))
    step7_test_results = load_json_if_exists(resolve(project_root, Path("outputs/rtfm_trn_boundary/test_video_results.json")))
    if isinstance(step7_test_results, dict):
        step7_test_results = None

    qualitative_improved = build_improvements_vs_step7(
        step8_test_results=test_results,
        step7_test_results=step7_test_results if isinstance(step7_test_results, list) else None,
        gt_by_video=gt_by_video,
    )

    before_after_table = []
    if step7_summary is not None:
        s7_upd = step7_summary.get("updated_metrics", {})
        s7_bin = s7_upd.get("binary", {})
        s7_cls = s7_upd.get("classification", {})
        s7_loc = step7_summary.get("temporal_localization", {}).get("tiou", {})
        before_after_table.append(
            {
                "model": "Step 7 direct training",
                "auc": s7_bin.get("auc"),
                "ap": s7_bin.get("ap"),
                "macro_f1": s7_cls.get("macro_f1"),
                "weighted_f1": s7_cls.get("weighted_f1"),
                "mAP@0.3": s7_loc.get("0.3", {}).get("mAP") if isinstance(s7_loc, dict) else None,
                "mAP@0.5": s7_loc.get("0.5", {}).get("mAP") if isinstance(s7_loc, dict) else None,
                "mAP@0.7": s7_loc.get("0.7", {}).get("mAP") if isinstance(s7_loc, dict) else None,
            }
        )

    before_after_table.append(
        {
            "model": "Step 8 progressive training",
            "auc": binary_auc,
            "ap": binary_ap,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mAP@0.3": localization["tiou"].get("0.3", {}).get("mAP"),
            "mAP@0.5": localization["tiou"].get("0.5", {}).get("mAP"),
            "mAP@0.7": localization["tiou"].get("0.7", {}).get("mAP"),
        }
    )

    comparison_table = [
        extract_model_metrics(load_json_if_exists(resolve(project_root, Path("outputs/rtfm_baseline/results_summary.json"))), "Step 4 baseline"),
        extract_model_metrics(load_json_if_exists(resolve(project_root, Path("outputs/rtfm_classifier/results_summary.json"))), "Step 5 + classifier"),
        extract_model_metrics(load_json_if_exists(resolve(project_root, Path("outputs/rtfm_trn/results_summary.json"))), "Step 6 + TRN"),
        extract_model_metrics(step7_summary, "Step 7 + Boundary"),
        {
            "model": "Step 8 progressive",
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
            "training_sampling_rule": "uniform sample to 32; pad by repeating last segment if shorter",
            "inference_sampling_rule": "full cached sequence via sliding windows",
        },
        "progressive_schedule": [
            {
                "stage_id": s.stage_id,
                "stage_name": s.stage_name,
                "epoch_range": [s.start_epoch, s.end_epoch],
                "use_classifier": s.use_classifier,
                "use_trn": s.use_trn,
                "use_boundary": s.use_boundary,
                "loss_weights": {
                    "cls_lambda": s.cls_lambda,
                    "bnd_lambda": s.bnd_lambda,
                    "smooth_lambda": s.smooth_lambda,
                },
                "trainable_modules": s.trainable_modules,
            }
            for s in stages
        ],
        "training_setup": {
            "checkpoint_initialization": str(init_ckpt) if init_ckpt else None,
            "initialization_details": init_info,
            "optimizer": "Adam (re-created at each stage transition over active trainable params)",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "total_epochs": total_epochs,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_stage": args.checkpoint_stage,
            "device": device,
            "seed": args.seed,
            "balanced_sampler": bool(args.balanced_sampler),
            "pos_weight": pos_weight,
            "stage_freeze_log": stage_freeze_log,
        },
        "inference_calibration_val_only": {
            "selected_settings": calibration["selected_settings"],
            "final_val_stats": calibration["final_val_stats"],
            "boundary_peak_rule": calibration["boundary_peak_rule"],
            "candidate_grids": {
                "threshold": threshold_candidates,
                "smooth_window": smooth_window_candidates,
                "min_event_len": min_event_len_candidates,
                "merge_gap": merge_gap_candidates,
                "boundary_radius": boundary_radius_candidates,
            },
            "tuning_log": calibration["tuning_log"],
            "note": "Validation split has no temporal boundary GT; calibration objective used binary F1 with event-density regularization to reduce overprediction.",
        },
        "training_curves": [
            {
                "epoch": h.epoch,
                "stage_id": h.stage_id,
                "stage_name": h.stage_name,
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
            "stage_id": best_ckpt.get("stage_id"),
            "stage_name": best_ckpt.get("stage_name"),
        },
        "temporal_localization": localization,
        "updated_metrics": {
            "binary": {
                "auc": binary_auc,
                "ap": binary_ap,
                "confusion_matrix_threshold": float(selected["threshold"]),
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
        "event_count_summary": {
            "gt_event_count": int(localization.get("gt_event_count", 0)),
            "pred_event_count": int(localization.get("pred_event_count", 0)),
        },
        "before_after_table": before_after_table,
        "qualitative_improved_videos": qualitative_improved,
        "one_failure_case": failure_case,
        "comparison_table": comparison_table,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "test_video_results.json").write_text(json.dumps(test_results, indent=2) + "\n")
    (output_dir / "pred_events_test.json").write_text(json.dumps(pred_events_all, indent=2) + "\n")
    (output_dir / "gt_events_test.json").write_text(json.dumps(gt_events, indent=2) + "\n")

    print("\nFinal Step-8 metrics")
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
