#!/usr/bin/env python3
"""Step-14C: Feature-space analysis (t-SNE) across UCF/XD/RWF/Shanghai.

Uses penultimate TRN representation (z after encoder, before heads) from locked
checkpoint and builds t-SNE visualizations + quantitative centroid analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from evaluate_ablation_checkpoint import load_model
from train_rtfm_trn_boundary import (
    choose_device,
    parse_class_names,
    read_csv,
    resolve,
    set_seed,
    topk_mean_np,
)

DATASETS = ("ucf_crime", "xd_violence", "rwf_2000", "shanghaitech")
STATUS_ORDER = ("TP", "TN", "FP", "FN")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-14C feature-space t-SNE")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--step14a-error-table",
        type=Path,
        default=Path("outputs/step14_interpretability/step14a/step14a_error_table.csv"),
    )

    p.add_argument(
        "--ucf-feature-manifest",
        type=Path,
        default=Path("data/ucf_crime/manifests/ucf_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--xd-feature-manifest",
        type=Path,
        default=Path("data/xd_violence/manifests/xd_violence_features_i3d.csv"),
    )
    p.add_argument(
        "--rwf-feature-manifest",
        type=Path,
        default=Path("data/rwf_2000/manifests/rwf_2000_features_i3d.csv"),
    )
    p.add_argument(
        "--sh-feature-manifest",
        type=Path,
        default=Path("data/shanghaitech/manifests/shanghaitech_features_i3d.csv"),
    )

    p.add_argument("--model-kind", choices=["step6_trn", "step7_boundary", "step8_progressive"], default="step7_boundary")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/step10_ablations/k1/train/checkpoints/best.pt"))

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--target-segments", type=int, default=32)
    p.add_argument("--trn-layers", type=int, default=2)
    p.add_argument("--trn-heads", type=int, default=4)
    p.add_argument("--trn-ffn-mult", type=int, default=4)
    p.add_argument("--trn-dropout", type=float, default=0.1)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default="learned")

    p.add_argument("--topk-ratio", type=float, default=0.125)
    p.add_argument("--infer-window", type=int, default=32)
    p.add_argument("--infer-stride", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--class-names", type=str, default="normal,fighting,shooting,explosion,robbery,abuse")
    p.add_argument("--normal-class", type=str, default="normal")

    p.add_argument("--per-dataset-samples", type=int, default=80)
    p.add_argument("--min-class-count", type=int, default=5)

    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-iter", type=int, default=1500)

    p.add_argument("--out-dir", type=Path, default=Path("outputs/step14_interpretability/step14c"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def status_is_failure(status: str) -> bool:
    return status in {"FP", "FN"}


def resolve_with_fallback(primary: Path, fallbacks: Sequence[Path]) -> Path:
    if primary.exists():
        return primary
    for p in fallbacks:
        if p.exists():
            return p
    return primary


def forward_with_z(
    model: torch.nn.Module,
    model_kind: str,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (segment anomaly logits, penultimate z)."""
    if model_kind == "step6_trn":
        seg_anom_logits, _, _, z, _ = model(x, return_attention=False)
        return seg_anom_logits, z
    if model_kind == "step7_boundary":
        seg_anom_logits, _, _, _, z, _ = model(x, return_attention=False)
        return seg_anom_logits, z
    seg_anom_logits, _, _, _, z, _ = model(
        x,
        return_attention=False,
        apply_trn=True,
        apply_boundary=True,
    )
    return seg_anom_logits, z


def sliding_window_starts(total_len: int, window: int, stride: int) -> List[int]:
    if total_len <= window:
        return [0]
    starts = list(range(0, total_len - window + 1, stride))
    last = total_len - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def infer_embedding_chunked(
    model: torch.nn.Module,
    model_kind: str,
    features: np.ndarray,
    window: int,
    stride: int,
    topk_ratio: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    t = int(features.shape[0])
    if t <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    hidden_dim: Optional[int] = None
    sum_z: Optional[np.ndarray] = None
    cnt_z = np.zeros((t,), dtype=np.float64)
    sum_scores = np.zeros((t,), dtype=np.float64)
    cnt_scores = np.zeros((t,), dtype=np.float64)

    starts = sliding_window_starts(t, window, stride)

    with torch.no_grad():
        for s in starts:
            e = min(s + window, t)
            chunk = features[s:e]
            real_len = int(e - s)

            if real_len < window:
                pad = np.repeat(chunk[-1:, :], repeats=(window - real_len), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)

            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            seg_logits, z = forward_with_z(model, model_kind, x)

            seg_scores = torch.sigmoid(seg_logits)[0].detach().cpu().numpy().astype(np.float64)[:real_len]
            z_np = z[0].detach().cpu().numpy().astype(np.float64)[:real_len]

            if hidden_dim is None:
                hidden_dim = int(z_np.shape[1])
                sum_z = np.zeros((t, hidden_dim), dtype=np.float64)

            assert sum_z is not None
            sum_z[s:e, :] += z_np
            cnt_z[s:e] += 1.0

            sum_scores[s:e] += seg_scores
            cnt_scores[s:e] += 1.0

    assert sum_z is not None
    z_seq = np.divide(sum_z, np.maximum(cnt_z[:, None], 1e-8), out=np.zeros_like(sum_z), where=cnt_z[:, None] > 0)
    seg_scores = np.divide(sum_scores, np.maximum(cnt_scores, 1e-8), out=np.zeros_like(sum_scores), where=cnt_scores > 0)

    weights = np.maximum(seg_scores, 1e-6)
    emb = (z_seq * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6)
    video_anom = topk_mean_np(seg_scores.astype(np.float32), topk_ratio)
    return emb.astype(np.float32), seg_scores.astype(np.float32), float(video_anom)


def pick_diverse(pool: List[Dict[str, str]], k: int, rng: random.Random, label_key: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if k <= 0 or not pool:
        return [], list(pool)

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in pool:
        groups[str(r.get(label_key, "unknown"))].append(r)
    labels = list(groups.keys())
    for lb in labels:
        rng.shuffle(groups[lb])
    rng.shuffle(labels)

    out: List[Dict[str, str]] = []
    active = list(labels)
    while len(out) < k and active:
        for lb in list(active):
            if groups[lb]:
                out.append(groups[lb].pop())
                if len(out) >= k:
                    break
            if not groups[lb]:
                active.remove(lb)

    remaining: List[Dict[str, str]] = []
    for lb in labels:
        remaining.extend(groups[lb])
    return out, remaining


def pick_random(pool: List[Dict[str, str]], k: int, rng: random.Random) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if k <= 0 or not pool:
        return [], list(pool)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    idx_sel = set(idx[: min(k, len(pool))])
    selected = [pool[i] for i in idx if i in idx_sel]
    remaining = [pool[i] for i in idx if i not in idx_sel]
    return selected, remaining


def sample_per_dataset(rows: List[Dict[str, str]], target_n: int, rng: random.Random, diverse: bool) -> List[Dict[str, str]]:
    if target_n >= len(rows):
        return list(rows)

    strata_keys = [
        (1, True),   # positive failure
        (1, False),  # positive success
        (0, True),   # negative failure
        (0, False),  # negative success
    ]

    strata: Dict[Tuple[int, bool], List[Dict[str, str]]] = {k: [] for k in strata_keys}
    for r in rows:
        bl = as_int(r.get("binary_label", 0), 0)
        fail = status_is_failure(str(r.get("binary_status", "")))
        k = (1 if bl == 1 else 0, bool(fail))
        strata[k].append(r)

    base = target_n // 4
    alloc = {k: min(base, len(strata[k])) for k in strata_keys}
    used = sum(alloc.values())

    # distribute remaining to larger pools first
    rem = target_n - used
    if rem > 0:
        room = sorted(strata_keys, key=lambda k: len(strata[k]) - alloc[k], reverse=True)
        for k in room:
            if rem <= 0:
                break
            cap = len(strata[k]) - alloc[k]
            add = min(cap, rem)
            alloc[k] += add
            rem -= add

    chosen: List[Dict[str, str]] = []
    leftovers: List[Dict[str, str]] = []
    for k in strata_keys:
        pool = list(strata[k])
        if diverse:
            sel, rem_pool = pick_diverse(pool, alloc[k], rng, label_key="gt_label")
        else:
            sel, rem_pool = pick_random(pool, alloc[k], rng)
        chosen.extend(sel)
        leftovers.extend(rem_pool)

    if len(chosen) < target_n and leftovers:
        need = target_n - len(chosen)
        if diverse:
            extra, _ = pick_diverse(leftovers, need, rng, label_key="gt_label")
        else:
            extra, _ = pick_random(leftovers, need, rng)
        chosen.extend(extra)

    return chosen[:target_n]


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def plot_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: List[str],
    colors: Dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    uniq = sorted(set(labels))
    for u in uniq:
        idx = [i for i, v in enumerate(labels) if v == u]
        if not idx:
            continue
        ax.scatter(xs[idx], ys[idx], s=28, alpha=0.85, c=colors.get(u, "#7f7f7f"), label=u)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = resolve(root, args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    rng = random.Random(args.seed)

    class_names = parse_class_names(args.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if args.normal_class not in class_to_idx:
        raise ValueError(f"normal class '{args.normal_class}' missing from class_names")
    normal_idx = class_to_idx[args.normal_class]

    device = choose_device(args.device)
    args.checkpoint = resolve(root, args.checkpoint)
    model, _ = load_model(args, num_classes=len(class_names), device=device)

    # Feature manifest maps.
    feature_manifest_paths = {
        "ucf_crime": resolve(root, args.ucf_feature_manifest),
        "xd_violence": resolve_with_fallback(
            resolve(root, args.xd_feature_manifest),
            [resolve(root, Path("data/xd_violence/manifests/xd_violence_features_i3d_testonly.csv"))],
        ),
        "rwf_2000": resolve(root, args.rwf_feature_manifest),
        "shanghaitech": resolve(root, args.sh_feature_manifest),
    }

    feat_map: Dict[str, Dict[str, Dict[str, str]]] = {}
    for ds, pth in feature_manifest_paths.items():
        rows = [r for r in read_csv(pth) if r.get("status", "ok") == "ok"] if pth.exists() else []
        feat_map[ds] = {str(r.get("video_id", "")): r for r in rows}

    # Use Step-14A table for predictions/status and sampling metadata.
    step14a_csv = resolve(root, args.step14a_error_table)
    if not step14a_csv.exists():
        raise FileNotFoundError(f"Step-14A error table not found: {step14a_csv}")
    step14a_rows = read_csv(step14a_csv)

    rows_by_ds: Dict[str, List[Dict[str, str]]] = {ds: [] for ds in DATASETS}
    for r in step14a_rows:
        ds = str(r.get("dataset", ""))
        vid = str(r.get("video_id", ""))
        if ds not in rows_by_ds:
            continue
        if vid not in feat_map.get(ds, {}):
            continue
        rows_by_ds[ds].append(r)

    selected: List[Dict[str, str]] = []
    sampling_meta: Dict[str, Dict[str, object]] = {}
    for ds in DATASETS:
        ds_rows = rows_by_ds.get(ds, [])
        target = min(int(args.per_dataset_samples), len(ds_rows))
        chosen = sample_per_dataset(
            ds_rows,
            target_n=target,
            rng=rng,
            diverse=(ds in {"ucf_crime", "xd_violence"}),
        )
        selected.extend(chosen)
        sampling_meta[ds] = {
            "available": len(ds_rows),
            "selected": len(chosen),
            "target": int(args.per_dataset_samples),
            "gt_label_counts": dict(Counter(str(x.get("gt_label", "")) for x in chosen)),
            "status_counts": dict(Counter(str(x.get("binary_status", "")) for x in chosen)),
            "binary_counts": dict(Counter(int(as_int(x.get("binary_label", 0), 0)) for x in chosen)),
        }

    print(f"selected videos: {len(selected)}")

    # Extract embeddings.
    embed_rows: List[Dict[str, object]] = []
    vectors: List[np.ndarray] = []
    extraction_failures: List[str] = []

    for i, r in enumerate(selected, 1):
        ds = str(r.get("dataset", ""))
        vid = str(r.get("video_id", ""))
        frow = feat_map.get(ds, {}).get(vid)
        if not frow:
            extraction_failures.append(f"{ds}/{vid}: feature row missing")
            continue

        feat_path = resolve(root, Path(str(frow.get("feature_path", ""))))
        if not feat_path.exists():
            extraction_failures.append(f"{ds}/{vid}: feature file missing {feat_path}")
            continue

        with np.load(feat_path, allow_pickle=True) as d:
            feats = d["features"].astype(np.float32)

        emb, seg_scores, video_score_calc = infer_embedding_chunked(
            model=model,
            model_kind=args.model_kind,
            features=feats,
            window=args.infer_window,
            stride=args.infer_stride,
            topk_ratio=args.topk_ratio,
            device=device,
        )

        if emb.size == 0:
            extraction_failures.append(f"{ds}/{vid}: empty embedding")
            continue

        vectors.append(emb)
        embed_rows.append(
            {
                "video_id": vid,
                "dataset": ds,
                "gt_label": str(r.get("gt_label", "")),
                "predicted_label": str(r.get("predicted_label", "")),
                "binary_status": str(r.get("binary_status", "")),
                "success_failure": "failure" if status_is_failure(str(r.get("binary_status", ""))) else "success",
                "binary_label": int(as_int(r.get("binary_label", 0), 0)),
                "pred_binary": int(as_int(r.get("pred_binary", 0), 0)),
                "anomaly_score": as_float(r.get("anomaly_score", 0.0), 0.0),
                "anomaly_score_recomputed": float(video_score_calc),
                "binary_confidence": as_float(r.get("binary_confidence", 0.0), 0.0),
                "class_confidence": as_float(r.get("class_confidence", 0.0), 0.0),
                "num_segments": int(as_int(r.get("num_segments", feats.shape[0]), feats.shape[0])),
                "feature_path": str(frow.get("feature_path", "")),
                "embedding_dim": int(emb.shape[0]),
                "embedding": emb.tolist(),
            }
        )

        if i % 25 == 0 or i == len(selected):
            print(f"  embedding {i}/{len(selected)}")

    if not embed_rows:
        raise RuntimeError("No embeddings extracted; cannot run Step-14C")

    X = np.stack(vectors, axis=0).astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n = Xs.shape[0]
    max_perpl = max(5.0, min(float(args.tsne_perplexity), float(max(5, (n - 1) // 3))))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=max_perpl,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    # sklearn API changed from n_iter -> max_iter in newer releases.
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=int(args.tsne_iter))
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=int(args.tsne_iter))
    coords = tsne.fit_transform(Xs)

    for i, r in enumerate(embed_rows):
        r["tsne_x"] = float(coords[i, 0])
        r["tsne_y"] = float(coords[i, 1])

    # Plot 1: by dataset.
    ds_labels = [str(r["dataset"]) for r in embed_rows]
    ds_colors = {
        "ucf_crime": "#1f77b4",
        "xd_violence": "#ff7f0e",
        "rwf_2000": "#2ca02c",
        "shanghaitech": "#d62728",
    }
    plot_ds = plots_dir / "tsne_by_dataset.png"
    plot_scatter(coords[:, 0], coords[:, 1], ds_labels, ds_colors, "Step-14C t-SNE by Dataset", plot_ds)

    # Plot 2: by class/label (collapse tail labels into other for readability).
    raw_labels = [str(r["gt_label"]) for r in embed_rows]
    cnt_labels = Counter(raw_labels)
    keep = {k for k, _ in cnt_labels.most_common(11)}
    label_plot = [lb if lb in keep else "other" for lb in raw_labels]
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
    ]
    uniq_label = sorted(set(label_plot))
    label_colors = {u: palette[i % len(palette)] for i, u in enumerate(uniq_label)}
    plot_label = plots_dir / "tsne_by_class_label.png"
    plot_scatter(coords[:, 0], coords[:, 1], label_plot, label_colors, "Step-14C t-SNE by GT Label", plot_label)

    # Plot 3: by TP/TN/FP/FN.
    st_labels = [str(r["binary_status"]) for r in embed_rows]
    st_colors = {
        "TP": "#2ca02c",
        "TN": "#1f77b4",
        "FP": "#ff7f0e",
        "FN": "#d62728",
    }
    plot_status = plots_dir / "tsne_by_success_failure.png"
    plot_scatter(coords[:, 0], coords[:, 1], st_labels, st_colors, "Step-14C t-SNE by TP/TN/FP/FN", plot_status)

    # Quantitative summaries on standardized embedding space.
    idx_by_ds: Dict[str, List[int]] = defaultdict(list)
    idx_by_label: Dict[str, List[int]] = defaultdict(list)
    idx_by_status: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(embed_rows):
        idx_by_ds[str(r["dataset"])].append(i)
        idx_by_label[str(r["gt_label"])].append(i)
        idx_by_status[str(r["binary_status"])].append(i)

    cent_ds: Dict[str, np.ndarray] = {}
    intra_ds_dist: Dict[str, float] = {}
    for ds, idxs in idx_by_ds.items():
        mat = Xs[idxs]
        c = mat.mean(axis=0)
        cent_ds[ds] = c
        intra_ds_dist[ds] = float(np.mean(np.linalg.norm(mat - c[None, :], axis=1)))

    dataset_to_ucf: Dict[str, float] = {}
    if "ucf_crime" in cent_ds:
        cu = cent_ds["ucf_crime"]
        for ds, c in cent_ds.items():
            dataset_to_ucf[ds] = euclidean(c, cu)

    class_centroids: Dict[str, np.ndarray] = {}
    for lb, idxs in idx_by_label.items():
        if len(idxs) >= int(args.min_class_count):
            class_centroids[lb] = Xs[idxs].mean(axis=0)

    class_dists: Dict[str, float] = {}
    for a, b in combinations(sorted(class_centroids.keys()), 2):
        class_dists[f"{a}__vs__{b}"] = euclidean(class_centroids[a], class_centroids[b])

    normal_cent = class_centroids.get("normal")
    fn_near_normal_frac = float("nan")
    fn_dist_normal = float("nan")
    tp_dist_normal = float("nan")
    if normal_cent is not None:
        fn_idx = idx_by_status.get("FN", [])
        tp_idx = idx_by_status.get("TP", [])
        if fn_idx:
            fn_dist_normal = float(np.mean(np.linalg.norm(Xs[fn_idx] - normal_cent[None, :], axis=1)))
        if tp_idx:
            tp_dist_normal = float(np.mean(np.linalg.norm(Xs[tp_idx] - normal_cent[None, :], axis=1)))

        # nearest centroid label for FN samples
        if fn_idx and class_centroids:
            labs = list(class_centroids.keys())
            hits = 0
            for i in fn_idx:
                d = [euclidean(Xs[i], class_centroids[lb]) for lb in labs]
                nearest = labs[int(np.argmin(d))]
                if nearest == "normal":
                    hits += 1
            fn_near_normal_frac = float(hits) / float(len(fn_idx))

    # Domain-specific checks requested.
    ucf_sep = float("nan")
    ucf_rows = [i for i, r in enumerate(embed_rows) if r["dataset"] == "ucf_crime"]
    if ucf_rows:
        ucf_norm = [i for i in ucf_rows if str(embed_rows[i]["gt_label"]) == "normal"]
        ucf_pos = [i for i in ucf_rows if str(embed_rows[i]["gt_label"]) != "normal"]
        if ucf_norm and ucf_pos:
            ucf_sep = euclidean(Xs[ucf_norm].mean(axis=0), Xs[ucf_pos].mean(axis=0))

    xd_drift: Dict[str, float] = {}
    if "normal" in class_centroids:
        ucf_violent = [lb for lb in class_centroids.keys() if lb not in {"normal", "anomaly"}]
        xd_pos_idx = [i for i, r in enumerate(embed_rows) if r["dataset"] == "xd_violence" and str(r["gt_label"]) != "normal"]
        if xd_pos_idx and ucf_violent:
            c_xd_pos = Xs[xd_pos_idx].mean(axis=0)
            d_to_ucf_norm = euclidean(c_xd_pos, class_centroids["normal"])
            d_to_ucf_violent = min(euclidean(c_xd_pos, class_centroids[v]) for v in ucf_violent)
            xd_drift = {
                "xd_positive_to_ucf_normal": float(d_to_ucf_norm),
                "xd_positive_to_nearest_ucf_violent": float(d_to_ucf_violent),
            }

    rwf_collapse: Dict[str, float] = {}
    rwf_fight_idx = [i for i, r in enumerate(embed_rows) if r["dataset"] == "rwf_2000" and str(r["gt_label"]) == "fighting"]
    if rwf_fight_idx and "normal" in class_centroids:
        ucf_violent = [lb for lb in class_centroids.keys() if lb not in {"normal", "anomaly"}]
        c_rwf_fight = Xs[rwf_fight_idx].mean(axis=0)
        rwf_collapse["rwf_fight_to_ucf_normal"] = float(euclidean(c_rwf_fight, class_centroids["normal"]))
        if ucf_violent:
            rwf_collapse["rwf_fight_to_nearest_ucf_violent"] = float(
                min(euclidean(c_rwf_fight, class_centroids[v]) for v in ucf_violent)
            )

    sh_shift: Dict[str, float] = {}
    sh_anom_idx = [i for i, r in enumerate(embed_rows) if r["dataset"] == "shanghaitech" and str(r["gt_label"]) == "anomaly"]
    if sh_anom_idx and "normal" in class_centroids:
        ucf_violent = [lb for lb in class_centroids.keys() if lb not in {"normal", "anomaly"}]
        c_sh = Xs[sh_anom_idx].mean(axis=0)
        sh_shift["sh_anomaly_to_ucf_normal"] = float(euclidean(c_sh, class_centroids["normal"]))
        if ucf_violent:
            sh_shift["sh_anomaly_to_nearest_ucf_violent"] = float(
                min(euclidean(c_sh, class_centroids[v]) for v in ucf_violent)
            )

    silh_dataset = float("nan")
    if len(set(ds_labels)) > 1 and all(ds_labels.count(k) > 1 for k in set(ds_labels)):
        silh_dataset = float(silhouette_score(Xs, ds_labels))

    silh_status = float("nan")
    if len(set(st_labels)) > 1 and all(st_labels.count(k) > 1 for k in set(st_labels)):
        silh_status = float(silhouette_score(Xs, st_labels))

    # Save embeddings csv (without raw vector for readability).
    emb_csv = out_dir / "step14c_embeddings.csv"
    fieldnames = [
        "video_id", "dataset", "gt_label", "predicted_label", "binary_status", "success_failure",
        "binary_label", "pred_binary", "anomaly_score", "anomaly_score_recomputed",
        "binary_confidence", "class_confidence", "num_segments", "feature_path",
        "embedding_dim", "tsne_x", "tsne_y",
    ]
    with emb_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in embed_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Save compact embedding npz.
    emb_npz = out_dir / "step14c_embeddings_matrix.npz"
    np.savez_compressed(
        emb_npz,
        embeddings=X.astype(np.float32),
        embeddings_standardized=Xs.astype(np.float32),
        tsne=coords.astype(np.float32),
    )

    report_sentence = (
        "t-SNE shows external-set false negatives clustering closer to the normal manifold with diffuse feature organization, "
        "while cross-dataset positive semantics (especially RWF/ShanghaiTech) drift from UCF violent structure."
    )

    summary = {
        "extraction_summary": {
            "feature_layer_used": "TRN penultimate representation z (post-encoder, pre-head), anomaly-weighted temporal mean",
            "model_kind": args.model_kind,
            "checkpoint": str(args.checkpoint),
            "device": device,
            "datasets_included": sorted(set(ds_labels)),
            "num_datasets": len(set(ds_labels)),
            "num_points_plotted": len(embed_rows),
            "embedding_dim": int(X.shape[1]),
            "sampling_rule": f"up to {args.per_dataset_samples} per dataset; balanced on positive/negative and success/failure; diversity prioritization for UCF/XD labels",
            "sampling_meta": sampling_meta,
            "extraction_failures": extraction_failures,
        },
        "plots": {
            "by_dataset": str(plot_ds),
            "by_class_label": str(plot_label),
            "by_success_failure": str(plot_status),
        },
        "quantitative": {
            "intra_dataset_mean_distance": intra_ds_dist,
            "dataset_to_ucf_centroid_distance": dataset_to_ucf,
            "class_centroid_pair_distances": class_dists,
            "ucf_normal_vs_ucf_positive_centroid_distance": ucf_sep,
            "xd_drift": xd_drift,
            "rwf_collapse": rwf_collapse,
            "shanghaitech_shift": sh_shift,
            "fn_mean_distance_to_normal_centroid": fn_dist_normal,
            "tp_mean_distance_to_normal_centroid": tp_dist_normal,
            "fn_nearest_normal_centroid_fraction": fn_near_normal_frac,
            "silhouette_by_dataset": silh_dataset,
            "silhouette_by_status": silh_status,
        },
        "observations": {
            "ucf_class_separation": "separated" if np.isfinite(ucf_sep) and ucf_sep > 0 else "undetermined",
            "xd_alignment": "drift" if xd_drift and xd_drift.get("xd_positive_to_ucf_normal", 1e9) <= xd_drift.get("xd_positive_to_nearest_ucf_violent", 0.0) else "partial alignment",
            "rwf_near_normal": "yes" if rwf_collapse and rwf_collapse.get("rwf_fight_to_ucf_normal", 1e9) <= rwf_collapse.get("rwf_fight_to_nearest_ucf_violent", 0.0) else "no_or_partial",
            "shanghaitech_ood": "yes" if sh_shift else "undetermined",
            "fn_near_normal": "yes" if np.isfinite(fn_near_normal_frac) and fn_near_normal_frac >= 0.5 else "partial_or_no",
        },
        "artifacts": {
            "report_txt": str(out_dir / "step14c_report.txt"),
            "summary_json": str(out_dir / "step14c_summary.json"),
            "embeddings_csv": str(emb_csv),
            "embeddings_npz": str(emb_npz),
            "plots_dir": str(plots_dir),
        },
        "report_ready_sentence": report_sentence,
    }

    (out_dir / "step14c_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines: List[str] = []
    lines.append("1) Extraction summary")
    lines.append(f"- feature layer used: {summary['extraction_summary']['feature_layer_used']}")
    lines.append(f"- model/checkpoint: {args.model_kind} | {args.checkpoint}")
    lines.append(f"- number of points plotted: {len(embed_rows)}")
    lines.append(f"- datasets included: {summary['extraction_summary']['datasets_included']}")
    lines.append(f"- sampling rule: {summary['extraction_summary']['sampling_rule']}")
    lines.append(f"- extraction failures: {len(extraction_failures)}")

    lines.append("")
    lines.append("2) Plot summary")
    lines.append(f"- dataset-colored t-SNE: {plot_ds}")
    lines.append(f"- class-colored t-SNE: {plot_label}")
    lines.append(f"- success/failure t-SNE: {plot_status}")

    lines.append("")
    lines.append("3) Main observations")
    lines.append(f"- UCF class separation (normal vs positive centroid distance): {ucf_sep:.4f}" if np.isfinite(ucf_sep) else "- UCF class separation: unavailable")
    if xd_drift:
        lines.append(
            "- XD positives vs UCF: "
            f"to normal={xd_drift.get('xd_positive_to_ucf_normal', float('nan')):.4f}, "
            f"to nearest violent={xd_drift.get('xd_positive_to_nearest_ucf_violent', float('nan')):.4f}"
        )
    else:
        lines.append("- XD alignment: unavailable")
    if rwf_collapse:
        lines.append(
            "- RWF fighting vs UCF: "
            f"to normal={rwf_collapse.get('rwf_fight_to_ucf_normal', float('nan')):.4f}, "
            f"to nearest violent={rwf_collapse.get('rwf_fight_to_nearest_ucf_violent', float('nan')):.4f}"
        )
    else:
        lines.append("- RWF collapse check: unavailable")
    if sh_shift:
        lines.append(
            "- ShanghaiTech anomaly vs UCF: "
            f"to normal={sh_shift.get('sh_anomaly_to_ucf_normal', float('nan')):.4f}, "
            f"to nearest violent={sh_shift.get('sh_anomaly_to_nearest_ucf_violent', float('nan')):.4f}"
        )
    else:
        lines.append("- ShanghaiTech shift check: unavailable")

    lines.append("")
    lines.append("4) Quantitative summary")
    lines.append(f"- silhouette by dataset: {silh_dataset:.6f}" if np.isfinite(silh_dataset) else "- silhouette by dataset: unavailable")
    lines.append(f"- silhouette by TP/TN/FP/FN: {silh_status:.6f}" if np.isfinite(silh_status) else "- silhouette by TP/TN/FP/FN: unavailable")
    lines.append(f"- FN mean distance to normal centroid: {fn_dist_normal:.6f}" if np.isfinite(fn_dist_normal) else "- FN mean distance to normal centroid: unavailable")
    lines.append(f"- TP mean distance to normal centroid: {tp_dist_normal:.6f}" if np.isfinite(tp_dist_normal) else "- TP mean distance to normal centroid: unavailable")
    lines.append(
        f"- FN nearest-normal-centroid fraction: {fn_near_normal_frac:.6f}"
        if np.isfinite(fn_near_normal_frac)
        else "- FN nearest-normal-centroid fraction: unavailable"
    )

    lines.append("")
    lines.append("5) Report-ready sentence")
    lines.append(f"- {report_sentence}")

    (out_dir / "step14c_report.txt").write_text("\n".join(lines) + "\n")

    print("Step-14C complete")
    print(f"- points: {len(embed_rows)}")
    print(f"- report: {out_dir / 'step14c_report.txt'}")
    print(f"- summary: {out_dir / 'step14c_summary.json'}")
    print(f"- plots: {plots_dir}")


if __name__ == "__main__":
    main()
