#!/usr/bin/env python3
"""Build and print Step-10 master ablation table/report."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationRow:
    model: str
    trn: str
    boundary: str
    k: str
    training_strategy: str
    auc: Optional[float]
    ap: Optional[float]
    macro_f1: Optional[float]
    weighted_f1: Optional[float]
    map03: Optional[float]
    map05: Optional[float]
    map07: Optional[float]
    pred_events: Optional[int]
    source: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Step-10 ablation report")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--output-dir", type=Path, default=Path("outputs/step10_ablations"))

    p.add_argument("--step5-json", type=Path, default=Path("outputs/step10_ablations/eval_step5_fixed.json"))
    p.add_argument("--step6-json", type=Path, default=Path("outputs/step10_ablations/eval_step6_fixed.json"))
    p.add_argument("--step7-on-json", type=Path, default=Path("outputs/step10_ablations/eval_step7_boundary_on_fixed.json"))
    p.add_argument("--step7-off-json", type=Path, default=Path("outputs/step10_ablations/eval_step7_boundary_off_fixed.json"))
    p.add_argument("--step8-json", type=Path, default=Path("outputs/step10_ablations/eval_step8_progressive_fixed.json"))

    p.add_argument("--k-values", type=str, default="1,3,5,10")
    p.add_argument("--k-json-template", type=str, default="outputs/step10_ablations/k{K}/eval_fixed.json")
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def parse_k_values(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        s = token.strip()
        if s:
            out.append(int(s))
    return out


def as_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


def as_int(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def load_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def row_from_summary(
    label: str,
    trn: str,
    boundary: str,
    k: str,
    training_strategy: str,
    path: Path,
) -> AblationRow:
    data = load_summary(path)
    if data is None:
        return AblationRow(
            model=label,
            trn=trn,
            boundary=boundary,
            k=k,
            training_strategy=training_strategy,
            auc=None,
            ap=None,
            macro_f1=None,
            weighted_f1=None,
            map03=None,
            map05=None,
            map07=None,
            pred_events=None,
            source=str(path),
        )

    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    binary = metrics.get("binary", {}) if isinstance(metrics, dict) else {}
    cls = metrics.get("classification", {}) if isinstance(metrics, dict) else {}

    return AblationRow(
        model=label,
        trn=trn,
        boundary=boundary,
        k=k,
        training_strategy=training_strategy,
        auc=as_float(binary.get("auc")),
        ap=as_float(binary.get("ap")),
        macro_f1=as_float(cls.get("macro_f1")),
        weighted_f1=as_float(cls.get("weighted_f1")),
        map03=as_float(data.get("mAP@0.3")),
        map05=as_float(data.get("mAP@0.5")),
        map07=as_float(data.get("mAP@0.7")),
        pred_events=as_int(data.get("pred_event_count")),
        source=str(path),
    )


def metric_key(v: Optional[float]) -> float:
    if v is None:
        return -1e18
    if math.isnan(v):
        return -1e18
    return float(v)


def fmt(v: Optional[float], nd: int = 6) -> str:
    if v is None:
        return "NA"
    if math.isnan(v):
        return "NA"
    return f"{v:.{nd}f}"


def find_best(rows: List[AblationRow], key_fn) -> Optional[AblationRow]:
    def primary_key_value(row: AblationRow) -> float:
        key = key_fn(row)
        if isinstance(key, tuple):
            if not key:
                return -1e18
            first = key[0]
            try:
                return float(first)
            except Exception:
                return -1e18
        try:
            return float(key)
        except Exception:
            return -1e18

    valid = [r for r in rows if primary_key_value(r) > -1e18]
    if not valid:
        return None
    return max(valid, key=key_fn)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = resolve(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[AblationRow] = []

    step5 = row_from_summary(
        label="Step5 RTFM+Cls (no TRN/bnd)",
        trn="off",
        boundary="off",
        k="4",
        training_strategy="joint",
        path=resolve(root, args.step5_json),
    )
    step6 = row_from_summary(
        label="Step6 +TRN",
        trn="on",
        boundary="off",
        k="4",
        training_strategy="joint",
        path=resolve(root, args.step6_json),
    )
    step7_on = row_from_summary(
        label="Step7 +TRN+bnd (refine on)",
        trn="on",
        boundary="on",
        k="4",
        training_strategy="joint",
        path=resolve(root, args.step7_on_json),
    )
    step7_off = row_from_summary(
        label="Step7 +TRN+bnd (refine off)",
        trn="on",
        boundary="off (decode)",
        k="4",
        training_strategy="joint",
        path=resolve(root, args.step7_off_json),
    )
    step8 = row_from_summary(
        label="Step8 progressive",
        trn="on",
        boundary="on",
        k="4",
        training_strategy="progressive",
        path=resolve(root, args.step8_json),
    )

    rows.extend([step5, step6, step7_on, step7_off, step8])

    for k in parse_k_values(args.k_values):
        k_path = resolve(root, Path(args.k_json_template.replace("{K}", str(k))))
        rows.append(
            row_from_summary(
                label=f"k-sweep full model (k={k})",
                trn="on",
                boundary="on",
                k=str(k),
                training_strategy="joint",
                path=k_path,
            )
        )

    overall_best = find_best(
        rows,
        key_fn=lambda r: (metric_key(r.map05), metric_key(r.macro_f1), metric_key(r.ap)),
    )
    cls_best = find_best(rows, key_fn=lambda r: (metric_key(r.macro_f1), metric_key(r.weighted_f1)))
    loc_best = find_best(rows, key_fn=lambda r: (metric_key(r.map05), metric_key(r.map03), -abs((r.pred_events or 10**9) - 59)))

    report = {
        "rows": [r.__dict__ for r in rows],
        "best_model_overall": overall_best.__dict__ if overall_best else None,
        "best_model_classification": cls_best.__dict__ if cls_best else None,
        "best_model_localization": loc_best.__dict__ if loc_best else None,
    }
    (out_dir / "step10_master_ablation.json").write_text(json.dumps(report, indent=2) + "\n")

    print("1) Master ablation table")
    print(
        "Model | TRN | Boundary | k | Training strategy | AUC | AP | Macro-F1 | Weighted-F1 | "
        "mAP@0.3 | mAP@0.5 | mAP@0.7 | Pred events"
    )
    for r in rows:
        print(
            f"{r.model} | {r.trn} | {r.boundary} | {r.k} | {r.training_strategy} | "
            f"{fmt(r.auc)} | {fmt(r.ap)} | {fmt(r.macro_f1)} | {fmt(r.weighted_f1)} | "
            f"{fmt(r.map03)} | {fmt(r.map05)} | {fmt(r.map07)} | "
            f"{r.pred_events if r.pred_events is not None else 'NA'}"
        )

    print("\n2) Best model after ablations")
    if overall_best:
        print(f"- overall winner: {overall_best.model}")
    else:
        print("- overall winner: NA")
    if cls_best:
        print(f"- classification winner (macro-F1): {cls_best.model}")
    else:
        print("- classification winner (macro-F1): NA")
    if loc_best:
        print(f"- localization winner (mAP@0.5): {loc_best.model}")
    else:
        print("- localization winner (mAP@0.5): NA")

    print("\n3) One-paragraph interpretation")
    bits: List[str] = []

    if step5.macro_f1 is not None and step6.macro_f1 is not None:
        diff = step6.macro_f1 - step5.macro_f1
        bits.append(f"TRN effect (Step6-Step5) on macro-F1 is {diff:+.4f}")

    if step6.map05 is not None and step7_on.map05 is not None:
        diff = step7_on.map05 - step6.map05
        bits.append(f"boundary head effect (Step7-Step6) on mAP@0.5 is {diff:+.4f}")

    if step7_on.map05 is not None and step8.map05 is not None:
        diff = step8.map05 - step7_on.map05
        bits.append(f"progressive-vs-joint gap (Step8-Step7) on mAP@0.5 is {diff:+.4f}")

    k_rows = [r for r in rows if r.model.startswith("k-sweep") and r.map05 is not None]
    if k_rows:
        k_best = max(k_rows, key=lambda r: (metric_key(r.map05), metric_key(r.macro_f1)))
        bits.append(f"best pseudo-label budget among tested k is {k_best.k} by localization priority")

    if not bits:
        bits.append("Some ablation files are missing; generate all eval JSONs first, then rerun this report for final conclusions")

    print("- " + "; ".join(bits) + ".")


if __name__ == "__main__":
    main()
