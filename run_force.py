#!/usr/bin/env python3
"""GeoMM-Bench scaled track runner (FORCE 2020).

Compares the off-the-shelf CLIP image encoder on rendered displays against a
numeric reference classifier on the curve statistics, with well-level
cross-validation and cluster bootstrap confidence intervals. Writes one results
file. The FORCE 2020 CSV is public open data (not redistributed in this repo);
point --csv at a local copy.

Usage:
    python run_force.py --csv data/force2020/train.csv --out results/force_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geomm_bench import baselines as B
from geomm_bench import force as F
from geomm_bench import metrics as M
from geomm_bench import stats as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/force2020/train.csv")
    ap.add_argument("--out", default="results/force_results.json")
    ap.add_argument("--min-coverage", type=float, default=0.7)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    intervals, wells = F.load_intervals(
        args.csv, min_coverage=args.min_coverage, max_wells=args.max_wells)
    y = [iv["lithology"] for iv in intervals]
    groups = [iv["well"] for iv in intervals]
    X = np.array([F.features(iv["window"]) for iv in intervals])
    print(f"{len(wells)} wells, {len(y)} intervals, dist {dict(Counter(y))}", flush=True)

    # Numeric reference: logistic regression on curve stats, well-level CV.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    num_pred = cross_val_predict(clf, X, y, groups=groups,
                                 cv=GroupKFold(n_splits=args.cv_folds))

    # Vision-only CLIP on the rendered display.
    vis_pred = [B.classify_vision_clip(F.render(iv["window"]))["predicted"] for iv in intervals]

    doc = {"benchmark": "GeoMM-Bench", "track": "FORCE 2020 (scaled, rendered displays)",
           "n_wells": len(wells), "n_intervals": len(y),
           "class_distribution": {k: int(v) for k, v in Counter(y).items()},
           "metric": "macro_f1_present_classes", "results": {}}
    for name, pred in [("numeric_curves", list(num_pred)), ("vision_clip", vis_pred)]:
        lo, hi = S.cluster_bootstrap_macro_f1(y, pred, groups)
        _, _, pc = M.per_class_prf(y, pred)
        doc["results"][name] = {
            "macro_f1": round(M.macro_f1(y, pred), 3), "accuracy": round(M.accuracy(y, pred), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "per_class_f1": {k: round(v, 3) for k, v in pc.items()},
            "pred_distribution": {k: int(v) for k, v in Counter(pred).items()}}
    doc["baselines"] = {"random_four_class_f1": 0.25}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {args.out}", flush=True)
    for name, r in doc["results"].items():
        print(f"  {name:16s} F1={r['macro_f1']} {tuple(r['ci95'])}  acc={r['accuracy']}", flush=True)


if __name__ == "__main__":
    main()
