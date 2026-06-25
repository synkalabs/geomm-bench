#!/usr/bin/env python3
"""Plot the FORCE 2020 scaled-track figure from results/force_results.json.

Macro-F1 for the numeric reference and vision-CLIP, with 95% bootstrap
confidence intervals as error bars and the majority-class baseline macro-F1 as a
dashed line. Numbers are read straight from the results file, so the figure cannot drift.

Usage:
    python scripts/make_force_figure.py \
        --results results/force_results.json \
        --out images/Figure_force_results.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYOUT = [
    ("numeric_curves", "Numeric reference\n(curves)", "#2E9E5B"),
    ("vision_clip", "Vision-CLIP\n(rendered display)", "#E0524A"),
]


def make_figure(results_path, out_path):
    doc = json.load(open(results_path))
    res = doc["results"]
    labels = [lab for _, lab, _ in LAYOUT]
    colors = [c for _, _, c in LAYOUT]
    f1 = [res[k]["macro_f1"] for k, _, _ in LAYOUT]
    lo = [res[k]["macro_f1"] - res[k]["ci95"][0] for k, _, _ in LAYOUT]
    hi = [res[k]["ci95"][1] - res[k]["macro_f1"] for k, _, _ in LAYOUT]

    fig, ax = plt.subplots(figsize=(6.4, 5))
    xs = range(len(labels))
    ax.bar(xs, f1, color=colors, edgecolor="black", linewidth=1.2, width=0.6,
           yerr=[lo, hi], capsize=8, error_kw={"elinewidth": 1.4})
    for x, v, h in zip(xs, f1, hi):
        ax.text(x, v + h + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    maj = doc.get("baselines", {}).get("majority_class", {}).get("macro_f1", 0.25)
    ax.axhline(maj, ls="--", color="#999", lw=1.3)
    ax.text(len(labels) - 0.5, maj + 0.01, "majority-class baseline", ha="right",
            va="bottom", fontsize=9, color="#999")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro F1-Score", fontsize=12, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    n_w, n_i = doc.get("n_wells"), doc.get("n_intervals")
    ax.set_title(f"FORCE 2020 scaled track ({n_w} wells, {n_i} intervals), "
                 f"95% CI error bars", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")
    for (k, lab, _), v in zip(LAYOUT, f1):
        print(f"  {lab.replace(chr(10), ' '):30s} F1={v}  CI={res[k]['ci95']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/force_results.json")
    ap.add_argument("--out", default="images/Figure_force_results.png")
    args = ap.parse_args()
    make_figure(args.results, args.out)
