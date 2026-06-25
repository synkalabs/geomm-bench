#!/usr/bin/env python3
"""Plot the GeoMM-Bench results figure from the results JSON.

Reads the numbers straight from the file written by run_geomm_bench.py:

    {"results": {"text_only": {"macro_f1": ..., "accuracy": ...}, ...}}

Approaches with no recorded value (e.g. before the source PDFs have been run) are
drawn as "n/a" rather than filled with a placeholder.

Usage:
    python scripts/make_results_figure.py \
        --results results/geomm_bench_results.json \
        --out images/Figure2_Results_Comparison.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"text": "#2E9E5B", "vision": "#E0524A", "multimodal": "#E8943A"}

# (json_key, display label, modality group)
LAYOUT = [
    ("text_only",             "Text-Only\n(CLIP)",      "text"),
    ("vision_clip",           "Vision\n(CLIP, logs)",   "vision"),
    ("vision_clip_fws",       "Vision\n(CLIP, +FWS)",   "vision"),
    ("multimodal_fusion",     "Multimodal\n(fusion)",   "multimodal"),
    ("multimodal_fusion_fws", "Multimodal\n(+FWS)",     "multimodal"),
    ("grounding_dino",        "Grounding\nDINO",        "vision"),
    ("blip2",                 "BLIP-2\n(VQA)",          "multimodal"),
]


def _val(results, key, field):
    entry = results.get(key, {})
    v = entry.get(field)
    return v if isinstance(v, (int, float)) else None


def make_figure(results_path, out_path):
    with open(results_path) as f:
        doc = json.load(f)
    results = doc["results"] if "results" in doc else doc

    labels = [lab for _, lab, _ in LAYOUT]
    colors = [COLORS[g] for _, _, g in LAYOUT]
    f1 = [_val(results, k, "macro_f1") for k, _, _ in LAYOUT]
    acc = [_val(results, k, "accuracy") for k, _, _ in LAYOUT]

    maj = doc.get("baselines", {}).get("majority_class", {})
    maj_f1 = maj.get("macro_f1", 0.25)
    maj_acc = maj.get("accuracy", 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    for ax, vals, title, ylab, base in (
        (axes[0], f1, "(a) Macro F1-Score", "Macro F1-Score", maj_f1),
        (axes[1], acc, "(b) Accuracy", "Accuracy", maj_acc),
    ):
        xs = range(len(labels))
        plotted = [(v if v is not None else 0.0) for v in vals]
        bars = ax.bar(xs, plotted, color=colors, edgecolor="black", linewidth=1.2, width=0.66)
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.02, "n/a",
                        ha="center", va="bottom", fontsize=10, style="italic", color="#555")
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(base, ls="--", color="#999", lw=1.3)
        ax.text(len(labels) - 0.5, base + 0.04, "majority class",
                ha="right", va="bottom", fontsize=9, color="#999")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("GeoMM-Bench (Vilkyciai-22, n=11) — backbone: openai/clip-vit-base-patch32",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print("Values plotted (None shown as n/a):")
    for (_, lab, _), a, b in zip(LAYOUT, f1, acc):
        print(f"  {lab.replace(chr(10), ' '):26s} F1={a}  Acc={b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/geomm_bench_results.json")
    ap.add_argument("--out", default="images/Figure2_Results_Comparison.png")
    args = ap.parse_args()
    make_figure(args.results, args.out)
