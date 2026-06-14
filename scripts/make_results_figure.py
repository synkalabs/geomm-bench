#!/usr/bin/env python3
"""Generate the GeoMM-Bench results figure directly from a results JSON.

The figure is plotted FROM the recorded numbers, so it cannot drift from the
reported results. Run after scripts/run_evaluation.py (or against the shipped
results/pilot_results.json).

Usage:
    python scripts/make_results_figure.py \
        --results results/pilot_results.json \
        --out images/Figure2_Results_Comparison.png

Output ordering and labels follow the paper:
    Text-Only (CLIP), Vision-Only (CLIP), Vision-Only (Grounding DINO),
    Multimodal (BLIP-2), Multimodal (CLIP Fusion)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_BASELINE = 0.25

# (json_key, display label, modality group)
LAYOUT = [
    ("text_only_clip",        "Text-Only\n(CLIP)",            "text"),
    ("vision_only_clip",      "Vision-Only\n(CLIP)",          "vision"),
    ("visual_grounding_dino", "Vision-Only\n(Grounding DINO)", "vision"),
    ("multimodal_vqa_blip2",  "Multimodal\n(BLIP-2)",         "multimodal"),
    ("multimodal_fusion_clip","Multimodal\n(CLIP Fusion)",    "multimodal"),
]

COLORS = {"text": "#2E9E5B", "vision": "#E0524A", "multimodal": "#E8943A"}

# The runner (run_evaluation.py) emits short keys; the canonical
# pilot_results.json uses fully-qualified keys. Accept either.
KEY_ALIASES = {
    "text_only_clip": ["text_only_clip", "text_only"],
    "vision_only_clip": ["vision_only_clip", "vision_clip"],
    "visual_grounding_dino": ["visual_grounding_dino", "grounding_dino"],
    "multimodal_vqa_blip2": ["multimodal_vqa_blip2", "blip2"],
    "multimodal_fusion_clip": ["multimodal_fusion_clip", "fusion"],
}


def _resolve(results, canonical_key):
    for k in KEY_ALIASES.get(canonical_key, [canonical_key]):
        if k in results:
            return results[k]
    return {}


def _val(results, key, field):
    entry = _resolve(results, key)
    v = entry.get(field, None)
    return v if isinstance(v, (int, float)) else None


def make_figure(results_path, out_path):
    with open(results_path) as f:
        doc = json.load(f)
    # Accept either the canonical pilot_results.json schema ({"results": {...}})
    # or the flat {approach: {...}} emitted by run_evaluation.py --out.
    results = doc["results"] if isinstance(doc, dict) and "results" in doc else doc

    labels = [l for _, l, _ in LAYOUT]
    colors = [COLORS[g] for _, _, g in LAYOUT]
    f1 = [_val(results, k, "macro_f1") for k, _, _ in LAYOUT]
    acc = [_val(results, k, "accuracy") for k, _, _ in LAYOUT]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, vals, title, ylab in (
        (axes[0], f1, "(a) Macro F1-Score", "Macro F1-Score"),
        (axes[1], acc, "(b) Accuracy", "Accuracy"),
    ):
        xs = range(len(labels))
        plotted = [(v if v is not None else 0.0) for v in vals]
        bars = ax.bar(xs, plotted, color=colors, edgecolor="black", linewidth=1.2, width=0.62)
        for i, (b, v) in enumerate(zip(bars, vals)):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.02, "n/a",
                        ha="center", va="bottom", fontsize=10, style="italic", color="#555")
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(RANDOM_BASELINE, ls="--", color="#999", lw=1.3)
        ax.text(len(labels) - 0.5, RANDOM_BASELINE + 0.01, "Random baseline",
                ha="right", va="bottom", fontsize=9, color="#999")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")
    # Console echo so the user can eyeball values vs the paper
    print("Values plotted (None shown as n/a):")
    for (k, lab, _), a, b in zip(LAYOUT, f1, acc):
        print(f"  {lab.replace(chr(10),' '):32s} F1={a}  Acc={b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/pilot_results.json")
    ap.add_argument("--out", default="images/Figure2_Results_Comparison.png")
    args = ap.parse_args()
    make_figure(args.results, args.out)
