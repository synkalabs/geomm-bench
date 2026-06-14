#!/usr/bin/env python3
"""Generate the GeoMM-Bench framework diagram.

Pure matplotlib (no data, no model weights). Corrects the typos in the
hand-made draft (Architecture, novel, Comparison) and lists only the metrics
the paper actually reports plus the taxonomy's intended scope.

Usage:
    python scripts/make_architecture_figure.py --out images/architecture.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, facecolor, edgecolor, title=None, lines=None,
         title_color="#111", title_size=12, line_size=10.5, italic_lines=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor))
    cx = x + w / 2
    cursor = y + h - 0.32
    if title:
        ax.text(cx, cursor, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=title_color)
        cursor -= 0.42
    for i, ln in enumerate(lines or []):
        it = bool(italic_lines and i in italic_lines)
        ax.text(cx, cursor, ln, ha="center", va="top", fontsize=line_size,
                style="italic" if it else "normal", color="#222")
        cursor -= 0.34


def _inner(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
                                linewidth=1.3, edgecolor="#2f6fb0", facecolor="white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, color="#111")


def make_figure(out_path):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Stage 1: Input data
    _box(ax, 0.3, 0.7, 3.0, 4.6, "#CFE6FB", "#2f6fb0", title="INPUT DATA")
    _inner(ax, 0.6, 4.05, 2.4, 0.75, "Well Log")
    _inner(ax, 0.6, 2.85, 2.4, 0.75, "Petrophysical\nInterpretation")
    _inner(ax, 0.6, 1.65, 2.4, 0.75, "Text Data")

    # Stage 2: three approaches
    _box(ax, 4.6, 4.05, 2.6, 1.2, "#F6B5AE", "#d9534f",
         title="TEXT-ONLY", lines=["LLM / RAG"])
    _box(ax, 4.6, 2.4, 2.6, 1.2, "#BFE6C3", "#4CAF50",
         title="VISION-ONLY", lines=["ViT / CLIP"])
    _box(ax, 4.6, 0.7, 2.6, 1.3, "#FAD9A8", "#E8943A",
         title="MULTIMODAL", lines=["Fusion", "Architecture"])

    # Stage 3: evaluation
    _box(ax, 8.5, 0.7, 3.2, 4.6, "#D9D9D9", "#888", title="EVALUATION",
         lines=["\u2022 Lithofacies Classification",
                "\u2022 Cross-Modal Retrieval",
                "\u2022 Visual QA",
                "\u2022 Interval Description",
                "",
                "Metrics (this paper):",
                "macro-F1, accuracy"],
         line_size=10, italic_lines={5})

    # Stage 4: results
    _box(ax, 12.4, 1.6, 2.4, 2.8, "#D8C8F2", "#7E57C2",
         title="BENCHMARK\nRESULTS",
         lines=["Performance", "Comparison", "", "Text + Vision", "+ Multimodal"],
         title_color="#5e35b1", italic_lines={3, 4})

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.6, color="#555"))
    arrow(3.3, 3.0, 4.6, 4.6)
    arrow(3.3, 3.0, 4.6, 3.0)
    arrow(3.3, 3.0, 4.6, 1.35)
    arrow(7.2, 4.6, 8.5, 3.0)
    arrow(7.2, 3.0, 8.5, 3.0)
    arrow(7.2, 1.35, 8.5, 3.0)
    arrow(11.7, 3.0, 12.4, 3.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="images/architecture.png")
    args = ap.parse_args()
    make_figure(args.out)
