#!/usr/bin/env python3
"""Generate the GeoMM-Bench results figure directly from a results JSON.

The figure is plotted FROM the recorded numbers, so it cannot drift from the
reported results. Accepts either results schema:

  * flat       {"results": {<approach>: {...}}}                       (legacy)
  * two-probe  {"probes": {"A_model_breadth": {"results": {...}},
                           "B_modality_fws":  {"results": {...}}}}     (written by
               run_geomm_bench.py and GeoMM-Bench_Experiment.ipynb)

For the two-probe schema, pick which probe to plot with --probe (default A).
The two probes use different CLIP backbones and are never merged into one figure.

Usage:
    python scripts/make_results_figure.py \
        --results results/geomm_bench_results.json \
        --out images/Figure2_Results_Comparison.png [--probe A|B]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_BASELINE = 0.25
COLORS = {"text": "#2E9E5B", "vision": "#E0524A", "multimodal": "#E8943A"}

# (canonical_key, display label, modality group, accepted key aliases)
# Probe A — model breadth: text, vision-CLIP, Grounding DINO, BLIP-2, CLIP fusion.
LAYOUT_A = [
    ("text_only_clip",         "Text-Only\n(CLIP)",             "text",       ["text_only_clip", "text_only"]),
    ("vision_only_clip",       "Vision-Only\n(CLIP)",           "vision",     ["vision_only_clip", "vision_clip"]),
    ("visual_grounding_dino",  "Vision-Only\n(Grounding DINO)", "vision",     ["visual_grounding_dino", "grounding_dino"]),
    ("multimodal_vqa_blip2",   "Multimodal\n(BLIP-2)",          "multimodal", ["multimodal_vqa_blip2", "blip2"]),
    ("multimodal_fusion_clip", "Multimodal\n(CLIP Fusion)",     "multimodal", ["multimodal_fusion_clip", "fusion"]),
]
# Probe B — modality adding: text, vision(logs), vision(logs+FWS), multimodal, +FWS.
LAYOUT_B = [
    ("text_only",           "Text-Only\n(CLIP)",       "text",       ["text_only", "text_only_clip"]),
    ("vision_logs",         "Vision\n(logs)",          "vision",     ["vision_logs"]),
    ("vision_logs_fws",     "Vision\n(logs+FWS)",      "vision",     ["vision_logs_fws"]),
    ("multimodal_basic",    "Multimodal\n(text+logs)", "multimodal", ["multimodal_basic"]),
    ("multimodal_full_fws", "Multimodal\n(+FWS)",      "multimodal", ["multimodal_full_fws"]),
]

PROBE_BLOCK = {"A": "A_model_breadth", "B": "B_modality_fws"}
PROBE_LAYOUT = {"A": LAYOUT_A, "B": LAYOUT_B}


def _select(doc, probe):
    """Return (results_dict, layout, subtitle) from either supported schema."""
    if "results" in doc:                      # flat / legacy schema
        return doc["results"], LAYOUT_A, None
    if "probes" in doc:                       # two-probe schema
        block_key = PROBE_BLOCK[probe]
        if block_key not in doc["probes"]:
            raise SystemExit(
                f"Probe {probe} ({block_key}) not in results; present: {list(doc['probes'])}")
        block = doc["probes"][block_key]
        subtitle = (f"Probe {probe} — {block_key.split('_', 1)[1].replace('_', ' ')}"
                    f"  ·  {block.get('_backbone', '')}")
        return block["results"], PROBE_LAYOUT[probe], subtitle
    raise SystemExit("Unrecognised results JSON: expected a 'results' or 'probes' key.")


def _val(results, aliases, field):
    for k in aliases:
        if k in results:
            v = results[k].get(field)
            return v if isinstance(v, (int, float)) else None
    return None


def make_figure(results_path, out_path, probe="A"):
    with open(results_path) as f:
        doc = json.load(f)
    results, layout, subtitle = _select(doc, probe)

    labels = [lab for _, lab, _, _ in layout]
    colors = [COLORS[g] for _, _, g, _ in layout]
    f1 = [_val(results, al, "macro_f1") for *_, al in layout]
    acc = [_val(results, al, "accuracy") for *_, al in layout]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, vals, title, ylab in (
        (axes[0], f1, "(a) Macro F1-Score", "Macro F1-Score"),
        (axes[1], acc, "(b) Accuracy", "Accuracy"),
    ):
        xs = range(len(labels))
        plotted = [(v if v is not None else 0.0) for v in vals]
        bars = ax.bar(xs, plotted, color=colors, edgecolor="black", linewidth=1.2, width=0.62)
        for b, v in zip(bars, vals):
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

    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")
    # Console echo so the user can eyeball values vs the results file
    print("Values plotted (None shown as n/a):")
    for (_, lab, _, _), a, b in zip(layout, f1, acc):
        print(f"  {lab.replace(chr(10), ' '):32s} F1={a}  Acc={b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/geomm_bench_results.json")
    ap.add_argument("--out", default="images/Figure2_Results_Comparison.png")
    ap.add_argument("--probe", choices=["A", "B"], default="A",
                    help="Which probe to plot from a two-probe results file (default A).")
    args = ap.parse_args()
    make_figure(args.results, args.out, args.probe)
