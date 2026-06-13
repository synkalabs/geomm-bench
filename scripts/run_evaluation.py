#!/usr/bin/env python3
"""Run GeoMM-Bench baselines and report macro-F1 / per-class results.

Usage:
    python scripts/run_evaluation.py --approaches text_only
    python scripts/run_evaluation.py --approaches text_only vision_clip fusion \
        --logs-pdf path/to/vilkyciai22_logs500.pdf

Text-only needs no images. Vision approaches require the source log PDF
(operator-provided, not redistributed; see DATASHEET.md) and pdf2image.
Grounding DINO and BLIP-2 are provided as optional approaches and require
transformers with the respective model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomm_bench import baselines as B
from geomm_bench import metrics as M


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)["intervals"]


def _load_page_images(logs_pdf, dpi=150):
    from pdf2image import convert_from_path
    pages = convert_from_path(logs_pdf, dpi=dpi)
    return {i + 1: img.convert("RGB") for i, img in enumerate(pages)}


def run(approaches, gt_path, logs_pdf=None):
    gt = load_ground_truth(gt_path)
    y_true = [g["lithology"] for g in gt]

    need_images = any(a in approaches for a in ("vision_clip", "fusion", "grounding_dino", "blip2"))
    page_images = None
    if need_images:
        if not logs_pdf or not os.path.exists(logs_pdf):
            print("ERROR: vision approaches require --logs-pdf (see DATASHEET.md).")
            sys.exit(2)
        page_images = _load_page_images(logs_pdf)

    preds = {a: [] for a in approaches}
    for g in gt:
        crop = None
        if need_images:
            full = page_images.get(g["page"])
            crop = B.crop_to_depth_interval(full, g["page"], g["start_depth"], g["end_depth"])
        if "text_only" in approaches:
            preds["text_only"].append(B.classify_text_only(g["description"])["predicted"])
        if "vision_clip" in approaches:
            preds["vision_clip"].append(B.classify_vision_clip(crop)["predicted"])
        if "fusion" in approaches:
            preds["fusion"].append(B.classify_multimodal_fusion(crop, g["description"])["predicted"])
        if "grounding_dino" in approaches:
            from geomm_bench.optional_models import classify_grounding_dino
            preds["grounding_dino"].append(classify_grounding_dino(crop)["predicted"])
        if "blip2" in approaches:
            from geomm_bench.optional_models import classify_blip2
            preds["blip2"].append(classify_blip2(crop)["predicted"])

    print("=" * 60)
    print(f"GeoMM-Bench v0.1  |  Vilkyciai-22  |  n={len(gt)}")
    print("=" * 60)
    out = {}
    for a in approaches:
        f1 = M.macro_f1(y_true, preds[a])
        acc = M.accuracy(y_true, preds[a])
        _, _, pc = M.per_class_prf(y_true, preds[a])
        out[a] = {"macro_f1": round(f1, 3), "accuracy": round(acc, 3),
                  "per_class_f1": {k: round(v, 3) for k, v in pc.items()}}
        print(f"\n{a}")
        print(f"  macro-F1 : {f1:.3f}")
        print(f"  accuracy : {acc:.3f}")
        print(f"  per-class F1: " + ", ".join(f"{k} {v:.2f}" for k, v in pc.items()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", nargs="+", default=["text_only"],
                    choices=["text_only", "vision_clip", "fusion", "grounding_dino", "blip2"])
    ap.add_argument("--ground-truth", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ground_truth.json"))
    ap.add_argument("--logs-pdf", default=None)
    ap.add_argument("--out", default=None, help="Optional path to write results JSON.")
    args = ap.parse_args()
    results = run(args.approaches, args.ground_truth, args.logs_pdf)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")
