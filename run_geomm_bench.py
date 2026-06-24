#!/usr/bin/env python3
"""GeoMM-Bench runner.

All CLIP approaches use one backbone, openai/clip-vit-base-patch32; Grounding DINO
and BLIP-2 are separate models. Seven approaches:

    text_only             CLIP text embedding of the description
    vision_clip           CLIP image embedding of the log crop
    vision_clip_fws       CLIP image embedding of log + Full Wave Sonic crops
    multimodal_fusion     CLIP text + log-image fusion
    multimodal_fusion_fws CLIP text + (log + FWS) image fusion
    grounding_dino        Grounding DINO (separate model)
    blip2                 BLIP-2 VQA (separate model)

It also writes a backbone_sensitivity block: text-only and vision-CLIP scored
under a second backbone (open-clip/LAION), since vision-CLIP varies a lot by
backbone. Text-only needs no imagery; the rest need the source PDFs
(operator-provided, not redistributed — see DATASHEET.md). See EXPERIMENT.md.

Usage:
    python run_geomm_bench.py --out results/text_only.json          # text-only
    python run_geomm_bench.py --logs-pdf logs.pdf --fws-pdf fws.pdf  # everything
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomm_bench import baselines as B
from geomm_bench import metrics as M

ALL_APPROACHES = [
    "text_only", "vision_clip", "vision_clip_fws",
    "multimodal_fusion", "multimodal_fusion_fws",
    "grounding_dino", "blip2",
]
BACKBONE = "openai/clip-vit-base-patch32 (CLIP approaches); Grounding DINO + BLIP-2 are separate models"
CAVEAT = ("vision-CLIP scores depend heavily on the CLIP backbone (see "
          "backbone_sensitivity), so vision results here should be read as unreliable "
          "rather than uniformly low.")


def _load_pages(pdf, dpi=150):
    from pdf2image import convert_from_path
    return {i + 1: im.convert("RGB") for i, im in enumerate(convert_from_path(pdf, dpi=dpi))}


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)["intervals"]


def run(approaches, gt, log_pages=None, fws_pages=None):
    """Score the requested approaches given already-rasterized page images."""
    from geomm_bench.fws_probe import classify_vision_multi_image, classify_multimodal_multi_image

    y_true = [g["lithology"] for g in gt]
    preds = {a: [] for a in approaches}
    for g in gt:
        log_crop = fws_crop = None
        if log_pages:
            log_crop = B.crop_to_depth_interval(
                log_pages[g["page"]], g["page"], g["start_depth"], g["end_depth"])
        if fws_pages:
            fp = fws_pages.get(g["page"], list(fws_pages.values())[0])
            fws_crop = B.crop_to_depth_interval(fp, g["page"], g["start_depth"], g["end_depth"])
        imgs_fws = [log_crop] + ([fws_crop] if fws_crop is not None else [])

        if "text_only" in approaches:
            preds["text_only"].append(B.classify_text_only(g["description"])["predicted"])
        if log_crop is None:
            continue  # image approaches need imagery; leave their lists empty -> null
        if "vision_clip" in approaches:
            preds["vision_clip"].append(classify_vision_multi_image([log_crop])["predicted"])
        if "vision_clip_fws" in approaches and fws_crop is not None:
            preds["vision_clip_fws"].append(classify_vision_multi_image(imgs_fws)["predicted"])
        if "multimodal_fusion" in approaches:
            preds["multimodal_fusion"].append(
                classify_multimodal_multi_image([log_crop], g["description"])["predicted"])
        if "multimodal_fusion_fws" in approaches and fws_crop is not None:
            preds["multimodal_fusion_fws"].append(
                classify_multimodal_multi_image(imgs_fws, g["description"])["predicted"])
        if "grounding_dino" in approaches:
            from geomm_bench.optional_models import classify_grounding_dino
            preds["grounding_dino"].append(classify_grounding_dino(log_crop)["predicted"])
        if "blip2" in approaches:
            from geomm_bench.optional_models import classify_blip2
            preds["blip2"].append(classify_blip2(log_crop)["predicted"])

    results = {}
    for a in approaches:
        yp = preds[a]
        if len(yp) != len(y_true):
            results[a] = {"macro_f1": None, "accuracy": None,
                          "note": "not run (source PDF required)"}
            continue
        _, _, pc = M.per_class_prf(y_true, yp)
        results[a] = {"macro_f1": round(M.macro_f1(y_true, yp), 3),
                      "accuracy": round(M.accuracy(y_true, yp), 3),
                      "per_class_f1": {c: round(v, 3) for c, v in pc.items()}}
    return results


def build_doc(gt, log_pages=None, fws_pages=None, approaches=ALL_APPROACHES, sensitivity=True):
    """Assemble the full results document (primary results + sensitivity block)."""
    doc = {
        "benchmark": "GeoMM-Bench", "version": "0.2.0", "well": "Vilkyciai-22",
        "n_intervals": len(gt), "backbone": BACKBONE,
        "metric": "macro_f1_present_classes",
        "results": run(approaches, gt, log_pages, fws_pages),
    }
    if sensitivity:
        from geomm_bench.sensitivity import run_sensitivity
        doc["backbone_sensitivity"] = run_sensitivity(gt, log_pages)
    doc["baselines"] = {"random_four_class_f1": 0.25}
    doc["notes"] = [CAVEAT]
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", nargs="+", default=ALL_APPROACHES, choices=ALL_APPROACHES)
    ap.add_argument("--ground-truth", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "ground_truth.json"))
    ap.add_argument("--logs-pdf", default=None)
    ap.add_argument("--fws-pdf", default=None)
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the backbone-sensitivity block")
    ap.add_argument("--out", default="results/geomm_bench_results.json")
    args = ap.parse_args()

    gt = load_ground_truth(args.ground_truth)
    log_pages = _load_pages(args.logs_pdf) if args.logs_pdf else None
    fws_pages = _load_pages(args.fws_pdf) if args.fws_pdf else None
    doc = build_doc(gt, log_pages, fws_pages, args.approaches, sensitivity=not args.no_sensitivity)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {args.out}  (backbone: {BACKBONE})")
    for a in args.approaches:
        v = doc["results"][a]
        print(f"  {a:24s} F1={v['macro_f1']}  acc={v['accuracy']}")
    if "backbone_sensitivity" in doc:
        print("backbone_sensitivity:")
        for bk, blk in doc["backbone_sensitivity"].items():
            row = "  ".join(f"{a}={blk['results'][a]['macro_f1']}" for a in blk["results"])
            print(f"  {bk:16s} {row}")


if __name__ == "__main__":
    main()
