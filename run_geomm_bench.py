#!/usr/bin/env python3
"""GeoMM-Bench: the single experiment entry point.

One command runs the benchmark. Two probes, reported separately because they
use different CLIP backbones (never merged into one row):

  PROBE A  model breadth   : text, vision(CLIP), grounding(DINO), vqa(BLIP-2), fusion
                             backbone: open-clip / LAION
  PROBE B  modality adding : text, vision(logs), vision(logs+FWS),
                             multimodal(basic), multimodal(+FWS)
                             backbone: OpenAI CLIP (matches the verified pilot)

Both answer one question: can off-the-shelf AI read well-log displays? Answer
(pilot): no — only text succeeds; every vision/multimodal/added-modality
approach fails. See SOURCE_OF_TRUTH.md.

Usage:
    python run_geomm_bench.py --probe A --logs-pdf logs.pdf
    python run_geomm_bench.py --probe B --logs-pdf logs.pdf --fws-pdf fws.pdf
    python run_geomm_bench.py --probe both --logs-pdf logs.pdf --fws-pdf fws.pdf
Text-only needs no imagery. Vision/probe-B approaches need the source PDFs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomm_bench import baselines as B
from geomm_bench import metrics as M


def _load_pages(pdf, dpi=150):
    from pdf2image import convert_from_path
    return {i + 1: im.convert("RGB") for i, im in enumerate(convert_from_path(pdf, dpi=dpi))}


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)["intervals"]


def run_probe_A(gt, logs_pdf):
    """Model-breadth probe (open-clip/LAION backbone)."""
    from geomm_bench.optional_models import classify_grounding_dino, classify_blip2
    pages = _load_pages(logs_pdf) if logs_pdf else None
    y_true = [g["lithology"] for g in gt]
    preds = {k: [] for k in ["text_only", "vision_clip", "grounding_dino", "blip2", "fusion"]}
    for g in gt:
        crop = B.crop_to_depth_interval(pages[g["page"]], g["page"], g["start_depth"], g["end_depth"]) if pages else None
        preds["text_only"].append(B.classify_text_only(g["description"])["predicted"])
        if crop is not None:
            preds["vision_clip"].append(B.classify_vision_clip(crop)["predicted"])
            preds["grounding_dino"].append(classify_grounding_dino(crop)["predicted"])
            preds["blip2"].append(classify_blip2(crop)["predicted"])
            preds["fusion"].append(B.classify_multimodal_fusion(crop, g["description"])["predicted"])
    return _score(y_true, preds, backbone="open-clip ViT-B-32 laion2b")


def run_probe_B(gt, logs_pdf, fws_pdf):
    """Modality-adding / FWS probe (OpenAI CLIP backbone)."""
    from geomm_bench.fws_probe import classify_vision_multi_image, classify_multimodal_multi_image
    log_pages = _load_pages(logs_pdf)
    fws_pages = _load_pages(fws_pdf) if fws_pdf else None
    y_true = [g["lithology"] for g in gt]
    preds = {k: [] for k in ["text_only", "vision_logs", "vision_logs_fws", "multimodal_basic", "multimodal_full_fws"]}
    for g in gt:
        log_crop = B.crop_to_depth_interval(log_pages[g["page"]], g["page"], g["start_depth"], g["end_depth"])
        fws_crop = None
        if fws_pages:
            fp = fws_pages.get(g["page"], list(fws_pages.values())[0])
            fws_crop = B.crop_to_depth_interval(fp, g["page"], g["start_depth"], g["end_depth"])
        preds["text_only"].append(B.classify_text_only(g["description"])["predicted"])
        preds["vision_logs"].append(classify_vision_multi_image([log_crop])["predicted"])
        imgs_fws = [log_crop] + ([fws_crop] if fws_crop else [])
        preds["vision_logs_fws"].append(classify_vision_multi_image(imgs_fws)["predicted"])
        preds["multimodal_basic"].append(classify_multimodal_multi_image([log_crop], g["description"])["predicted"])
        preds["multimodal_full_fws"].append(classify_multimodal_multi_image(imgs_fws, g["description"])["predicted"])
    return _score(y_true, preds, backbone="OpenAI clip-vit-base-patch32")


def _score(y_true, preds, backbone):
    out = {"_backbone": backbone, "results": {}}
    for k, yp in preds.items():
        if not yp:
            out["results"][k] = {"macro_f1": None, "accuracy": None, "note": "not run (imagery absent)"}
            continue
        _, _, pc = M.per_class_prf(y_true, yp)
        out["results"][k] = {
            "macro_f1": round(M.macro_f1(y_true, yp), 3),
            "accuracy": round(M.accuracy(y_true, yp), 3),
            "per_class_f1": {c: round(v, 3) for c, v in pc.items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["A", "B", "both"], default="both")
    ap.add_argument("--ground-truth", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ground_truth.json"))
    ap.add_argument("--logs-pdf", default=None)
    ap.add_argument("--fws-pdf", default=None)
    ap.add_argument("--out", default="results/geomm_bench_results.json")
    args = ap.parse_args()

    gt = load_ground_truth(args.ground_truth)
    doc = {"benchmark": "GeoMM-Bench", "well": "Vilkyciai-22", "n_intervals": len(gt), "probes": {}}

    if args.probe in ("A", "both"):
        print(">> Probe A (model breadth, open-clip/LAION)")
        doc["probes"]["A_model_breadth"] = run_probe_A(gt, args.logs_pdf)
    if args.probe in ("B", "both"):
        if not args.logs_pdf:
            print("Probe B needs --logs-pdf"); sys.exit(2)
        print(">> Probe B (modality adding + FWS, OpenAI CLIP)")
        doc["probes"]["B_modality_fws"] = run_probe_B(gt, args.logs_pdf, args.fws_pdf)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {args.out}")
    for pname, p in doc["probes"].items():
        print(f"\n[{pname}]  backbone: {p['_backbone']}")
        for k, v in p["results"].items():
            print(f"  {k:24s} F1={v['macro_f1']}  acc={v['accuracy']}")


if __name__ == "__main__":
    main()
