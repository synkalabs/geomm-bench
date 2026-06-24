"""Score text-only and vision-CLIP under a second CLIP backbone.

The benchmark's main backbone is openai/clip-vit-base-patch32 (baselines.py).
Vision-CLIP scores vary a lot by backbone, so this module also runs text-only and
vision-CLIP under open-clip/LAION ViT-B-32, and the runner reports both. open_clip
is only used here and is optional; without it, only the OpenAI backbone is scored.
"""
from __future__ import annotations

import importlib.util

import torch

from geomm_bench import baselines as B
from geomm_bench import metrics as M
from geomm_bench.baselines import DEVICE, TEXT_CLASS_PROMPTS, VISION_CLASS_PROMPTS

OPEN_CLIP_AVAILABLE = importlib.util.find_spec("open_clip") is not None

# The backbone-dependent CLIP approaches compared across backbones.
SENSITIVITY_APPROACHES = ["text_only", "vision_clip"]


class _OpenAIEncoder:
    """Primary backbone — reuses the loaded CLIP model from baselines.py."""
    name = B.CLIP_MODEL_NAME

    def text(self, texts):
        return B._text_embeddings(list(texts))

    def image(self, image):
        return B._image_embedding(image)


class _OpenCLIPEncoder:
    """Second backbone — open-clip / LAION (only used for the sensitivity check)."""
    name = "open-clip ViT-B-32 (laion2b_s34b_b79k)"

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def text(self, texts):
        toks = self.tokenizer(list(texts)).to(DEVICE)
        with torch.no_grad():
            emb = self.model.encode_text(toks)
        return emb / emb.norm(dim=-1, keepdim=True)

    def image(self, image):
        t = self.preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model.encode_image(t)
        return emb / emb.norm(dim=-1, keepdim=True)


def _argmax_class(query_emb, class_prompts, enc):
    class_emb = enc.text(list(class_prompts.values()))
    sims = (query_emb @ class_emb.T).squeeze(0)
    return list(class_prompts.keys())[int(sims.argmax())]


def _predict(enc, gt, approach, log_pages):
    preds = []
    for g in gt:
        if approach == "text_only":
            preds.append(_argmax_class(enc.text([g["description"]]), TEXT_CLASS_PROMPTS, enc))
        elif approach == "vision_clip":
            crop = B.crop_to_depth_interval(
                log_pages[g["page"]], g["page"], g["start_depth"], g["end_depth"])
            preds.append(_argmax_class(enc.image(crop), VISION_CLASS_PROMPTS, enc))
    return preds


def _score(enc, gt, log_pages):
    y = [g["lithology"] for g in gt]
    out = {}
    for a in SENSITIVITY_APPROACHES:
        if a != "text_only" and log_pages is None:
            out[a] = {"macro_f1": None, "accuracy": None, "note": "not run (source PDF required)"}
            continue
        preds = _predict(enc, gt, a, log_pages)
        out[a] = {"macro_f1": round(M.macro_f1(y, preds), 3),
                  "accuracy": round(M.accuracy(y, preds), 3)}
    return out


def run_sensitivity(gt, log_pages=None):
    """Score text-only and vision-CLIP under each available CLIP backbone.

    text-only needs no imagery; vision-CLIP needs the log rasters (log_pages).
    """
    blocks = {"openai_clip": {"backbone": _OpenAIEncoder.name,
                              "results": _score(_OpenAIEncoder(), gt, log_pages)}}
    if OPEN_CLIP_AVAILABLE:
        enc = _OpenCLIPEncoder()
        blocks["open_clip_laion"] = {"backbone": enc.name, "results": _score(enc, gt, log_pages)}
    else:
        blocks["open_clip_laion"] = {
            "backbone": "open-clip ViT-B-32 (laion2b_s34b_b79k)",
            "results": {a: {"macro_f1": None, "accuracy": None,
                            "note": "install open-clip-torch to score this backbone"}
                        for a in SENSITIVITY_APPROACHES}}
    return blocks
