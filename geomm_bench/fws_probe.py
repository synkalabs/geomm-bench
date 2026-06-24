"""Full Wave Sonic (FWS) multi-image probe for GeoMM-Bench.

This is the second of the benchmark's two probes. Where the model-breadth probe
(baselines.py + optional_models.py) asks "do specialized vision models do better
than CLIP?", this probe asks "does adding more visual modalities help?":

    text-only  ->  vision (logs)  ->  vision (logs + FWS)
                   multimodal (text+logs)  ->  multimodal (text+logs+FWS)

The verified pilot answer is NO: adding the Full Wave Sonic display and fusing
modalities degrades macro-F1 rather than improving it (multimodal+FWS 0.103 vs
text-only 0.746), evidence that the bottleneck is representational, not a lack
of input data.

Faithful to the executed GeoMM_Code.ipynb (OpenAI CLIP backbone). Model-free
parts are importable without weights.
"""
from __future__ import annotations

import numpy as np

from geomm_bench.baselines import (
    LITHOLOGY_CLASSES, TEXT_CLASS_PROMPTS, VISION_CLASS_PROMPTS,
    _text_embeddings, _image_embedding, _classify_by_similarity,
)

# GeoMM_Code uses the OpenAI CLIP backbone. The model-breadth probe uses
# open-clip/LAION. Results from the two backbones are reported separately and
# never merged into one row (see SOURCE_OF_TRUTH and the paper's methods).
BACKBONE_NOTE = "FWS probe uses OpenAI clip-vit-base-patch32 (matches GeoMM_Code.ipynb)."


def classify_vision_multi_image(images, class_prompts=VISION_CLASS_PROMPTS):
    """Vision classification over one OR several stacked displays (logs, +FWS).

    images: a single PIL image or a list of PIL images (e.g. [logs] or [logs, fws]).
    Embeddings of the provided images are averaged before comparison, which is
    the multi-image fusion used in the pilot.
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    embs = [_image_embedding(im) for im in images]
    mean_emb = sum(embs) / len(embs)
    mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
    return _classify_by_similarity(mean_emb, class_prompts)


def classify_multimodal_multi_image(images, description,
                                    text_weight=0.6, vision_weight=0.4,
                                    class_prompts_text=TEXT_CLASS_PROMPTS,
                                    class_prompts_vision=VISION_CLASS_PROMPTS):
    """Fuse a text distribution with a (possibly multi-image) vision distribution."""
    t = _classify_by_similarity(_text_embeddings([description]), class_prompts_text)["probabilities"]
    v = classify_vision_multi_image(images, class_prompts_vision)["probabilities"]
    fused = {k: text_weight * t.get(k, 0) + vision_weight * v.get(k, 0) for k in LITHOLOGY_CLASSES}
    total = sum(fused.values()) or 1.0
    fused = {k: x / total for k, x in fused.items()}
    pred = max(fused, key=fused.get)
    return {"predicted": pred, "confidence": fused[pred], "probabilities": fused}


# The five approaches of the FWS probe, in reporting order.
FWS_PROBE_APPROACHES = [
    "text_only",
    "vision_logs",
    "vision_logs_fws",
    "multimodal_basic",
    "multimodal_full_fws",
]
