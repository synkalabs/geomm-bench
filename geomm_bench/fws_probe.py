"""Multi-image (logs + Full Wave Sonic) CLIP approaches.

These add the FWS display as a second image to the vision and fusion approaches,
to check whether more visual modality helps. They use the CLIP backbone from
baselines.py (openai/clip-vit-base-patch32), not a separate model. In the pilot,
adding FWS lowers macro-F1 rather than raising it.
"""
from __future__ import annotations

from geomm_bench.baselines import (
    LITHOLOGY_CLASSES, TEXT_CLASS_PROMPTS, VISION_CLASS_PROMPTS,
    _text_embeddings, _image_embedding, _classify_by_similarity,
)


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
