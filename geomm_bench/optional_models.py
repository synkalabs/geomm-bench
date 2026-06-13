"""Optional vision baselines requiring transformers and model weights.

These reproduce the visual-grounding and VQA baselines:
  - Grounding DINO (IDEA-Research/grounding-dino-tiny), box/text threshold 0.15
  - BLIP-2 (Salesforce/blip2-opt-2.7b), one-word lithology VQA

Imported lazily by the runner so the core text/vision CLIP path has no
dependency on these weights.
"""
from __future__ import annotations

import numpy as np

from geomm_bench.baselines import DEVICE, GROUNDING_QUERIES, VISION_CLASS_PROMPTS, LITHOLOGY_CLASSES

_gdino = {}
_blip2 = {}


def _load_gdino():
    if "model" in _gdino:
        return _gdino
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    name = "IDEA-Research/grounding-dino-tiny"
    _gdino["processor"] = AutoProcessor.from_pretrained(name)
    _gdino["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(DEVICE).eval()
    return _gdino


def classify_grounding_dino(image, box_threshold=0.15, text_threshold=0.15):
    """Score each class by detection count x mean confidence; argmax over classes."""
    import torch
    g = _load_gdino()
    scores = {}
    for lith, query in GROUNDING_QUERIES.items():
        inputs = g["processor"](images=image, text=query, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = g["model"](**inputs)
        res = g["processor"].post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
        s = res["scores"].cpu().numpy()
        scores[lith] = float(len(s) * s.mean()) if len(s) else 0.0
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    pred = max(probs, key=probs.get) if total > 0 else LITHOLOGY_CLASSES[0]
    return {"predicted": pred, "probabilities": probs}


def _load_blip2():
    if "model" in _blip2:
        return _blip2
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    name = "Salesforce/blip2-opt-2.7b"
    _blip2["processor"] = Blip2Processor.from_pretrained(name)
    _blip2["model"] = Blip2ForConditionalGeneration.from_pretrained(name).to(DEVICE).eval()
    return _blip2


def classify_blip2(image, max_new_tokens=10):
    """One-word lithology VQA; parse the generation for a class keyword."""
    import torch
    b = _load_blip2()
    prompt = ("Question: What lithology does this well log show? "
              "Answer with one word: sandstone, shale, limestone, or dolomite. Answer:")
    inputs = b["processor"](images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = b["model"].generate(**inputs, max_new_tokens=max_new_tokens)
    text = b["processor"].decode(out[0], skip_special_tokens=True).lower()
    pred = next((c for c in LITHOLOGY_CLASSES if c in text), "unresolved")
    return {"predicted": pred, "raw": text}
