"""CLIP baselines for lithofacies classification from well-log displays.

The text-only, vision-only and fusion approaches (and the multi-image FWS
variants in fws_probe.py) all use one CLIP backbone, openai/clip-vit-base-patch32.
Grounding DINO and BLIP-2 (optional_models.py) are separate models. Everything is
off the shelf, no fine-tuning. See EXPERIMENT.md for results and the
backbone-sensitivity note.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def get_device():
    """Select the best available device and dtype."""
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


DEVICE, DTYPE = get_device()


# --------------------------------------------------------------------------
# Class prompts (shared by every CLIP-based approach)
# --------------------------------------------------------------------------
from geomm_bench.constants import LITHOLOGY_CLASSES

TEXT_CLASS_PROMPTS = {
    "sandstone": ("A geological formation dominated by quartz sand grains with low gamma ray, "
                  "yellow coloring in mineral models, and good porosity."),
    "shale": ("A clay-rich formation with high gamma ray readings above 80 GAPI, "
              "magenta coloring in mineral models, and low permeability."),
    "limestone": ("A calcite-dominated carbonate formation with low gamma ray, "
                  "green coloring in mineral models, and high density."),
    "dolomite": ("A magnesium-rich carbonate formation with low gamma ray, "
                 "cyan coloring in mineral models, and moderate to high density."),
}

VISION_CLASS_PROMPTS = {
    "sandstone": "A well log image showing yellow filled regions with low gamma ray curve deflection.",
    "shale": "A well log image showing magenta or hatched filled regions with high gamma ray curve deflection.",
    "limestone": "A well log image showing green filled regions with low gamma ray curve deflection.",
    "dolomite": "A well log image showing cyan or light blue filled regions with low gamma ray curve deflection.",
}

GROUNDING_QUERIES = {
    "sandstone": "yellow region. yellow fill. sand colored area.",
    "shale": "magenta region. pink fill. hatched area.",
    "limestone": "green region. green fill. green colored area.",
    "dolomite": "cyan region. light blue fill. blue colored area.",
}

# Depth-to-pixel calibration for the Vilkyciai-22 pilot (1:500 scale, ~125 m/page).
# These are the exact constants used to produce the reported numbers.
PAGE_DEPTHS = {2: (1950, 2050), 3: (2025, 2150), 4: (2125, 2250)}
HEADER_FRAC = 0.08
FOOTER_FRAC = 0.05
CROP_MARGIN_FRAC = 0.02


# --------------------------------------------------------------------------
# CLIP backbone — single shared model: openai/clip-vit-base-patch32
# --------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_clip = {}


def load_clip(model_name=CLIP_MODEL_NAME):
    """Load and cache the single CLIP model shared by all CLIP-based approaches."""
    if "model" in _clip:
        return _clip
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    _clip.update(model=model, processor=processor)
    return _clip


def _normalize(emb):
    return emb / emb.norm(dim=-1, keepdim=True)


def _text_embeddings(texts):
    # Canonical CLIP text forward: pooled text-model output -> text projection.
    c = load_clip()
    inputs = c["processor"](text=list(texts), return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        pooled = c["model"].text_model(**inputs).pooler_output
        emb = c["model"].text_projection(pooled)
    return _normalize(emb)


def _image_embedding(image):
    # Canonical CLIP image forward: pooled vision-model output -> visual projection.
    c = load_clip()
    inputs = c["processor"](images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        pooled = c["model"].vision_model(**inputs).pooler_output
        emb = c["model"].visual_projection(pooled)
    return _normalize(emb)


def _classify_by_similarity(query_emb, class_prompts):
    class_emb = _text_embeddings(list(class_prompts.values()))
    sims = (query_emb @ class_emb.T).squeeze(0)
    probs = torch.softmax(sims * 100, dim=0).cpu().numpy()
    names = list(class_prompts.keys())
    idx = int(probs.argmax())
    return {"predicted": names[idx], "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(names, probs)}}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def classify_text_only(description, class_prompts=TEXT_CLASS_PROMPTS):
    """Text-only baseline: CLIP text embedding of the description vs class prompts."""
    return _classify_by_similarity(_text_embeddings([description]), class_prompts)


def classify_vision_clip(image, class_prompts=VISION_CLASS_PROMPTS):
    """Vision-only baseline: CLIP image embedding vs visual class prompts."""
    return _classify_by_similarity(_image_embedding(image), class_prompts)


def classify_multimodal_fusion(image, description, text_weight=0.6, vision_weight=0.4):
    """CLIP fusion baseline: weighted blend of text and vision distributions."""
    t = classify_text_only(description)["probabilities"]
    v = classify_vision_clip(image)["probabilities"]
    fused = {k: text_weight * t.get(k, 0) + vision_weight * v.get(k, 0) for k in LITHOLOGY_CLASSES}
    total = sum(fused.values()) or 1.0
    fused = {k: x / total for k, x in fused.items()}
    pred = max(fused, key=fused.get)
    return {"predicted": pred, "confidence": fused[pred], "probabilities": fused}


def crop_to_depth_interval(image, page, start_depth, end_depth):
    """Crop a rasterized log page to a depth interval using the pilot calibration.

    Note: approximate page-to-depth mapping (1:500 scale, ~125 m/page). The
    expanded release will replace this with verified depth-tick registration.
    """
    w, h = image.size
    if page not in PAGE_DEPTHS:
        return image
    page_start, page_end = PAGE_DEPTHS[page]
    page_range = page_end - page_start
    usable = h * (1 - HEADER_FRAC - FOOTER_FRAC)
    rel_start = (start_depth - page_start) / page_range
    rel_end = (end_depth - page_start) / page_range
    y1 = int(h * HEADER_FRAC + usable * max(0.0, rel_start))
    y2 = int(h * HEADER_FRAC + usable * min(1.0, rel_end))
    margin = int(h * CROP_MARGIN_FRAC)
    return image.crop((0, max(0, y1 - margin), w, min(h, y2 + margin)))
