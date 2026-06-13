# GeoMM-Bench

A benchmark for evaluating multimodal AI on **well log display interpretation**.

GeoMM-Bench measures whether vision–language models can read the visual artifacts
petrophysicists actually use — composite log displays and computed mineral-volume
interpretations — rather than pre-digitised numerical curves. The pilot evaluates
lithofacies classification and documents a large gap between text-based and
vision-based performance.

> **Status: v0.1 pilot.** One well (Vilkyciai-22), 11 labelled intervals, four
> lithofacies. Results establish the existence and approximate magnitude of the
> text–vision gap, not precise population estimates. A FORCE 2020-based expansion
> with well-level splits and confidence intervals is planned for v0.2.

## Results (pilot, n = 11)

| Approach | Model | Macro-F1 | Accuracy |
|---|---|---|---|
| Text-only | CLIP ViT-B/32 (text) | **0.886** | 90.9% |
| Vision-only | CLIP ViT-B/32 (image) | 0.620 | — |
| Visual grounding | Grounding DINO tiny | 0.071 | 9.1% |
| Multimodal VQA | BLIP-2 OPT-2.7B | — | 18.2% |
| Random baseline | — | 0.25 | 25% |

Text descriptions classify well; off-the-shelf vision encoders fail, and
multimodal fusion does not recover the lost signal. See `results/pilot_results.json`.

## Install

```bash
pip install -r requirements.txt
```

Core text-only and CLIP-vision baselines need only `open-clip-torch`, `torch`,
`Pillow`, `numpy`. Grounding DINO and BLIP-2 additionally need `transformers`.
The vision approaches also need `pdf2image` (and a `poppler` install) to
rasterize source displays.

## Reproduce

Text-only (no imagery required — downloads CLIP weights on first run):

```bash
python scripts/run_evaluation.py --approaches text_only --out results/text_only.json
```

Vision approaches (require the source log PDF; see `DATASHEET.md` on availability):

```bash
python scripts/run_evaluation.py \
    --approaches text_only vision_clip fusion \
    --logs-pdf path/to/vilkyciai22_logs500.pdf
```

Optional heavy models:

```bash
python scripts/run_evaluation.py --approaches grounding_dino blip2 \
    --logs-pdf path/to/vilkyciai22_logs500.pdf
```

## What's in this release

```
data/ground_truth.json      11 labelled intervals: depth, label, description, visual features
geomm_bench/baselines.py    CLIP text/vision/fusion, exact prompts, crop calibration constants
geomm_bench/optional_models.py  Grounding DINO + BLIP-2
geomm_bench/metrics.py      macro-F1, per-class P/R/F1, accuracy
scripts/run_evaluation.py   CLI runner
results/pilot_results.json  Reported pilot numbers
DATASHEET.md                Dataset documentation (composition, collection, distribution)
```

The operator-provided source log rasters are **not** redistributed here (see
`DATASHEET.md`). The crop calibration is published so holders of equivalent
displays can regenerate the exact interval crops.

## Citation

```bibtex
@inproceedings{masaba2026geommbench,
  title     = {Toward Multimodal RAG Agents for Subsurface Characterization:
               Introducing GeoMM-Bench and Baseline Results},
  author    = {Masaba, Martin and Pal, Mayur},
  booktitle = {EAGE Annual Conference & Exhibition},
  year      = {2026}
}
```

## License

Code: Apache-2.0 (`LICENSE`). Labels and descriptions: CC BY 4.0.
Source operator rasters: not included; rights reserved by the data provider.
