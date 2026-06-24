# GeoMM-Bench — The Experiment

**One experiment. One backbone. One results file.**

If you are picking this up cold, this is the only document you need to reproduce
the benchmark. You can run it two equivalent ways — the unified notebook
`GeoMM-Bench_Experiment.ipynb` or the CLI runner `run_geomm_bench.py`. Both call
the same package code and write the same results file.

## What the experiment asks

Can off-the-shelf, general-purpose AI interpret well-log *display images*? All
CLIP-based approaches share a single backbone (`openai/clip-vit-base-patch32`);
Grounding DINO and BLIP-2 are separate, non-CLIP models. Seven approaches:

| Approach | Input | Model |
|---|---|---|
| `text_only` | description | CLIP (text) |
| `vision_clip` | log image | CLIP (image) |
| `vision_clip_fws` | log + FWS images | CLIP (image) |
| `multimodal_fusion` | text + log image | CLIP fusion |
| `multimodal_fusion_fws` | text + log + FWS images | CLIP fusion |
| `grounding_dino` | log image | Grounding DINO |
| `blip2` | log image | BLIP-2 VQA |

**Pilot result (Vilkyciai-22, n=11):** only text classifies reasonably
(`text_only` macro-F1 0.726 / accuracy 0.727). Every vision, grounding, VQA and
added-modality approach fails or is unreliable, and adding Full Wave Sonic makes
it worse. The bottleneck is the visual representation, not the data.

> **Backbone caveat (reported honestly).** Vision-CLIP is sensitive to the CLIP
> backbone: under `openai/clip-vit-base-patch32` it scores ~0.09 macro-F1, but
> under open-clip/LAION ViT-B-32 it reached ~0.62. The conclusion is therefore
> "no off-the-shelf approach is *reliable*," **not** that vision is uniformly
> near-random. `text_only` is backbone-robust (~0.73 under both).

## Reproduce

```bash
pip install -r requirements.txt

# Text-only (no imagery; downloads CLIP weights on first run)
python run_geomm_bench.py --approaches text_only --out results/text_only.json

# All approaches (need the source PDFs; see DATASHEET.md on availability)
python run_geomm_bench.py \
  --logs-pdf vilkyciai22_logs500.pdf \
  --fws-pdf vilkyciai22_fws_im_dt.pdf \
  --out results/geomm_bench_results.json
```

Then regenerate the figure FROM the results (it cannot drift):

```bash
python scripts/make_results_figure.py \
  --results results/geomm_bench_results.json \
  --out images/Figure2_Results_Comparison.png
```

The shipped `results/geomm_bench_results.json` has `text_only` filled (the only
approach reproducible without the operator rasters); the image approaches are
`null` until you supply the PDFs and re-run.

## Package layout

```
GeoMM-Bench_Experiment.ipynb  <- notebook
run_geomm_bench.py            <- OR the CLI entry point (all approaches)
geomm_bench/
  constants.py                class set (no heavy deps)
  baselines.py                CLIP backbone (openai/clip-vit-base-patch32): text / vision / fusion + crop calibration
  fws_probe.py                multi-image logs+FWS CLIP approaches
  optional_models.py          Grounding DINO, BLIP-2 (separate, non-CLIP models)
  metrics.py                  macro-F1, per-class P/R/F1 (torch-free)
data/ground_truth.json        11 intervals: depth, label, description, features
scripts/make_results_figure.py    plots from the results JSON
scripts/make_architecture_figure.py
results/                      results JSON written here
```

## The one rule

A number enters a paper, figure, or the repo only if `run_geomm_bench.py`
produced it and you can open the results JSON right now. No copied-forward
priors, no hand-typed tables, no narrative-template conclusions.
