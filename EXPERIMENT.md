# Reproducing GeoMM-Bench

There are two ways to run the experiment: the notebook
`GeoMM-Bench_Experiment.ipynb` or the CLI `run_geomm_bench.py`. They call the same
package code and write the same results file.

## Approaches

All CLIP approaches use one backbone, `openai/clip-vit-base-patch32`. Grounding
DINO and BLIP-2 are separate models.

| Approach | Input | Model |
|---|---|---|
| `text_only` | description | CLIP (text) |
| `vision_clip` | log image | CLIP (image) |
| `vision_clip_fws` | log + FWS images | CLIP (image) |
| `multimodal_fusion` | text + log image | CLIP fusion |
| `multimodal_fusion_fws` | text + log + FWS images | CLIP fusion |
| `grounding_dino` | log image | Grounding DINO |
| `blip2` | log image | BLIP-2 VQA |

On the Vilkyciai-22 pilot (n=11) only text classifies reasonably (`text_only`
macro-F1 0.726 / accuracy 0.727). The vision, grounding, VQA and added-modality
approaches all do poorly, and adding the Full Wave Sonic display makes the fusion
worse rather than better.

## Backbone sensitivity

Vision-CLIP results vary with the CLIP backbone, so the runner also scores
text-only and vision-CLIP under open-clip/LAION and writes a
`backbone_sensitivity` block next to the main results.

| macro-F1 | OpenAI CLIP | open-clip / LAION |
|---|---|---|
| `text_only` | 0.726 | 0.726 |
| `vision_clip` | 0.278 | 0.578 |

Text-only is the same on both backbones; vision-CLIP roughly doubles (0.278 →
0.578) just by changing the pretraining corpus. That is why both are reported
instead of picking one. The LAION backbone needs `open-clip-torch`; without it,
only the OpenAI row is scored.

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

Regenerate the figure from the results file:

```bash
python scripts/make_results_figure.py \
  --results results/geomm_bench_results.json \
  --out images/Figure2_Results_Comparison.png
```

The committed `results/geomm_bench_results.json` was produced from a run with the
operator PDFs, so all approaches are filled. Without those rasters (which are not
redistributed) only `text_only` is reproducible; the image approaches need the
PDFs.

## Package layout

```
GeoMM-Bench_Experiment.ipynb  notebook
run_geomm_bench.py            CLI entry point (all approaches)
geomm_bench/
  constants.py                class set (no heavy deps)
  baselines.py                CLIP backbone (openai/clip-vit-base-patch32): text / vision / fusion + crop calibration
  fws_probe.py                multi-image logs+FWS CLIP approaches
  optional_models.py          Grounding DINO, BLIP-2 (separate models)
  sensitivity.py              the same approaches under open-clip/LAION (optional)
  metrics.py                  macro-F1, per-class P/R/F1 (no torch)
data/ground_truth.json        11 intervals: depth, label, description, features
scripts/make_results_figure.py    plots from the results JSON
scripts/make_architecture_figure.py
results/                      results JSON written here
```

The README table and the figure are both generated from the results JSON. If you
change a prompt, crop, or model, re-run the benchmark so they stay in step.
