# GeoMM-Bench

> To reproduce, see **EXPERIMENT.md**.

A benchmark for evaluating multimodal AI on **well log display interpretation**.

GeoMM-Bench measures whether vision–language models can read the visual artifacts
petrophysicists actually use, namely composite log displays and computed mineral-volume
interpretations. The pilot evaluates
lithofacies classification and documents a large gap between text-based and
vision-based performance.

> **Status: v0.2.** Two tracks, real operator displays (Vilkyciai-22, 11 labelled
> intervals) and rendered displays at scale (FORCE 2020, well-level splits and
> bootstrap confidence intervals). A single CLIP backbone
> (`openai/clip-vit-base-patch32`) is used for all CLIP approaches. The version is
> set in `geomm_bench/__init__.py`; see `CHANGELOG.md` for what changed since v0.1.

## Results (pilot, n = 11)

Single backbone for all CLIP approaches: `openai/clip-vit-base-patch32`. Grounding
DINO and BLIP-2 are separate, non-CLIP models. Metric: macro-F1 over the
lithofacies present in the pilot.

| Approach | Macro-F1 | Accuracy |
|---|---|---|
| Text-Only (CLIP) | **0.726** | 72.7% |
| Multimodal fusion (CLIP) | 0.631 | 63.6% |
| Multimodal fusion (CLIP + FWS) | 0.524 | 54.5% |
| Vision (CLIP, logs + FWS) | 0.324 | 36.4% |
| Vision (CLIP, logs) | 0.278 | 27.3% |
| Grounding DINO | 0.205 | 36.4% |
| BLIP-2 (VQA) | 0.143 | 27.3% |
| Random baseline | 0.25 | 25% |

Image approaches require the operator source rasters (not redistributed; see
`DATASHEET.md`); `text_only` runs without them. The numbers above are from a run
with the PDFs and live in `results/geomm_bench_results.json`.

In the pilot, text classifies best (0.726 macro-F1). Every image-based approach is
weaker: multimodal fusion (0.631) does not reach text, the pure-vision approaches
trail further (0.278–0.324), grounding (0.205) and VQA (0.143) fall below the
random-baseline F1, and adding Full Wave Sonic does not close the gap.

### Backbone sensitivity

Vision-CLIP depends on the CLIP backbone, so the same two approaches are scored
under a second backbone and reported explicitly (the `backbone_sensitivity` block
in `results/geomm_bench_results.json`, produced by the same runner):

| Approach (macro-F1) | OpenAI CLIP | open-clip / LAION |
|---|---|---|
| Text-Only (CLIP) | **0.726** | **0.726** |
| Vision-Only (CLIP) | 0.278 | 0.578 |

Text-only is the same on both backbones; vision-CLIP roughly doubles (0.278 →
0.578) just by changing the pretraining corpus. Because the vision result depends
so much on the backbone, the safer reading is that no off-the-shelf approach is
dependable here, rather than that vision always fails.

## Install

```bash
pip install -r requirements.txt
```

All CLIP approaches use `openai/clip-vit-base-patch32` via `transformers`, so the
core needs `torch`, `transformers`, `Pillow`, `numpy`, `matplotlib`. The image
approaches additionally need `pdf2image` (and a `poppler` install) to rasterize
the source displays.

## Reproduce

See **EXPERIMENT.md** for details. The notebook `GeoMM-Bench_Experiment.ipynb` and
the CLI runner below run the same code and write the same results file.

Text-only (no imagery; downloads CLIP weights on first run):

```bash
python run_geomm_bench.py --approaches text_only --out results/text_only.json
```

All approaches (require the source PDFs; see `DATASHEET.md` on availability):

```bash
python run_geomm_bench.py \
    --logs-pdf path/to/vilkyciai22_logs500.pdf \
    --fws-pdf  path/to/vilkyciai22_fws_im_dt.pdf \
    --out results/geomm_bench_results.json
```

Then regenerate the figure from the results file:

```bash
python scripts/make_results_figure.py \
    --results results/geomm_bench_results.json \
    --out images/Figure2_Results_Comparison.png
```

## What's in this release

```
GeoMM-Bench_Experiment.ipynb    notebook (runs all approaches)
run_geomm_bench.py              CLI runner (runs all approaches)
EXPERIMENT.md                   how to reproduce
data/ground_truth.json          11 labelled intervals: depth, label, description, visual features
geomm_bench/constants.py        lithology class set (torch-free)
geomm_bench/baselines.py        CLIP backbone (openai/clip-vit-base-patch32): text/vision/fusion + crop calibration
geomm_bench/fws_probe.py        multi-image logs+FWS CLIP approaches
geomm_bench/optional_models.py  Grounding DINO + BLIP-2 (separate, non-CLIP models)
geomm_bench/metrics.py          macro-F1, per-class P/R/F1, accuracy
scripts/make_results_figure.py  Figure 2, plotted from the results file
results/geomm_bench_results.json  reported pilot numbers
DATASHEET.md                    dataset documentation (composition, collection, distribution)
```

The operator-provided source log rasters are **not** redistributed here (see
`DATASHEET.md`). The crop calibration is published so holders of equivalent
displays can regenerate the exact interval crops.

## Citation

```bibtex
@misc{masaba2026geommbench,
  title  = {Toward Multimodal RAG Agents for Subsurface Characterization:
            Introducing GeoMM-Bench and Baseline Results},
  author = {Masaba, Martin and Pal, Mayur},
  note   = {Preprint; venue to be determined},
  year   = {2026}
}
```

## License

Code: Apache-2.0 (`LICENSE`). Labels and descriptions: CC BY 4.0.
Source operator rasters: not included; rights reserved by the data provider.
