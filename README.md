# GeoMM-Bench

> To reproduce, see **EXPERIMENT.md**.

A benchmark for evaluating multimodal AI on **well log display interpretation**.

GeoMM-Bench measures whether vision–language models can read the visual artifacts
petrophysicists actually use — composite log displays and computed mineral-volume
interpretations. The pilot evaluates
lithofacies classification and documents a large gap between text-based and
vision-based performance.

> **Status: v0.2 pilot.** One well (Vilkyciai-22), 11 labelled intervals, four
> lithofacies. A single CLIP backbone (`openai/clip-vit-base-patch32`) is used for
> all CLIP approaches. Results establish the existence and approximate magnitude
> of the text–vision gap, not precise population estimates. A FORCE 2020-based
> expansion with well-level splits and confidence intervals is planned next.

## Results (pilot, n = 11)

Single backbone for all CLIP approaches: `openai/clip-vit-base-patch32`. Grounding
DINO and BLIP-2 are separate, non-CLIP models. Metric: macro-F1 over the
lithofacies present in the pilot.

| Approach | Macro-F1 | Accuracy |
|---|---|---|
| Text-Only (CLIP) | **0.726** | 72.7% |
| Vision (CLIP, logs) | —&nbsp;† | —&nbsp;† |
| Vision (CLIP, logs + FWS) | —&nbsp;† | —&nbsp;† |
| Multimodal fusion (CLIP) | —&nbsp;† | —&nbsp;† |
| Multimodal fusion (CLIP + FWS) | —&nbsp;† | —&nbsp;† |
| Grounding DINO | —&nbsp;† | —&nbsp;† |
| BLIP-2 (VQA) | —&nbsp;† | —&nbsp;† |
| Random baseline | 0.25 | 25% |

† Image approaches require the operator source rasters (not redistributed; see
`DATASHEET.md`) and are filled in by running `run_geomm_bench.py` with the PDFs.
Only `text_only` runs without them.

In the pilot, text classifies reasonably (0.726 macro-F1) but the image-based
approaches do not: CLIP-vision, grounding, VQA and the added-modality fusion all
do poorly, and adding Full Wave Sonic makes the fusion worse.

### Backbone sensitivity

Vision-CLIP depends on the CLIP backbone, so the same two approaches are scored
under a second backbone and reported explicitly (the `backbone_sensitivity` block
in `results/geomm_bench_results.json`, produced by the same runner):

| Approach (macro-F1) | OpenAI CLIP | open-clip / LAION |
|---|---|---|
| Text-Only (CLIP) | **0.726** | **0.726** |
| Vision-Only (CLIP) | —&nbsp;† | —&nbsp;† |

† Vision needs the source PDFs; run `run_geomm_bench.py` with them to fill these.
In earlier runs vision-CLIP scored about 0.09 on OpenAI CLIP and about 0.62 on
LAION, while text-only stayed at 0.726 on both. Because the vision result depends
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
