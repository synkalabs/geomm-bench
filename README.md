# GeoMM-Bench

> To reproduce: see **EXPERIMENT.md** — notebook or CLI runner, one results file.

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

The pilot finding is that **no off-the-shelf visual approach reliably reads the
displays**: text classifies reasonably (0.726 macro-F1), while CLIP-vision,
grounding, VQA and added-modality fusion fail or are unreliable, and adding Full
Wave Sonic makes it worse.

**Backbone caveat (for honesty).** Vision-CLIP is backbone-sensitive: under
`openai/clip-vit-base-patch32` it scores ~0.09 macro-F1, but under open-clip/LAION
ViT-B-32 it reached ~0.62. So the claim is "no off-the-shelf approach is
*reliable*," not that vision is uniformly near-random. Numbers are in
`results/geomm_bench_results.json` and plotted in
`images/Figure2_Results_Comparison.png`.

## Install

```bash
pip install -r requirements.txt
```

All CLIP approaches use `openai/clip-vit-base-patch32` via `transformers`, so the
core needs `torch`, `transformers`, `Pillow`, `numpy`, `matplotlib`. The image
approaches additionally need `pdf2image` (and a `poppler` install) to rasterize
the source displays.

## Reproduce

See **EXPERIMENT.md** for the full guide. Two equivalent entry points — the
unified notebook `GeoMM-Bench_Experiment.ipynb` or the CLI runner below — call
the same package code and write the same results file.

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

Then regenerate the figure from the results file (it cannot drift):

```bash
python scripts/make_results_figure.py \
    --results results/geomm_bench_results.json \
    --out images/Figure2_Results_Comparison.png
```

## What's in this release

```
GeoMM-Bench_Experiment.ipynb    unified notebook  — or —
run_geomm_bench.py              CLI runner (all approaches)
EXPERIMENT.md                   how to reproduce (single source of truth)
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
