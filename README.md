# GeoMM-Bench

> To reproduce: see **EXPERIMENT.md** — one unified notebook or CLI runner, one results file.

A benchmark for evaluating multimodal AI on **well log display interpretation**.

GeoMM-Bench measures whether vision–language models can read the visual artifacts
petrophysicists actually use — composite log displays and computed mineral-volume
interpretations — rather than pre-digitised numerical curves. The pilot evaluates
lithofacies classification and documents a large gap between text-based and
vision-based performance.

> **Status: v0.2 pilot.** One well (Vilkyciai-22), 11 labelled intervals, four
> lithofacies, evaluated under two CLIP backbones (probes A and B). Results
> establish the existence and approximate magnitude of the text–vision gap, not
> precise population estimates. A FORCE 2020-based expansion with well-level
> splits and confidence intervals is planned next.

## Results (pilot, n = 11)

The finding holds across model families *and* input configurations: only text
succeeds; every off-the-shelf vision, grounding, VQA, multimodal, and
added-modality approach fails, and adding Full Wave Sonic makes it worse, not
better. The bottleneck is the visual representation, not the data.

| Approach | Macro-F1 | Accuracy |
|---|---|---|
| Text-Only (CLIP) | **0.746** | 72.7% |
| Vision-Only (CLIP) | 0.091 | 18.2% |
| Vision-Only (Grounding DINO) | n/a | n/a |
| Multimodal (BLIP-2) | 0.343 | 36.4% |
| Multimodal (CLIP Fusion) | 0.103 | 18.2% |
| Random baseline | 0.25 | 25% |

Two CLIP backbones (probes A and B) are evaluated and reported separately, never
merged — see **EXPERIMENT.md**. Numbers above are from
`results/geomm_bench_results.json` and are reproduced by
`images/Figure2_Results_Comparison.png`.

## Install

```bash
pip install -r requirements.txt
```

Core text-only and CLIP-vision baselines need only `open-clip-torch`, `torch`,
`Pillow`, `numpy`. Grounding DINO and BLIP-2 additionally need `transformers`.
The vision approaches also need `pdf2image` (and a `poppler` install) to
rasterize source displays.

## Reproduce

See **EXPERIMENT.md** for the full guide. Two equivalent entry points — the
unified notebook `GeoMM-Bench_Experiment.ipynb` or the CLI runner below — call
the same package code and write the same results file.

Text-only sanity check (no imagery; downloads CLIP weights on first run):

```bash
python run_geomm_bench.py --probe A --out results/text_only.json
```

Both probes (require the source PDFs; see `DATASHEET.md` on availability):

```bash
python run_geomm_bench.py --probe both \
    --logs-pdf path/to/vilkyciai22_logs500.pdf \
    --fws-pdf  path/to/vilkyciai22_fws_im_dt.pdf \
    --out results/geomm_bench_results.json
```

Then regenerate the figure from the results file (it cannot drift):

```bash
python scripts/make_results_figure.py \
    --results results/geomm_bench_results.json \
    --out images/Figure2_Results_Comparison.png   # --probe A|B selects the probe
```

## What's in this release

```
GeoMM-Bench_Experiment.ipynb    unified notebook (both probes)  — or —
run_geomm_bench.py              CLI runner (both probes)
EXPERIMENT.md                   how to reproduce (single source of truth)
data/ground_truth.json          11 labelled intervals: depth, label, description, visual features
geomm_bench/constants.py        lithology class set (torch-free)
geomm_bench/baselines.py        CLIP text/vision/fusion, exact prompts, crop calibration (Probe A)
geomm_bench/fws_probe.py        multi-image logs+FWS classifier (Probe B)
geomm_bench/optional_models.py  Grounding DINO + BLIP-2 (Probe A)
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
