# GeoMM-Bench — The Experiment

**One experiment. One entry point. One results file.**

If you are picking this up cold, this is the only document you need to reproduce
the benchmark. There is exactly one runner (`run_geomm_bench.py`) and one results
file it writes. You can run the experiment two equivalent ways: the unified notebook
`GeoMM-Bench_Experiment.ipynb`, or the command-line runner `run_geomm_bench.py`.
Both call the same package code and write the same results file.

## What the experiment asks

Can off-the-shelf, general-purpose AI interpret well-log *display images*?
The benchmark answers this with two complementary probes, so that the conclusion
does not depend on one model or one input configuration:

| Probe | Question it closes | Approaches | Backbone |
|---|---|---|---|
| **A — model breadth** | "Try a real grounding / VLM model, not just CLIP." | text, vision-CLIP, Grounding DINO, BLIP-2, CLIP-fusion | open-clip / LAION |
| **B — modality adding** | "Give it more visual data / fuse modalities." | text, vision(logs), vision(logs+FWS), multimodal(basic), multimodal(+FWS) | OpenAI CLIP |

**Verified pilot result (Vilkyciai-22, n=11):** only text succeeds
(0.727 acc / 0.746 macro-F1). Every vision, grounding, VQA, multimodal, and
added-modality approach fails (all ≤ 0.34 F1). Adding Full Wave Sonic makes it
worse, not better. The bottleneck is the visual representation, not the data.

> The two probes use different CLIP backbones (A: open-clip/LAION; B: OpenAI CLIP).
> Their numbers are reported in separate blocks and never merged into one row.
> On 11 samples the CLIP text score ranges ~0.73–0.89 across backbones — itself
> evidence of pilot-scale fragility, reported honestly.

## Reproduce

```bash
pip install -r requirements.txt

# Both probes (needs the source PDFs; see DATASHEET.md on availability)
python run_geomm_bench.py --probe both \
  --logs-pdf vilkyciai22_logs500.pdf \
  --fws-pdf vilkyciai22_fws_im_dt.pdf \
  --out results/geomm_bench_results.json

# Text-only sanity check (no imagery, downloads CLIP weights on first run)
python run_geomm_bench.py --probe A --out results/text_only.json
```

Then regenerate the figure FROM the results (it cannot drift):

```bash
python scripts/make_results_figure.py \
  --results results/geomm_bench_results.json \
  --out images/Figure2_Results_Comparison.png
```

## Package layout

```
GeoMM-Bench_Experiment.ipynb  <- unified notebook (both probes)
run_geomm_bench.py            <- OR the CLI entry point (both probes)
geomm_bench/
  constants.py                class set (no heavy deps)
  baselines.py                text / vision-CLIP / fusion + crop calibration
  optional_models.py          Grounding DINO, BLIP-2  (Probe A)
  fws_probe.py                multi-image logs+FWS    (Probe B)
  metrics.py                  macro-F1, per-class P/R/F1  (torch-free)
data/ground_truth.json        11 intervals: depth, label, description, features
scripts/make_results_figure.py    plots from results JSON
scripts/make_architecture_figure.py
results/                      results JSON written here
```

## The one rule

A number enters a paper, figure, or the repo only if `run_geomm_bench.py`
produced it and you can open the results JSON right now. No copied-forward
priors, no hand-typed tables, no narrative-template conclusions.
