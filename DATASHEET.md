# Datasheet — GeoMM-Bench v0.1

Following the *Datasheets for Datasets* convention (Gebru et al.).

## Motivation
GeoMM-Bench evaluates whether multimodal AI systems can interpret well log
*display imagery*, the visual form in which much subsurface data (including
legacy scanned archives) exists. It was built as a shared, reproducible benchmark
for vision-based lithofacies interpretation, alongside text-only petroleum
benchmarks and numerical-log datasets.

## Composition
- **Instances.** 11 labelled depth intervals from the Vilkyciai-22 well (Baltic
  Basin, Lithuania), 1975–2215 m TVD, 1:500 scale.
- **Label.** One of four lithofacies: sandstone, shale, limestone, dolomite.
  Class counts: limestone 5, shale 3, sandstone 3, dolomite 0 (dolomite is a
  target class with no labelled interval in this pilot).
- **Per instance.** Depth span, page index, a free-text geological description, a
  short list of salient visual features, and the lithology label
  (`data/ground_truth.json`).
- **Imagery.** Each interval corresponds to a depth-cropped region of an
  operator-provided composite log display and a computed mineral-volume
  interpretation. **These source rasters are not redistributed in this release**
  (see Distribution). The crop calibration constants are published in
  `geomm_bench/baselines.py` so that holders of the source PDFs can regenerate
  the exact crops.

## Collection process
Lithology labels and descriptions were produced by petrophysical interpretation
of the Vilkyciai-22 log displays and mineral-volume model. Descriptions were
synthesised from petrophysical knowledge and visual interpretation; expert
geological validation is requested (see open questions in
`data/ground_truth.json` notes).

## Preprocessing
Source PDFs are rasterized at 150 DPI; intervals are cropped by a linear
page-to-depth mapping (`PAGE_DEPTHS`, header/footer fractions in
`baselines.py`). This mapping is approximate; the expanded release will replace
it with depth-tick-verified registration.

## Uses
Intended for benchmarking lithofacies classification and, in future versions,
cross-modal retrieval, VQA, and interval description generation. **Not** intended
as a production interpretation tool or as geological ground truth for the
Vilkyciai-22 well; the pilot is too small for operational use.

## Distribution
- **Released:** labels, descriptions, visual features, crop calibration, the full
  evaluation harness, and the reported results.
- **Not released:** the operator-provided source log/interpretation rasters,
  pending confirmation of redistribution rights. Users who hold equivalent
  displays can run the vision baselines by supplying `--logs-pdf`.
- The FORCE 2020-based expansion (planned v0.2+) will be fully self-contained and
  redistributable under that dataset's terms.

## Limitations
Single well, single basin, 11 intervals, imbalanced classes, dolomite unlabelled.
Reported scores show that a text–vision gap exists and roughly how large it is;
they are not precise population estimates. See the paper's Threats to Validity and
Limitations sections.

## Maintenance
Maintained by Synka Labs. Versioned; v0.2 will add FORCE 2020 wells with
well-level splits and confidence intervals. Issues and contributions via the
project repository.
