# Licensing

This repository contains three categories of material under different terms.

## 1. Code — Apache License 2.0
All source code (the `geomm_bench/` package, `scripts/`, configuration) is
licensed under Apache-2.0. See `LICENSE`. You may use, modify, and distribute
it, including commercially, subject to the license terms (which include a
patent grant and require preservation of notices).

## 2. Benchmark labels and descriptions — CC BY 4.0
The lithofacies labels, geological interval descriptions, visual-feature
annotations, and derived metadata in `data/` are licensed under Creative
Commons Attribution 4.0 International (CC BY 4.0). You may share and adapt them,
including commercially, provided you give appropriate credit (cite the paper;
see `CITATION.cff`).

## 3. Source well log rasters — NOT included
The original Vilkyciai-22 composite log displays and computed mineral-volume
interpretation rasters are operator-provided. They are **not** included or
redistributed in this repository and are **not** covered by either license
above. The crop calibration constants are published in
`geomm_bench/baselines.py` so that holders of equivalent displays can regenerate
the interval crops. Redistribution of the source rasters is pending confirmation
of rights with the data provider. The planned FORCE 2020-based expansion will be
fully self-contained and redistributable under that dataset's terms.

---
Summary: Apache-2.0 (code), CC BY 4.0 (labels/descriptions), source rasters
withheld pending rights.
