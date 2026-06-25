# Changelog

Notable changes to GeoMM-Bench. The version is defined once in
`geomm_bench/__init__.py`. The runners stamp it into the results files, and
`CITATION.cff`, the data files, the README, and the paper track it. Versions
follow semantic versioning. A change that makes results no longer comparable to a
prior release is a minor (or major) bump and is recorded here.

## v0.2.0

Added
- The FORCE 2020 scaled track. Synthetic display rendering from curves, a numeric
  reference classifier, well-level cross-validation, and cluster bootstrap
  confidence intervals (`geomm_bench/force.py`, `geomm_bench/stats.py`,
  `run_force.py`).
- A two-track benchmark, real operator displays (Vilkyciai-22) and rendered
  displays at scale (FORCE 2020).
- A backbone-sensitivity block that scores text and vision under a second CLIP
  backbone (open-clip/LAION).

Changed
- Single CLIP backbone, `openai/clip-vit-base-patch32`, for all CLIP approaches.
- Headline finding reframed. The lithofacies signal is learnable from the curves
  (numeric reference 0.45 macro-F1 with confidence intervals) while off-the-shelf
  vision does not read the display (0.09). The earlier "text beats vision"
  framing was confounded and is dropped.
- Pilot text-only reported as an oracle, since the descriptions name the label
  (0.726 macro-F1, falling to 0.398 when the label is redacted).
- Pilot numbers are not comparable to v0.1, because the backbone and prompts
  changed.
- Baselines are now computed from the label distribution rather than fixed at
  0.25. The runners report a majority-class baseline and a uniform-random
  baseline, each with macro-F1 and accuracy (`metrics.baseline_scores`). The
  honest macro-F1 floor is the majority-class baseline (pilot 0.21, scaled track
  0.20); the old 0.25 held only as uniform-random accuracy, not as a macro-F1
  floor. Tables and figures use these computed values.

Fixed
- transformers 5.x compatibility. The CLIP image-embedding path now uses the
  submodel forward, and Grounding DINO uses the `threshold` argument.

Note
- The pilot ground truth and Vilkyciai labels are unchanged from v0.1.

## v0.1.0

- Initial pilot. Vilkyciai-22, 11 labelled intervals, four lithofacies classes.
- Text, vision (CLIP), Grounding DINO, BLIP-2, and CLIP fusion baselines on real
  operator displays.
