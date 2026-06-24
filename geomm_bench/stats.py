"""Confidence intervals for the scaled track.

Cluster bootstrap that resamples whole wells with replacement, so the intervals
respect the dependence between samples from the same well rather than treating
intervals as independent.
"""
from __future__ import annotations

import numpy as np

from geomm_bench import metrics as M


def cluster_bootstrap_macro_f1(y_true, y_pred, groups, n=2000, seed=0):
    """95% CI for macro-F1, resampling wells (groups) with replacement."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    groups = np.asarray(groups, dtype=object)
    wells = np.unique(groups)
    idx_by_well = {w: np.where(groups == w)[0] for w in wells}
    out = []
    for _ in range(n):
        samp = rng.choice(wells, len(wells), replace=True)
        idx = np.concatenate([idx_by_well[w] for w in samp])
        out.append(M.macro_f1(list(y_true[idx]), list(y_pred[idx])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
