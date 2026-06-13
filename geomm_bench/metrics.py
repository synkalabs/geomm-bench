"""Evaluation metrics for GeoMM-Bench.

Primary metric: macro-averaged F1 over the four lithofacies classes.
Unresolved predictions (a prediction not in the class set) are counted as
errors (they cannot match any true label), which is the convention behind
the reported pilot numbers.
"""
from __future__ import annotations

from collections import defaultdict

from geomm_bench.baselines import LITHOLOGY_CLASSES


def per_class_prf(y_true, y_pred, classes=LITHOLOGY_CLASSES):
    """Return per-class precision, recall, F1 as dicts keyed by class."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if p == t:
            tp[t] += 1
        else:
            fn[t] += 1
            if p in classes:        # unresolved preds add no FP to any real class
                fp[p] += 1
    precision, recall, f1 = {}, {}, {}
    for c in classes:
        p_den = tp[c] + fp[c]
        r_den = tp[c] + fn[c]
        precision[c] = tp[c] / p_den if p_den else 0.0
        recall[c] = tp[c] / r_den if r_den else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) else 0.0)
    return precision, recall, f1


def macro_f1(y_true, y_pred, classes=LITHOLOGY_CLASSES, present_only=True):
    """Macro-averaged F1.

    present_only=True averages over classes that appear in y_true (the pilot
    convention, since dolomite has no labelled interval). Set False to average
    over all four classes.
    """
    _, _, f1 = per_class_prf(y_true, y_pred, classes)
    keys = [c for c in classes if (c in y_true)] if present_only else list(classes)
    if not keys:
        return 0.0
    return sum(f1[c] for c in keys) / len(keys)


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
