"""Evaluation metrics for GeoMM-Bench.

Primary metric: macro-averaged F1 over the four lithofacies classes.
Unresolved predictions (a prediction not in the class set) are counted as
errors (they cannot match any true label), which is the convention behind
the reported pilot numbers.
"""
from __future__ import annotations

from collections import defaultdict

from geomm_bench.constants import LITHOLOGY_CLASSES


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


def baseline_scores(y_true, classes=LITHOLOGY_CLASSES):
    """Reference baselines computed from the label distribution.

    Returns a majority-class baseline, which always predicts the most frequent
    class, and a uniform-random baseline. The majority-class scores are exact.
    The uniform-random accuracy is 1/K, and its macro-F1 is the expected value of
    predicting each of the K classes with probability 1/K, using the per-class
    expectations precision = prevalence and recall = 1/K. These give honest floors
    on an imbalanced label set, where a fixed 1/K is not the macro-F1 of chance.
    """
    from collections import Counter
    counts = Counter(y_true)
    n = len(y_true)
    k = len(classes)
    present = [c for c in classes if counts.get(c, 0) > 0]
    if not present or n == 0:
        return {}
    maj = counts.most_common(1)[0][0]
    maj_pred = [maj] * n

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    unif_f1 = sum(_f1(counts[c] / n, 1.0 / k) for c in present) / len(present)
    return {
        "majority_class": {
            "predict": maj,
            "macro_f1": round(macro_f1(y_true, maj_pred, classes), 3),
            "accuracy": round(accuracy(y_true, maj_pred), 3),
        },
        "random_uniform": {
            "macro_f1": round(unif_f1, 3),
            "accuracy": round(1.0 / k, 3),
            "note": "expected value of predicting each class with probability 1/K",
        },
    }
