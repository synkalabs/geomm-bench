#!/usr/bin/env python3
"""Assert that the repo reproduces the reported text-only pilot numbers.

Runs the text-only CLIP baseline (no imagery required; downloads CLIP weights
on first use) and checks macro-F1 / accuracy against the values recorded in
results/pilot_results.json. Exits non-zero on mismatch so it can gate CI.

Usage:
    python scripts/check_reproduction.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomm_bench import baselines as B
from geomm_bench import metrics as M

TOL = 0.001
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    gt = json.load(open(os.path.join(ROOT, "data", "ground_truth.json")))["intervals"]
    reported = json.load(open(os.path.join(ROOT, "results", "pilot_results.json")))
    expected = reported["results"]["text_only_clip"]

    y_true = [g["lithology"] for g in gt]
    y_pred = [B.classify_text_only(g["description"])["predicted"] for g in gt]
    f1 = M.macro_f1(y_true, y_pred)
    acc = M.accuracy(y_true, y_pred)

    print(f"text_only_clip  macro_f1={f1:.3f} (expected {expected['macro_f1']})  "
          f"accuracy={acc:.3f} (expected {expected['accuracy']})")

    ok = (abs(f1 - expected["macro_f1"]) <= TOL
          and abs(acc - expected["accuracy"]) <= TOL)
    if not ok:
        print("MISMATCH: text-only baseline does not reproduce the reported numbers.")
        return 1
    print("OK: text-only baseline reproduces the reported numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
