"""FORCE 2020 scaled track: load, segment, render, and curve features.

The scaled track of GeoMM-Bench. FORCE 2020 ships numerical wireline curves and
per-sample lithology labels, so we segment each well into fixed depth windows,
render a synthetic composite display from the curves, and derive interval-level
curve statistics for the numeric reference. The CLIP image approach in
baselines.py is applied to the rendered display unchanged.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geomm_bench.constants import LITHOLOGY_CLASSES

# FORCE 2020 lithology codes mapped onto the four GeoMM-Bench classes. Codes
# outside this map (marl, chalk, halite, anhydrite, tuff, coal, basement) are
# dropped from the four-class scaled track.
CODE2NAME = {30000: "sandstone", 65030: "sandstone", 65000: "shale", 80000: "shale",
             70000: "limestone", 70032: "limestone", 74000: "dolomite"}
CURVES = ["GR", "RHOB", "NPHI", "RDEP"]


def load_intervals(csv_path, win_m=10.0, purity=0.6, min_coverage=0.7,
                   min_classes=3, max_wells=None):
    """Return (intervals, wells). Each interval has well, lithology, and a window.

    A well qualifies if at least ``min_coverage`` of its samples have all four
    curves and it carries at least ``min_classes`` of the target lithologies.
    Each interval is a ``win_m`` metre window whose majority class exceeds
    ``purity`` and maps into the four classes.
    """
    df = pd.read_csv(csv_path, sep=";",
                     usecols=["DEPT", "FORCE_2020_LITHOFACIES_LITHOLOGY", "WELL"] + CURVES)
    df = df.rename(columns={"FORCE_2020_LITHOFACIES_LITHOLOGY": "CODE"})
    df["name"] = df["CODE"].map(CODE2NAME)

    has_all = df[CURVES].notna().all(axis=1)
    cov = has_all.groupby(df["WELL"]).mean()
    ndist = df.dropna(subset=["name"]).groupby("WELL")["name"].nunique()
    wells = [w for w in cov.index if cov[w] > min_coverage and ndist.get(w, 0) >= min_classes]
    if max_wells:
        wells = wells[:max_wells]

    sub = df[df.WELL.isin(wells)].dropna(subset=CURVES).sort_values(["WELL", "DEPT"])
    intervals = []
    for w, g in sub.groupby("WELL"):
        d0, d1 = g.DEPT.min(), g.DEPT.max()
        s = d0
        while s + win_m <= d1:
            win = g[(g.DEPT >= s) & (g.DEPT < s + win_m)]
            s += win_m
            if len(win) < 20:
                continue
            vc = win["name"].dropna()
            if len(vc) == 0:
                continue
            maj = vc.value_counts()
            name = maj.index[0]
            if name in LITHOLOGY_CLASSES and maj.iloc[0] / len(vc) >= purity:
                intervals.append({"well": w, "lithology": name,
                                  "window": win[["DEPT"] + CURVES].reset_index(drop=True)})
    return intervals, wells


def features(win):
    """Interval-level curve statistics for the numeric reference."""
    return [win.GR.mean(), win.RHOB.mean(), win.NPHI.mean(),
            float(np.log10(win.RDEP.clip(lower=0.01)).median())]


def render(win, size_px=(400, 600)):
    """Render a synthetic composite display (GR, resistivity, density-neutron).

    Curve tracks only, fixed axis ranges and output size, no color fills. Depth
    increases downward.
    """
    fig, ax = plt.subplots(1, 3, figsize=(size_px[0] / 100, size_px[1] / 100), sharey=True)
    d = win["DEPT"].values
    ax[0].plot(win["GR"], d, "g", lw=0.8); ax[0].set_xlim(0, 150)
    ax[1].plot(win["RDEP"].clip(lower=0.01), d, "r", lw=0.8); ax[1].set_xscale("log"); ax[1].set_xlim(0.2, 2000)
    ax[2].plot(win["RHOB"], d, "k", lw=0.8); ax[2].set_xlim(1.95, 2.95)
    a2 = ax[2].twiny(); a2.plot(win["NPHI"], d, "b", lw=0.8); a2.set_xlim(0.45, -0.15)
    ax[0].set_ylim(d.max(), d.min())
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
