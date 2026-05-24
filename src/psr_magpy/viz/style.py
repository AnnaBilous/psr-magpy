"""Matplotlib style helpers for PA snapshot composite figures."""

from __future__ import annotations

from typing import Dict

import matplotlib as mpl

PA_SNAPSHOT_RCPARAMS: Dict[str, float] = {
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
}


def apply_pa_snapshot_rcparams() -> None:
    """Apply compact tick/label sizes for 4×4 PA snapshot figures."""
    mpl.rcParams.update(PA_SNAPSHOT_RCPARAMS)
