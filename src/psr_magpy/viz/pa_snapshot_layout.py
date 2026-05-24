"""4×4 GridSpec layout for PA snapshot composite figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


@dataclass(frozen=True)
class PASnapshotAxes:
    """Axes handles for one PA snapshot figure."""

    figure: Figure
    skymap: Axes
    oflr: Axes
    field_lines: Tuple[Axes, ...]
    pa: Axes


@dataclass(frozen=True)
class PASnapshotLayout:
    """Standard Multipole PA snapshot panel geometry (4×4 GridSpec).

    Panel map (row × col):

    - Skymap: rows 0–1, col 0
    - Pseudo-3D field lines: rows 0–1, cols 1–3
    - OFLR (+ em overlay): rows 2–3, col 0
    - PA vs φ: rows 2–3, cols 1–3
    """

    n_rows: int = 4
    n_cols: int = 4
    n_field_line_panels: int = 3
    figure_size: Tuple[float, float] = (4.0, 3.4)
    constrained_layout: bool = True
    w_pad: float = 0.01
    h_pad: float = 0.01

    @property
    def skymap_slice(self) -> tuple:
        return (slice(0, 2), 0)

    @property
    def oflr_slice(self) -> tuple:
        return (slice(2, 4), 0)

    @property
    def pa_slice(self) -> tuple:
        return (slice(2, 4), slice(1, 4))

    def field_line_slice(self, index: int) -> tuple:
        """Slice for pseudo-3D panel ``index`` in ``0 .. n_field_line_panels-1``."""
        if index < 0 or index >= self.n_field_line_panels:
            error_msg = (
                f"field line panel index {index} out of range "
                f"[0, {self.n_field_line_panels})"
            )
            raise IndexError(error_msg)
        return (slice(0, 2), index + 1)

    def create_figure(self) -> Tuple[Figure, GridSpec]:
        """Build figure and :class:`~matplotlib.gridspec.GridSpec`."""
        fig = plt.figure(
            figsize=self.figure_size,
            constrained_layout=self.constrained_layout,
        )
        if self.constrained_layout:
            fig.set_constrained_layout_pads(w_pad=self.w_pad, h_pad=self.h_pad)
        gs = gridspec.GridSpec(self.n_rows, self.n_cols, figure=fig)
        return fig, gs

    def add_axes(self, figure: Figure, grid: GridSpec) -> PASnapshotAxes:
        """Attach subplots to ``grid`` and return axis handles."""
        skymap_row, skymap_col = self.skymap_slice
        oflr_row, oflr_col = self.oflr_slice
        pa_row, pa_col = self.pa_slice

        ax_skymap = figure.add_subplot(grid[skymap_row, skymap_col])
        ax_oflr = figure.add_subplot(grid[oflr_row, oflr_col])
        axs_fl: List[Axes] = []
        for i in range(self.n_field_line_panels):
            fl_row, fl_col = self.field_line_slice(i)
            ax_fl = figure.add_subplot(grid[fl_row, fl_col])
            ax_fl.set_aspect("equal")
            axs_fl.append(ax_fl)
        ax_pa = figure.add_subplot(grid[pa_row, pa_col])

        return PASnapshotAxes(
            figure=figure,
            skymap=ax_skymap,
            oflr=ax_oflr,
            field_lines=tuple(axs_fl),
            pa=ax_pa,
        )

    def create_axes(self) -> PASnapshotAxes:
        """Create figure, GridSpec, and all panel axes in one call."""
        figure, grid = self.create_figure()
        return self.add_axes(figure, grid)
