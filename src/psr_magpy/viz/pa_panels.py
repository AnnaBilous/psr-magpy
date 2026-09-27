"""PA snapshot subplot helpers (draw on an existing ``Axes``).

Composite layout and batch rendering live in Multipole ``PA_snapshot``;
see ``analysis/00_miscellaneous/PA_snapshot/REORG_ARCHITECTURE.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.pa_theta_filter import FilteredPAData, filter_fieldline_points_by_theta
from psr_magpy.viz.field_line_style import FieldLinePlotConfig
from psr_magpy.viz.healpix import project_array_to_equirectangular
from psr_magpy.viz.oflr import (
    overlay_oflr_emission_scatter,
    plot_OFLR_em_map,
    plot_OFLR_map,
)
from psr_magpy.viz.pa import PHI_ROLL, PreparedPAScatter, plot_PA_vs_phi, prepare_pa_scatter_data
from psr_magpy.viz.pseudo3d import ProjectionConfig, plot_field_lines_on_axes
from psr_magpy.viz.skymap import plot_skymap
from psr_magpy.viz.snapshot import SnapshotData

__all__ = [
    "filter_snapshot_for_theta",
    "skymap_vmax_from_snapshot",
    "plot_snapshot_skymap_panel",
    "plot_snapshot_oflr_panel",
    "plot_snapshot_pa_panel",
    "plot_snapshot_field_lines_panel",
]


def filter_snapshot_for_theta(
    snapshot: SnapshotData,
    theta_obs: float,
    theta_tolerance: float,
) -> FilteredPAData:
    """θ-filter field-line samples for one observer colatitude."""
    return filter_fieldline_points_by_theta(
        snapshot.field_line_set,
        theta_obs,
        theta_tolerance,
        grid=snapshot.grid,
    )


def skymap_vmax_from_snapshot(
    snapshot: SnapshotData,
    percentile: float = 99.0,
) -> Optional[float]:
    """Percentile vmax for skymap panel; ``None`` if no finite data."""
    valid = snapshot.skymap[~np.isnan(snapshot.skymap)]
    if len(valid) == 0:
        return None
    return float(np.percentile(valid, percentile))


def plot_snapshot_skymap_panel(
    snapshot: SnapshotData,
    ax: Axes,
    theta_obs: float,
    *,
    vmax: Optional[float] = None,
    vmax_percentile: float = 99.0,
    cl_los: str = "b",
    lw_los: float = 0.7,
    hide_x_labels: bool = True,
    **imshow_kwargs: Any,
) -> Tuple[Figure, Axes, Any]:
    """Draw skymap panel for one PA snapshot case."""
    if vmax is None:
        vmax = skymap_vmax_from_snapshot(snapshot, percentile=vmax_percentile)

    fig, ax, im = plot_skymap(
        snapshot.skymap,
        snapshot.skymap_longitude_deg,
        snapshot.skymap_latitude_deg,
        ax=ax,
        theta_obs=theta_obs,
        cl_LOS=cl_los,
        lw_LOS=lw_los,
        vmax=vmax,
        **imshow_kwargs,
    )
    if hide_x_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    return fig, ax, im


def plot_snapshot_oflr_panel(
    snapshot: SnapshotData,
    ax: Axes,
    theta_obs: float,
    filtered: FilteredPAData,
    *,
    color_NS: str = "cyan",
    cl_los: str = "b",
    lw_los: float = 0.7,
    em_overlay: Literal["imshow", "scatter"] = "scatter",
    **imshow_kwargs: Any,
) -> Tuple[Figure, Axes, Any]:
    """Draw OFLR surface map and emission overlay on ``ax``."""
    # Never forward panel-only kwargs to ``plot_OFLR_map`` / ``imshow``.
    em_overlay = imshow_kwargs.pop("em_overlay", em_overlay)  # type: ignore[assignment]
    for key in ("s", "alpha", "edgecolors", "zorder", "cmap", "vmin", "vmax"):
        imshow_kwargs.pop(key, None)

    fig, ax, im = plot_OFLR_map(
        snapshot.oflr_image,
        snapshot.oflr_longitude_deg,
        snapshot.oflr_latitude_deg,
        ax=ax,
        NScl=color_NS,
        theta_obs=theta_obs,
        cl_LOS=cl_los,
        lw_LOS=lw_los,
        **imshow_kwargs,
    )
    if em_overlay == "scatter":
        if filtered.oflr_em_map is not None:
            overlay_oflr_emission_scatter(
                ax,
                filtered.oflr_em_map,
                int(snapshot.nside_oflr),
            )
    else:
        oflr_em_values, oflr_em_lon, oflr_em_lat = project_array_to_equirectangular(
            filtered.oflr_em_map,
            snapshot.grid_indices,
            snapshot.nside_oflr,
            image_size=snapshot.image_size,
        )
        plot_OFLR_em_map(
            oflr_em_values,
            oflr_em_lon,
            oflr_em_lat,
            ax=ax,
            NScl=color_NS,
            theta_obs=theta_obs,
            cl_LOS=cl_los,
            lw_LOS=lw_los,
        )
    return fig, ax, im


def plot_snapshot_pa_panel(
    snapshot: SnapshotData,
    ax: Axes,
    theta_obs: float,
    scatter: PreparedPAScatter,
    *,
    theta_tolerance: float = 0.5,
    scatter_size: float = 0.5,
    scatter_alpha: float = 1.0,
    scatter_marker: str = ".",
    scatter_edgecolors: Optional[str] = "none",
    phi_roll_deg: float = PHI_ROLL,
) -> Tuple[Figure, Axes]:
    """Draw PA vs φ panel (RVM curve + scatter)."""
    fig = ax.figure
    if len(scatter.pa_values) == 0:
        ax.text(
            0.5,
            0.5,
            f"No PA data for theta = {theta_obs:.1f}°",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, ax

    plot_kwargs = dict(
        s=scatter_size,
        alpha=scatter_alpha,
        marker=scatter_marker,
        theta_tolerance=theta_tolerance,
        phi_roll_deg=phi_roll_deg,
    )
    if scatter_edgecolors is not None:
        plot_kwargs["edgecolors"] = scatter_edgecolors
    if scatter.scatter_colors is not None:
        plot_PA_vs_phi(
            scatter.pa_values,
            scatter.phi_values,
            theta_obs,
            snapshot.alpha_deg,
            ax=ax,
            colors=scatter.scatter_colors,
            **plot_kwargs,
        )
    else:
        plot_PA_vs_phi(
            scatter.pa_values,
            scatter.phi_values,
            theta_obs,
            snapshot.alpha_deg,
            ax=ax,
            c=scatter.colatitude_values,
            cmap="prism",
            vmin=0,
            vmax=180,
            **plot_kwargs,
        )
    return fig, ax


def plot_snapshot_field_lines_panel(
    axes: Sequence[Axes],
    snapshot: SnapshotData,
    oflr_edges_dir: Union[str, Path],
    scatter: PreparedPAScatter,
    *,
    rot_ph0: Sequence[float],
    incl: Sequence[float],
    projection: ProjectionConfig,
    field_line_cfg: FieldLinePlotConfig,
    show_axes_inset: bool = False,
    edge_field_line_set: Optional[FieldLineSet] = None,
    edge_r_max: Optional[float] = None,
    edge_trim_type: str = "sphere",
    edge_r_trim_min: Optional[float] = None,
) -> None:
    """Draw pseudo-3D field-line panels on ``axes`` (typically three subplots)."""
    plot_field_lines_on_axes(
        list(axes),
        oflr_edges_dir=Path(oflr_edges_dir),
        experiment_number=snapshot.experiment_number,
        R_LC=snapshot.R_LC,
        rotph0=rot_ph0,
        incl=incl,
        projection=projection,
        cfg=field_line_cfg,
        r_line_stacked=scatter.r_line_stacked,
        scatter_colors=scatter.scatter_colors,
        show_axes_inset=show_axes_inset,
        edge_field_line_set=edge_field_line_set,
        edge_r_max=edge_r_max,
        edge_trim_type=edge_trim_type,
        edge_r_trim_min=edge_r_trim_min,
    )
