"""Pseudo-3D field-line projection (PA snapshot edge panels)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.open_field_line_edges import load_OFLR_line_data
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.viz.plotting_tools import (
    compute_fraction_of_line_length,
    fill_lines,
    get_line_projection,
    get_Nlayers,
    initialize_lines,
    initialize_subplot,
    prepare_lines_for_plotting,
)

logger = get_logger(__name__, console_format_brief=True)


@dataclass
class ProjectionConfig:
    """Oblique projection parameters for edge field-line pseudo-3D panels.

    Attributes
    ----------
    XZview
        Half-extent of the line-tracing volume in the plane of the main view
        (typically scaled with ``R_LC``).
    YtoX, ZtoX
        Anisotropy: depth (Y) and vertical (Z) stretch relative to X.
    K
        Perspective strength passed to :func:`~psr_magpy.viz.get_Nlayers`.
    axis_half_extent
        Symmetric x/y axis limits; defaults to ``XZview`` when ``None``.
    """

    XZview: float
    YtoX: float
    ZtoX: float
    K: float
    axis_half_extent: Optional[float] = None

    def axis_limit(self) -> float:
        """Half-width for ``set_xlim`` / base limit before ``ZtoX`` scaling."""
        return self.axis_half_extent if self.axis_half_extent is not None else self.XZview


def extract_field_lines_from_set(fls: FieldLineSet) -> List[np.ndarray]:
    """Extract field line coordinates from :class:`FieldLineSet`.

    Returns
    -------
    list of ndarray
        Each array has shape ``(N, 3)``.
    """
    field_lines: List[np.ndarray] = []
    for r_line in fls.r_line:
        if r_line is not None and len(r_line) > 0:
            field_lines.append(np.asarray(r_line).reshape(-1, 3))
    return field_lines


def create_plot_figure(
    XZview: float,
    ZtoX: float,
    Nlayers: int,
    K: float,
    cfg,
    n_plots: int,
) -> tuple:
    """Create figure and initialize line collections for pseudo-3D plotting.

    Returns
    -------
    tuple
        ``(fig, ax, lines, zorder_layer)``
    """
    fig = plt.figure(figsize=(4, ZtoX * 4), dpi=300)
    ax = initialize_subplot(XZview, ZtoX)
    ax.set_facecolor(cfg.color_space)

    ns = plt.Circle((0, 0), 1, color=cfg.color_NS, zorder=Nlayers / 2)
    ax.add_artist(ns)

    zorder_layer = np.arange(0, Nlayers, 1)
    lines = []
    for i in range(n_plots):
        cur_lines = initialize_lines(ax, cfg.width_functions[i], zorder_layer, Nlayers)
        lines.append(cur_lines)

    return fig, ax, lines, zorder_layer


def plot_field_lines_for_r_lc(
    fls: FieldLineSet,
    lines,
    rotph0: float,
    incl: float,
    XZview: float,
    YtoX: float,
    K: float,
    zorder_layer: np.ndarray,
    color_func,
    plot_idx: int,
) -> None:
    """Project and draw one edge-line set on pseudo-3D axes."""
    field_lines = extract_field_lines_from_set(fls)
    if not field_lines:
        logger.warning("No field lines to plot for plot index %s", plot_idx)
        return

    segments_all, zorder_all = prepare_lines_for_plotting(
        field_lines, rotph0, incl, XZview, YtoX, K
    )
    color_param_all = compute_fraction_of_line_length(field_lines)
    fill_lines(
        lines[plot_idx],
        segments_all,
        zorder_all,
        color_param_all,
        zorder_layer,
        color_func,
    )


def plot_scattered_points(
    coordinates: np.ndarray,
    values: np.ndarray,
    ax,
    rotph0: float,
    incl: float,
    XZview: float,
    YtoX: float,
    K: float,
    zorder_layer: np.ndarray,
    cmap=None,
    vmin: float = None,
    vmax: float = None,
    s: float = 1.0,
    alpha: float = 1.0,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Scatter emission points with pseudo-3D projection and depth ordering."""
    if coordinates.shape[0] != 3:
        error_msg = (
            f"coordinates must have shape (3, Npoints), got {coordinates.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(values) != coordinates.shape[1]:
        error_msg = (
            f"values length ({len(values)}) must match "
            f"coordinates.shape[1] ({coordinates.shape[1]})"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    coords_t = coordinates.T
    coords_proj = get_line_projection([coords_t], rotph0, incl)[0]
    xproj, yproj, zproj = coords_proj

    zorder_all = ((yproj + XZview * YtoX) * K).astype(int)
    zorder_all = np.clip(zorder_all, 0, len(zorder_layer) - 1)

    if colors is not None and len(colors) == len(values):
        point_colors = colors
    else:
        if cmap is None or vmin is None or vmax is None:
            error_msg = "Either colors or (cmap, vmin, vmax) must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        point_colors = cmap(plt.Normalize(vmin=vmin, vmax=vmax)(values))

    for k in zorder_layer:
        mask = zorder_all == k
        if np.any(mask):
            ax.scatter(
                xproj[mask],
                zproj[mask],
                c=point_colors[mask],
                s=s,
                alpha=alpha,
                zorder=zorder_layer[k],
                edgecolors="none",
            )


def load_oflr_edge_lines_for_r_lc(
    oflr_edges_dir: Path,
    experiment_number: int,
    r_lc: float,
    *,
    r_max: Optional[float] = None,
    trim_type: str = "sphere",
    r_trim_min: Optional[float] = None,
) -> Optional[FieldLineSet]:
    """Load OFLR edge lines for one experiment and ``R_LC``.

    Trims in-place to ``r_max`` (emission cap) when given; otherwise to ``r_lc``.
    """
    try:
        fls_all = load_OFLR_line_data(
            experiment_number,
            np.array([r_lc]),
            lines_dir=oflr_edges_dir,
            template=FieldLineTemplate.template_line_coordinates_only(),
            data_type="edge",
        )
    except Exception as exc:
        logger.warning("Failed to load edge field lines: %s", exc)
        return None

    logger.info(
        "Requested R_LC=%.2f, available R_LC values: %s",
        r_lc,
        list(fls_all.keys()),
    )
    if r_lc in fls_all:
        fls = fls_all[r_lc]
    else:
        r_lc_rounded = round(r_lc, 2)
        if r_lc_rounded not in fls_all:
            logger.warning(
                "No edge field lines for R_LC=%.2f (rounded %.2f)",
                r_lc,
                r_lc_rounded,
            )
            return None
        logger.warning(
            "No exact match for R_LC=%.2f, using %.2f from files",
            r_lc,
            r_lc_rounded,
        )
        fls = fls_all[r_lc_rounded]

    r_trim = float(r_max) if r_max is not None else float(r_lc)
    fls.trim_to_light_cylinder(
        R_LC=r_trim,
        trim_type=trim_type,
        r_trim_min=r_trim_min,
    )
    return fls


def plot_3d_axes_inset(ax, rotph0: float, incl: float) -> None:
    """Small x/y/z triad inset (bottom centre of ``ax``), matching panel rotation."""
    fig = ax.figure
    bbox = ax.get_position()
    inset_width = 0.08
    inset_height = 0.08
    inset_x0 = bbox.x0 + (bbox.width - inset_width) / 2
    inset_y0 = bbox.y0 - 0.01

    ax_inset = fig.add_axes(
        [inset_x0, inset_y0, inset_width, inset_height],
        facecolor="white",
        alpha=0.9,
        zorder=1000,
    )

    axis_length = 1.0
    axes_3d = np.array(
        [
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ]
    ).T

    rotation_longitude = R.from_rotvec(np.deg2rad(rotph0) * np.array([0, 0, 1]))
    rotation_latitude = R.from_rotvec(np.deg2rad(incl) * np.array([1, 0, 0]))
    rot = rotation_latitude * rotation_longitude
    axes_rotated = rot.apply(axes_3d.T).T

    x_proj = axes_rotated[0, :]
    z_proj = axes_rotated[2, :]
    colors = [(0.7, 0.3, 0.3), (0.3, 0.7, 0.3), (0.3, 0.3, 0.7)]
    y_comp = axes_rotated[:, 1]
    labels = [
        ("x", "y", "z")[i]
        + ("⊙" if y_comp[i] > 0 else "⊗" if y_comp[i] < 0 else "")
        for i in range(3)
    ]

    for i in range(3):
        ax_inset.plot(
            [0, x_proj[i]],
            [0, z_proj[i]],
            color=colors[i],
            linewidth=1.0,
            zorder=1001,
        )
        ax_inset.text(
            x_proj[i] * 1.7,
            z_proj[i] * 1.7,
            labels[i],
            color=colors[i],
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=1002,
        )

    max_extent = 1.5
    ax_inset.set_xlim(-max_extent, max_extent)
    ax_inset.set_ylim(-max_extent, max_extent)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")


def plot_field_lines_on_axes(
    axs_FL: Sequence,
    *,
    oflr_edges_dir: Path,
    experiment_number: int,
    R_LC: float,
    rotph0: Sequence[float],
    incl: Sequence[float],
    projection: ProjectionConfig,
    cfg,
    r_line_stacked: Optional[np.ndarray] = None,
    scatter_colors: Optional[np.ndarray] = None,
    show_axes_inset: bool = False,
    edge_field_line_set: Optional[FieldLineSet] = None,
    edge_r_max: Optional[float] = None,
    edge_trim_type: str = "sphere",
    edge_r_trim_min: Optional[float] = None,
) -> None:
    """Pseudo-3D edge lines + optional PA scatter on each axis in ``axs_FL``."""
    fls = edge_field_line_set
    if fls is None:
        fls = load_oflr_edge_lines_for_r_lc(
            Path(oflr_edges_dir),
            experiment_number,
            R_LC,
            r_max=edge_r_max,
            trim_type=edge_trim_type,
            r_trim_min=edge_r_trim_min,
        )

    nlayers = get_Nlayers(projection.XZview, projection.YtoX, projection.K)
    zorder_layer = np.arange(0, nlayers, 1)

    for i, ax_fl in enumerate(axs_FL):
        ax_fl.set_facecolor(cfg.color_space)
        ns = plt.Circle((0, 0), 1, color=cfg.color_NS, zorder=nlayers / 2)
        ax_fl.add_artist(ns)

        lines = initialize_lines(ax_fl, cfg.width_functions[0], zorder_layer, nlayers)
        line_groups = [lines]

        if fls is not None:
            plot_field_lines_for_r_lc(
                fls,
                line_groups,
                rotph0[i],
                incl[i],
                projection.XZview,
                projection.YtoX,
                projection.K,
                zorder_layer,
                cfg.color_functions[0],
                0,
            )

        if (
            r_line_stacked is not None
            and len(r_line_stacked) > 0
            and scatter_colors is not None
            and len(scatter_colors) > 0
        ):
            r_t = r_line_stacked.T if r_line_stacked.shape[1] == 3 else r_line_stacked
            if len(scatter_colors) != r_t.shape[1]:
                logger.warning(
                    "scatter_colors length (%s) != n_points (%s)",
                    len(scatter_colors),
                    r_t.shape[1],
                )
            else:
                dummy_values = np.zeros(r_t.shape[1])
                plot_scattered_points(
                    r_t,
                    dummy_values,
                    ax_fl,
                    rotph0[i],
                    incl[i],
                    projection.XZview,
                    projection.YtoX,
                    projection.K,
                    zorder_layer,
                    s=0.5,
                    alpha=1.0,
                    colors=scatter_colors,
                )

        half = projection.axis_limit()
        ax_fl.set_xlim(-half, half)
        ax_fl.set_ylim(-half * projection.ZtoX, half * projection.ZtoX)
        ax_fl.axis("off")

        if show_axes_inset:
            plot_3d_axes_inset(ax_fl, rotph0[i], incl[i])
