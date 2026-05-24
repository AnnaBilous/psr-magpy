"""Render one PA snapshot composite figure for a single ``theta_obs``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from psr_magpy.line.pa_theta_filter import FilteredPAData, filter_fieldline_points_by_theta
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.viz.field_line_plotting import FieldLinePlotConfig
from psr_magpy.viz.healpix import project_array_to_equirectangular
from psr_magpy.viz.oflr import plot_OFLR_em_map, plot_OFLR_map
from psr_magpy.viz.pa import compute_colors_with_radius_saturation, plot_PA_vs_phi
from psr_magpy.viz.pa_snapshot_layout import PASnapshotAxes, PASnapshotLayout
from psr_magpy.viz.presets import (
    DEFAULT_INCL,
    DEFAULT_ROT_PH0,
    DEFAULT_SHOW_AXES_INSET,
    DEFAULT_THETA_TOLERANCE,
    pa_field_line_cfg_for_r_lc,
    pa_snapshot_projection_config,
)
from psr_magpy.viz.pseudo3d import ProjectionConfig, plot_field_lines_on_axes
from psr_magpy.viz.skymap import plot_skymap
from psr_magpy.viz.snapshot import SnapshotData
from psr_magpy.viz.style import apply_pa_snapshot_rcparams

logger = get_logger(__name__)

_DEFAULT_SAVE_DIR_ENV_VARS: Tuple[str, ...] = (
    "PA_SNAPSHOT_OUTPUT_DIR",
    "PA_SNAPSHOT_SAVE_DIR",
)


@dataclass
class SnapshotRenderConfig:
    """Rendering options for :func:`render_pa_snapshot_panel` / batch."""

    theta_tolerance: float = DEFAULT_THETA_TOLERANCE
    save_dir: Optional[Path] = None
    resolve_save_dir_from_env: bool = True
    save_dir_env_vars: Tuple[str, ...] = _DEFAULT_SAVE_DIR_ENV_VARS
    show_plot: bool = False
    dpi: int = 150
    close_figures: bool = True
    apply_rcparams: bool = True
    layout: Optional[PASnapshotLayout] = None
    projection: Optional[ProjectionConfig] = None
    rot_ph0: Sequence[float] = field(default_factory=lambda: list(DEFAULT_ROT_PH0))
    incl: Sequence[float] = field(default_factory=lambda: list(DEFAULT_INCL))
    nscl: str = "LightCyan"
    cl_los: str = "b"
    lw_los: float = 1.0
    show_axes_inset: bool = DEFAULT_SHOW_AXES_INSET
    skymap_vmax_percentile: float = 99.0
    field_line_cfg_for_r_lc: Optional[
        Callable[[float], FieldLinePlotConfig]
    ] = None

    def resolved_save_dir(self) -> Optional[Path]:
        """Explicit ``save_dir`` or first set env var (if enabled)."""
        if self.save_dir is not None:
            return Path(self.save_dir)
        if self.resolve_save_dir_from_env:
            return resolve_pa_snapshot_save_dir(env_vars=self.save_dir_env_vars)
        return None

    def projection_for(self, r_lc: float) -> ProjectionConfig:
        """Per-panel projection; default scales with ``r_lc``."""
        if self.projection is not None:
            return self.projection
        return pa_snapshot_projection_config(r_lc)

    def field_line_cfg_for(self, r_lc: float) -> FieldLinePlotConfig:
        """Per-panel field-line style; default from ``viz.presets``."""
        if self.field_line_cfg_for_r_lc is not None:
            return self.field_line_cfg_for_r_lc(r_lc)
        return pa_field_line_cfg_for_r_lc(r_lc)


@dataclass
class PreparedPAScatter:
    """θ-filtered PA samples ready for PA and pseudo-3D scatter panels."""

    pa_values: np.ndarray
    phi_values: np.ndarray
    colatitude_values: Optional[np.ndarray]
    r_line_stacked: Optional[np.ndarray]
    scatter_colors: Optional[np.ndarray]


def format_pa_snapshot_title(snapshot: SnapshotData, theta_obs: float) -> str:
    """LaTeX suptitle matching Multipole snapshot figures."""
    return (
        rf"$\theta_\mathrm{{D}}={snapshot.alpha_deg:.0f}\degree$, "
        rf"$f_\mathrm{{Q}}={snapshot.BQ:.1f}$, "
        rf"$P={snapshot.period_ms:.1f}\,\mathrm{{ms}}$, "
        rf"$\theta_\mathrm{{obs}}={theta_obs:.0f}\degree$"
    )


def format_pa_snapshot_filename(
    snapshot: SnapshotData,
    theta_obs: float,
) -> str:
    """Output basename (with ``.png``) for one rendered panel."""
    name = (
        f"PA_snap_{snapshot.experiment_number:03d}_alpha_{snapshot.alpha_deg:02.0f}_"
        f"fQ_{snapshot.BQ:.1f}_R_LC_{snapshot.R_LC:05.2f}_{theta_obs:03.0f}"
    )
    return name.replace(".", "_") + ".png"


def prepare_pa_scatter_data(
    filtered: FilteredPAData,
    r_lc: float,
    *,
    radius_coeff: float = 0.2,
) -> PreparedPAScatter:
    """Sort by radius and optional HSV scatter colours."""
    pa_values = filtered.pa_values
    phi_values = filtered.phi_values
    colatitude_values = filtered.colatitude_values
    r_line_stacked = filtered.r_line_stacked
    scatter_colors = None

    if (
        len(pa_values) > 0
        and r_line_stacked is not None
        and len(r_line_stacked) == len(pa_values)
        and colatitude_values is not None
    ):
        radii = np.linalg.norm(r_line_stacked, axis=1)
        scatter_colors = compute_colors_with_radius_saturation(
            colatitude_values,
            radii,
            radius_coeff * float(r_lc),
            cmap_name="hsv",
            vmin=0,
            vmax=180,
            cycle_period=10.0,
        )
        order = np.argsort(radii)
        pa_values = pa_values[order]
        phi_values = phi_values[order]
        colatitude_values = colatitude_values[order]
        r_line_stacked = r_line_stacked[order]
        scatter_colors = scatter_colors[order]

    return PreparedPAScatter(
        pa_values=pa_values,
        phi_values=phi_values,
        colatitude_values=colatitude_values,
        r_line_stacked=r_line_stacked,
        scatter_colors=scatter_colors,
    )


def render_pa_snapshot_panel(
    snapshot: SnapshotData,
    theta_obs: float,
    oflr_edges_dir: Path,
    config: Optional[SnapshotRenderConfig] = None,
) -> PASnapshotAxes:
    """Draw full 4×4 PA snapshot for one observer colatitude.

    Parameters
    ----------
    snapshot
        Static data from :func:`~psr_magpy.viz.snapshot.load_snapshot_data`.
    theta_obs
        Observer colatitude in degrees.
    oflr_edges_dir
        Directory with OFLR edge line FITS (``data_products/OFLR_edges``).
    config
        Panel styling and filtering; defaults from :class:`SnapshotRenderConfig`.
    """
    cfg = config or SnapshotRenderConfig(apply_rcparams=False)

    layout = cfg.layout or PASnapshotLayout()
    projection = cfg.projection_for(snapshot.R_LC)
    rot_ph0 = list(cfg.rot_ph0)
    incl = list(cfg.incl)

    panels = layout.create_axes()
    ax_em = panels.skymap
    ax_surf = panels.oflr
    ax_pa = panels.pa

    valid_skymap = snapshot.skymap[~np.isnan(snapshot.skymap)]
    vmax_skymap = None
    if len(valid_skymap) > 0:
        vmax_skymap = float(np.percentile(valid_skymap, cfg.skymap_vmax_percentile))

    plot_skymap(
        snapshot.skymap,
        snapshot.skymap_longitude_deg,
        snapshot.skymap_latitude_deg,
        ax=ax_em,
        theta_obs=theta_obs,
        cl_LOS=cfg.cl_los,
        lw_LOS=cfg.lw_los,
        vmax=vmax_skymap,
    )
    ax_em.set_xticklabels([])
    ax_em.set_xlabel("")

    plot_OFLR_map(
        snapshot.oflr_image,
        snapshot.oflr_longitude_deg,
        snapshot.oflr_latitude_deg,
        ax=ax_surf,
        NScl=cfg.nscl,
        theta_obs=theta_obs,
        cl_LOS=cfg.cl_los,
        lw_LOS=cfg.lw_los,
    )

    filtered = filter_fieldline_points_by_theta(
        snapshot.field_line_set,
        theta_obs,
        cfg.theta_tolerance,
        grid=snapshot.grid,
    )
    scatter = prepare_pa_scatter_data(filtered, snapshot.R_LC)

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
        ax=ax_surf,
        NScl=cfg.nscl,
        theta_obs=theta_obs,
        cl_LOS=cfg.cl_los,
        lw_LOS=cfg.lw_los,
    )

    if len(scatter.pa_values) > 0:
        if scatter.scatter_colors is not None:
            plot_PA_vs_phi(
                scatter.pa_values,
                scatter.phi_values,
                theta_obs,
                snapshot.alpha_deg,
                ax=ax_pa,
                s=0.5,
                alpha=1.0,
                colors=scatter.scatter_colors,
                theta_tolerance=cfg.theta_tolerance,
            )
        else:
            plot_PA_vs_phi(
                scatter.pa_values,
                scatter.phi_values,
                theta_obs,
                snapshot.alpha_deg,
                ax=ax_pa,
                s=0.5,
                alpha=1.0,
                c=scatter.colatitude_values,
                cmap="prism",
                vmin=0,
                vmax=180,
                theta_tolerance=cfg.theta_tolerance,
            )
    else:
        ax_pa.text(
            0.5,
            0.5,
            f"No PA data for theta = {theta_obs:.1f}°",
            ha="center",
            va="center",
            transform=ax_pa.transAxes,
        )

    plot_field_lines_on_axes(
        list(panels.field_lines),
        oflr_edges_dir=Path(oflr_edges_dir),
        experiment_number=snapshot.experiment_number,
        R_LC=snapshot.R_LC,
        rotph0=rot_ph0,
        incl=incl,
        projection=projection,
        cfg=cfg.field_line_cfg_for(snapshot.R_LC),
        r_line_stacked=scatter.r_line_stacked,
        scatter_colors=scatter.scatter_colors,
        show_axes_inset=cfg.show_axes_inset,
    )

    panels.figure.suptitle(
        format_pa_snapshot_title(snapshot, theta_obs),
        fontsize=7,
    )
    return panels


def save_pa_snapshot_panel(
    panels: PASnapshotAxes,
    path: Path,
    *,
    dpi: int = 150,
) -> Path:
    """Save rendered figure to ``path`` and return the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved PA snapshot figure to %s", path)
    return path


def resolve_pa_snapshot_save_dir(
    save_dir: Optional[Path] = None,
    *,
    env_vars: Sequence[str] = ("PA_SNAPSHOT_OUTPUT_DIR", "PA_SNAPSHOT_SAVE_DIR"),
) -> Optional[Path]:
    """Return explicit ``save_dir`` or the first set environment variable path."""
    if save_dir is not None:
        return Path(save_dir)
    for name in env_vars:
        value = os.getenv(name)
        if value:
            return Path(value)
    return None


def render_pa_snapshot_batch(
    snapshot: SnapshotData,
    theta_obs_list: Sequence[float],
    oflr_edges_dir: Path,
    config: Optional[SnapshotRenderConfig] = None,
) -> List[Optional[Path]]:
    """Render one composite figure per ``theta_obs``, reusing ``snapshot``.

    Returns
    -------
    list
        Saved path per angle, or ``None`` when that panel was not written.
    """
    cfg = config or SnapshotRenderConfig()
    theta_obs_list = list(theta_obs_list)
    logger.info(
        "Rendering %d PA snapshot panels for experiment %03d, R_LC=%.2f",
        len(theta_obs_list),
        snapshot.experiment_number,
        snapshot.R_LC,
    )

    output_dir = cfg.resolved_save_dir()

    if cfg.apply_rcparams:
        apply_pa_snapshot_rcparams()

    panel_cfg = SnapshotRenderConfig(
        theta_tolerance=cfg.theta_tolerance,
        layout=cfg.layout,
        projection=cfg.projection,
        rot_ph0=cfg.rot_ph0,
        incl=cfg.incl,
        nscl=cfg.nscl,
        cl_los=cfg.cl_los,
        lw_los=cfg.lw_los,
        show_axes_inset=cfg.show_axes_inset,
        skymap_vmax_percentile=cfg.skymap_vmax_percentile,
        field_line_cfg_for_r_lc=cfg.field_line_cfg_for_r_lc,
        resolve_save_dir_from_env=False,
        apply_rcparams=False,
    )

    saved_paths: List[Optional[Path]] = []
    for index, theta_obs in enumerate(theta_obs_list):
        panels = render_pa_snapshot_panel(
            snapshot,
            float(theta_obs),
            oflr_edges_dir,
            config=panel_cfg,
        )

        saved_path = None
        if output_dir is not None:
            filepath = Path(output_dir) / format_pa_snapshot_filename(
                snapshot, float(theta_obs)
            )
            saved_path = save_pa_snapshot_panel(panels, filepath, dpi=cfg.dpi)
        saved_paths.append(saved_path)

        if cfg.show_plot and index == len(theta_obs_list) - 1:
            plt.show()

        if cfg.close_figures:
            plt.close(panels.figure)

    return saved_paths
