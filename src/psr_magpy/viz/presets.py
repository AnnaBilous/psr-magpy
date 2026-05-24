"""Field-line and PA snapshot view presets."""

from __future__ import annotations

import numpy as np

from psr_magpy.viz.field_line_plotting import (
    FieldLinePlotConfig,
    NamedColorGenerator,
    PowerLawWidthGenerator,
)
from psr_magpy.viz.pseudo3d import ProjectionConfig

COLOR_NS_DEFAULT = "PaleTurquoise"
COLOR_SPACE_DEFAULT = "w"

DEFAULT_THETA_TOLERANCE = 0.5
DEFAULT_ROT_PH0 = [-90.0, 0.0, 90.0]
DEFAULT_INCL = [-7.0, -7.0, -7.0]
DEFAULT_SHOW_AXES_INSET = False


def _slate_color_generators() -> list:
    return [
        NamedColorGenerator(
            colors=["LightSlateGray", "SlateGray", "DarkSlateGray"]
        )
    ]


def _powerlaw_widths(
    lw_min: float, lw_max: float, lw_med: float
) -> list:
    return [PowerLawWidthGenerator(np.array([lw_min]), np.array([lw_max]), np.array([lw_med]))]


def pa_field_line_cfg_small_rlc() -> FieldLinePlotConfig:
    """Pseudo-3D line style for ``R_LC < 10`` (Multipole PA snapshot)."""
    return FieldLinePlotConfig(
        _slate_color_generators(),
        _powerlaw_widths(0.002, 0.08, 0.01),
        color_NS=COLOR_NS_DEFAULT,
        color_space=COLOR_SPACE_DEFAULT,
    )


def pa_field_line_cfg_large_rlc() -> FieldLinePlotConfig:
    """Pseudo-3D line style for ``R_LC >= 10``."""
    return FieldLinePlotConfig(
        _slate_color_generators(),
        _powerlaw_widths(0.001, 0.04, 0.002),
        color_NS=COLOR_NS_DEFAULT,
        color_space=COLOR_SPACE_DEFAULT,
    )


def pa_field_line_cfg_caustics() -> FieldLinePlotConfig:
    """Wide dynamic-range widths for caustics-style panels."""
    return FieldLinePlotConfig(
        _slate_color_generators(),
        _powerlaw_widths(0.01, 0.8, 0.1),
        color_NS=COLOR_NS_DEFAULT,
        color_space=COLOR_SPACE_DEFAULT,
    )


# Legacy names (Multipole PA_snapshot)
PA_cfg_small_RLC = pa_field_line_cfg_small_rlc()
PA_cfg_large_RLC = pa_field_line_cfg_large_rlc()
PA_cfg_caustics = pa_field_line_cfg_caustics()


def pa_field_line_cfg_for_r_lc(r_lc: float) -> FieldLinePlotConfig:
    """Pick small- vs large-``R_LC`` preset (module singletons)."""
    return PA_cfg_small_RLC if r_lc < 10.0 else PA_cfg_large_RLC


def pa_snapshot_projection_config(r_lc: float) -> ProjectionConfig:
    """Default pseudo-3D projection for PA snapshot panels."""
    r_lc = float(r_lc)
    return ProjectionConfig(
        XZview=1.05 * r_lc,
        YtoX=3.5,
        ZtoX=2.0,
        K=30.0,
        axis_half_extent=0.7 * r_lc,
    )
