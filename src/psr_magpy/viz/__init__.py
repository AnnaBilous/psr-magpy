"""Visualization module for psr_magpy."""

from .field_2d import (
    Quiver2DFieldPlotter,
    Streamline2DFieldPlotter,
    Grid3DVisualizer,
    MultipoleFieldVisualizer,
)
from .field_line_plotting import (
    FieldLineColorGenerator,
    ColormapSmoothColorGenerator,
    NamedColorGenerator,
    SingleColorGenerator,
    FadingColorGenerator,
    FieldLineWidthGenerator,
    PowerLawWidthGenerator,
    ConstantWidthGenerator,
    FieldLinePlotConfig,
)
from .plotting_tools import (
    get_Nlayers,
    initialize_subplot,
    initialize_lines,
    get_line_projection,
    prepare_lines_for_plotting,
    compute_fraction_of_line_length,
    fill_lines,
)
from .field_line_plotting import shift_color_by_saturation_value

__all__ = [
    "Quiver2DFieldPlotter",
    "Streamline2DFieldPlotter",
    "Grid3DVisualizer",
    "MultipoleFieldVisualizer",
    "FieldLineColorGenerator",
    "ColormapSmoothColorGenerator",
    "NamedColorGenerator",
    "SingleColorGenerator",
    "FadingColorGenerator",
    "FieldLineWidthGenerator",
    "PowerLawWidthGenerator",
    "ConstantWidthGenerator",
    "FieldLinePlotConfig",
    "get_Nlayers",
    "initialize_subplot",
    "initialize_lines",
    "get_line_projection",
    "prepare_lines_for_plotting",
    "compute_fraction_of_line_length",
    "fill_lines",
    "shift_color_by_saturation_value",
]
