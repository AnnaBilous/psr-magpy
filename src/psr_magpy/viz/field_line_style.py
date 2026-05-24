"""Field line style configuration and color/width generators."""

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Callable, Union


class FieldLineColorGenerator:
    """Base class for color generation."""

    def generate_colors(self) -> Union[Callable, matplotlib.colors.Colormap]:
        """Generate color function or colormap.

        Returns
        -------
        Union[Callable, matplotlib.colors.Colormap]
            Color function or colormap object.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ColormapSmoothColorGenerator(FieldLineColorGenerator):
    """Color generator using a smooth colormap."""

    def __init__(self, colormap_name: str = "inferno", y_frac: float = 1.0):
        """Initialize colormap color generator.

        Parameters
        ----------
        colormap_name : str, optional
            Name of matplotlib colormap. Default is "inferno".
        y_frac : float, optional
            Fraction of colormap to use. Default is 1.0.
        """
        self.colormap = matplotlib.colormaps[colormap_name]
        self.y_frac = y_frac

    def generate_colors(self) -> Callable:
        """Generate color function based on layer number.

        Returns
        -------
        Callable
            Function that takes (layer_no, Nlayers) and returns color.
        """
        def cl_func(layer_no: int, Nlayers: int) -> np.ndarray:
            y_start = (1 - self.y_frac) / 2.0
            y_end = y_start + self.y_frac
            norm = matplotlib.colors.Normalize(
                vmin=y_start * Nlayers, vmax=y_end * Nlayers, clip=True
            )
            return self.colormap(norm(layer_no))

        return cl_func


class NamedColorGenerator(FieldLineColorGenerator):
    """Color generator from named color list."""

    def __init__(self, colors: List[str]):
        """Initialize named color generator.

        Parameters
        ----------
        colors : List[str]
            List of color names or hex codes.
        """
        self.colors = colors

    def generate_colors(self) -> matplotlib.colors.LinearSegmentedColormap:
        """Generate colormap from color list.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            Colormap created from color list.
        """
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            "segmented_colormap", self.colors
        )


class SingleColorGenerator(FieldLineColorGenerator):
    """Color generator for single color."""

    def __init__(self, color: str):
        """Initialize single color generator.

        Parameters
        ----------
        color : str
            Color name or hex code.
        """
        self.color = color

    def generate_colors(self) -> matplotlib.colors.ListedColormap:
        """Generate single-color colormap.

        Returns
        -------
        matplotlib.colors.ListedColormap
            Single-color colormap.
        """
        return matplotlib.colors.ListedColormap(self.color)


class FadingColorGenerator(FieldLineColorGenerator):
    """Color generator with fading gradient."""

    def __init__(
        self,
        color: str = "r",
        delta_hue: float = 0.3,
        min_sat: float = 0.1,
        max_sat: float = 0.8,
    ):
        """Initialize fading color generator.

        Parameters
        ----------
        color : str, optional
            Base color name or hex code. Default is "r".
        delta_hue : float, optional
            Hue variation range. Default is 0.3.
        min_sat : float, optional
            Minimum saturation. Default is 0.1.
        max_sat : float, optional
            Maximum saturation. Default is 0.8.
        """
        self.color = color
        self.delta_hue = delta_hue
        self.max_sat = max_sat
        self.min_sat = min_sat

    def generate_colors(self) -> matplotlib.colors.LinearSegmentedColormap:
        """Generate fading gradient colormap.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            Fading gradient colormap.
        """
        n_colors = 256
        base_color_rgb = matplotlib.colors.to_rgb(self.color)
        base_color_hsv = matplotlib.colors.rgb_to_hsv(base_color_rgb)

        base_hue = base_color_hsv[0]
        dh = self.delta_hue
        hues = np.linspace(base_hue - dh, base_hue + dh, n_colors)
        hues[hues >= 1] = 0.99
        hues[hues < 0] = 0
        base_saturation = base_color_hsv[1]
        saturations = np.linspace(self.max_sat, self.min_sat, n_colors)
        values = np.full(n_colors, base_color_hsv[2])

        gradient_colors = [
            matplotlib.colors.hsv_to_rgb([h, s, v])
            for h, s, v in zip(hues, saturations, values)
        ]
        return LinearSegmentedColormap.from_list(
            "FadingColor", gradient_colors, N=n_colors
        )


class FieldLineWidthGenerator:
    """Base class for line width generation."""

    def generate_widths(self) -> Callable:
        """Generate width function.

        Returns
        -------
        Callable
            Function that takes (layer_no, Nlayers) and returns line width.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class PowerLawWidthGenerator(FieldLineWidthGenerator):
    """Line width generator with power-law distribution."""

    def __init__(self, lw_min: float, lw_max: float, lw_med: float):
        """Initialize power-law width generator.

        Parameters
        ----------
        lw_min : float
            Minimum line width.
        lw_max : float
            Maximum line width.
        lw_med : float
            Median line width.
        """
        self.lw_min = lw_min
        self.lw_max = lw_max
        self.lw_med = lw_med

    def generate_widths(self) -> Callable:
        """Generate power-law width function.

        Returns
        -------
        Callable
            Function that takes (layer_no, Nlayers) and returns line width.
        """
        def lw_func(layer_no: int, Nlayers: int) -> float:
            pli = (
                -np.log(self.lw_med - self.lw_min)
                / (self.lw_max - self.lw_min)
                / np.log(2)
            )
            
            return (self.lw_max - self.lw_min) * (
                layer_no * 1.0 / Nlayers
            ) ** pli + self.lw_min

        return lw_func


class ConstantWidthGenerator(FieldLineWidthGenerator):
    """Line width generator with constant width."""

    def __init__(self, lw: float):
        """Initialize constant width generator.

        Parameters
        ----------
        lw : float
            Constant line width.
        """
        self.lw = lw

    def generate_widths(self) -> Callable:
        """Generate constant width function.

        Returns
        -------
        Callable
            Function that takes (layer_no, Nlayers) and returns constant line width.
        """
        def lw_func(layer_no: int, Nlayers: int) -> float:
            return self.lw

        return lw_func


class FieldLinePlotConfig:
    """Configuration for field line plotting."""

    def __init__(
        self,
        color_generators: List[FieldLineColorGenerator],
        width_generators: List[FieldLineWidthGenerator],
        color_NS: str = "w",
        color_space: str = "k",
        colors_OFL: List[str] = None,
        colormap_OFL: matplotlib.colors.Colormap = None,
    ):
        """Initialize field line plot configuration.

        Parameters
        ----------
        color_generators : List[FieldLineColorGenerator]
            List of color generators.
        width_generators : List[FieldLineWidthGenerator]
            List of width generators.
        color_NS : str, optional
            Color for neutron star surface. Default is "w".
        color_space : str, optional
            Color for space background. Default is "k".
        colors_OFL : List[str], optional
            List of colors for OFL lines. Default is ["r"].
        colormap_OFL : matplotlib.colors.Colormap, optional
            Colormap for OFL lines. Default is matplotlib.colormaps["Reds"].
        """
        if colors_OFL is None:
            colors_OFL = ["r"]
        if colormap_OFL is None:
            colormap_OFL = matplotlib.colormaps["Reds"]

        self.color_NS = color_NS
        self.color_space = color_space
        self.colors_OFL = colors_OFL
        self.colormap_OFL = colormap_OFL
        self.color_generators = color_generators
        self.width_generators = width_generators

        self.color_functions = [
            generator.generate_colors() for generator in self.color_generators
        ]
        self.width_functions = [
            generator.generate_widths() for generator in self.width_generators
        ]


def shift_color_by_saturation_value(colors, saturation_shift, value_shift, hex=False):
    """Shift colors by saturation and value.

    Parameters
    ----------
    colors : list
        List of colors (hex strings or RGB tuples).
    saturation_shift : float
        Amount to shift saturation (subtracted).
    value_shift : float
        Amount to shift value/brightness (added).
    hex : bool, optional
        If True, colors are hex strings. Default is False.

    Returns
    -------
    list
        List of shifted colors as RGBA tuples.
    """
    import colorsys
    import matplotlib.colors as mcolors

    colors_shifted = []
    for color in colors:
        if hex:
            color_rgb = mcolors.hex2color(color)
        else:
            color_rgb = color
        h, s, v = colorsys.rgb_to_hsv(color_rgb[0], color_rgb[1], color_rgb[2])
        s -= saturation_shift
        v += value_shift
        if s < 0:
            s = 0
        if v > 1:
            v = 1
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors_shifted.append((r, g, b, 1.0))
    return colors_shifted

