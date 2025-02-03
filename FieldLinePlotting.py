import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Base class for color generation
class FieldLineColorGenerator:
    def generate_colors(self):
        raise NotImplementedError("Subclasses must implement this method.")


class ColormapSmoothColorGenerator(FieldLineColorGenerator):
    def __init__(self, colormap_name="inferno", y_frac=1.0):
        self.colormap = matplotlib.colormaps[colormap_name]
        self.y_frac = y_frac

    def generate_colors(self):
        def cl_func(layer_no, Nlayers):
            y_start = (1 - self.y_frac) / 2.0
            y_end = y_start + self.y_frac
            norm = matplotlib.colors.Normalize(
                vmin=y_start * Nlayers, vmax=y_end * Nlayers, clip=True
            )
            return self.colormap(norm(layer_no))

        return cl_func


class NamedColorGenerator(FieldLineColorGenerator):
    def __init__(self, colors):
        self.colors = colors

    def generate_colors(self):
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            "segmented_colormap", self.colors
        )


class SingleColorGenerator(FieldLineColorGenerator):
    def __init__(self, color):
        self.color = color

    def generate_colors(self):
        return matplotlib.colors.ListedColormap(self.color)


class FadingColorGenerator(FieldLineColorGenerator):
    def __init__(self, color="r", delta_hue=0.3, min_sat=0.1, max_sat=0.8):
        print(color)

        self.color = color
        self.delta_hue = delta_hue
        self.max_sat = max_sat
        self.min_sat = min_sat

    def generate_colors(self):
        n_colors = 256
        # base_color_rgb = mcolors.to_rgb(self.color)  # Ensure the color is in RGB format
        base_color_hsv = matplotlib.colors.rgb_to_hsv(self.color[:-1])
        print(base_color_hsv)
        # Generate a gradient of hues and saturations
        base_hue = base_color_hsv[0]
        dh = self.delta_hue
        hues = np.linspace(base_hue - dh, base_hue + dh, n_colors)  # Hue gradient
        hues[hues >= 1] = 0.99
        hues[hues < 0] = 0
        base_saturation = base_color_hsv[1]
        saturations = np.linspace(
            self.max_sat, self.min_sat, n_colors
        )  # Saturation gradient
        values = np.full(
            n_colors, base_color_hsv[2]
        )  # Keep value (brightness) constant

        # Convert HSV gradient to RGB
        gradient_colors = [
            matplotlib.colors.hsv_to_rgb([h, s, v])
            for h, s, v in zip(hues, saturations, values)
        ]
        # Create the colormap
        return LinearSegmentedColormap.from_list(
            "FadingColor", gradient_colors, N=n_colors
        )


class FieldLineWidthGenerator:
    def generate_colors(self):
        raise NotImplementedError("Subclasses must implement this method.")


class PowerLawWidthGenerator(FieldLineWidthGenerator):
    def __init__(self, lw_min, lw_max, lw_med):
        self.lw_min = lw_min
        self.lw_max = lw_max
        self.lw_med = lw_med

    def generate_widths(self):
        def lw_func(layer_no, Nlayers):
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
    def __init__(self, lw):
        self.lw = lw

    def generate_widths(self):
        def lw_func(layer_no, Nlayers):
            return self.lw

        return lw_func


class FieldLinePlotConfig:
    def __init__(
        self,
        color_generators,
        width_generators,
        color_NS="w",
        color_space="k",
        colors_OFL=["r"],
        colormap_OFL=matplotlib.colormaps["Reds"],
    ):
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
