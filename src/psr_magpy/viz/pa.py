"""PA / RVM panel plotting for snapshot figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from psr_magpy.line.pa_theta_filter import FilteredPAData
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.physical_conversions import PA_RVM, tangent_phi_to_observer_phi

logger = get_logger(__name__, console_format_brief=True)

PHI_ROLL = 270.0  # default φ-axis roll for PA panels (degrees)


@dataclass
class PreparedPAScatter:
    """θ-filtered PA samples ready for PA and pseudo-3D scatter panels."""

    pa_values: np.ndarray
    phi_values: np.ndarray
    colatitude_values: Optional[np.ndarray]
    r_line_stacked: Optional[np.ndarray]
    scatter_colors: Optional[np.ndarray]


def prepare_pa_scatter_data(
    filtered: FilteredPAData,
    r_lc: float,
    *,
    radius_coeff: float = 0.2,
) -> PreparedPAScatter:
    """Sort by radius (descending draw order: smallest r on top) and optional colours."""
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
        order = np.argsort(radii)[::-1]
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

def compute_colors_with_radius_saturation(
    colatitude_values: np.ndarray,
    radii: np.ndarray,
    R_LC: float,
    cmap_name: str = 'hsv',
    vmin: float = 0.0,
    vmax: float = 180.0,
    max_saturation_shift: float = 0.8,
    cycle_period: float = 10.0,
) -> np.ndarray:
    """Compute colors from colatitude with saturation controlled by radii.
    
    Parameters
    ----------
    colatitude_values : np.ndarray
        Colatitude values for base color mapping.
    radii : np.ndarray
        Radii values to control saturation (larger radii = more grey).
    R_LC : float
        Light cylinder radius for normalization.
    cmap_name : str, optional
        Colormap name. Default is 'hsv'.
    vmin : float, optional
        Minimum value for colormap normalization. Default is 0.0.
    vmax : float, optional
        Maximum value for colormap normalization. Default is 180.0.
    max_saturation_shift : float, optional
        Maximum saturation shift for largest radii. Default is 0.8.
    cycle_period : float, optional
        Period in degrees for full colormap cycle. Default is 5.0.
    
    Returns
    -------
    np.ndarray
        Array of RGBA colors with shape (N, 4).
    """
    import colorsys
    import matplotlib.colors as mcolors
    
    cmap = matplotlib.colormaps[cmap_name]
    
    # Normalize colatitude for colormap with cycling
    # Each cycle_period degrees maps to full colormap cycle
    colatitude_normalized = (colatitude_values % cycle_period) / cycle_period
    base_colors = cmap(colatitude_normalized)  # Shape: (N, 4) RGBA
    
    # Normalize radii by R_LC (instead of min/max)
    radii_normalized = (radii - 1.0) / (1.9*R_LC - 1.0)
    
    # Compute blend factor: larger radii get more blend with gray (more grey)
    # Blend factor of 1.0 = fully gray, 0.0 = original color
    # Clamp normalized radii to [0, 1] for blend factor
    radii_clamped = np.clip(radii_normalized, 0, 1)
    blend_factors = radii_clamped * max_saturation_shift
    
    # Apply gray blending to each color
    colors_shifted = []
    for base_color, blend_factor in zip(base_colors, blend_factors):
        rgb = np.array(base_color[:3])  # Extract RGB
        
        # Calculate luminance (perceived brightness) using standard weights
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        
        # Create lighter gray (increase luminance)
        gray_luminance = min(1.0, luminance * 1.25)  # Make gray 25% lighter, capped at 1.0
        gray = np.array([gray_luminance, gray_luminance, gray_luminance])
        
        # Blend original color with gray
        blended_rgb = (1 - blend_factor) * rgb + blend_factor * gray
        
        # Clamp to [0, 1]
        blended_rgb = np.clip(blended_rgb, 0, 1)
        
        colors_shifted.append((blended_rgb[0], blended_rgb[1], blended_rgb[2], base_color[3]))  # Preserve alpha
    
    return np.array(colors_shifted)


def _transform_scatter_phi(phi_values: np.ndarray, phi_roll_deg: float) -> np.ndarray:
    """Map tangent φ to panel x-coordinates (observer frame, then roll)."""
    phi = tangent_phi_to_observer_phi(phi_values)
    phi = phi + phi_roll_deg
    return np.mod(phi, 360.0)


def rolled_phi_ticks(
    phi_roll_deg: float = PHI_ROLL,
    tick_step: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis positions and rolled φ labels for PA / skymap / OFLR panels.

    Returns
    -------
    positions, labels
        ``positions`` include a closing ``360``; corresponding label is ``90``.
    """
    seed = np.arange(0.0, 360.0, float(tick_step))
    if seed.size == 0:
        return np.asarray([0.0, 360.0]), np.asarray([0.0, 90.0])
    if seed.size == 1:
        return np.append(seed, 360.0), np.append(seed.copy(), 90.0)
    roll_steps = int(round(float(phi_roll_deg) / float(tick_step))) % int(seed.size)
    labels = np.roll(seed, roll_steps)
    positions = np.append(seed, 360.0)
    labels = np.append(labels, 90.0)
    return positions, labels


def _phi_tick_labels(x_ticks: np.ndarray, phi_roll_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper around :func:`rolled_phi_ticks`."""
    if len(x_ticks) < 2:
        return np.asarray(x_ticks, dtype=float).copy(), np.asarray(x_ticks, dtype=float).copy()
    tick_step = float(x_ticks[1] - x_ticks[0])
    return rolled_phi_ticks(phi_roll_deg=float(phi_roll_deg), tick_step=tick_step)


def calculate_RVM_curve(
    target_theta: float,
    alpha_angle: float,
    phi_roll_deg: float = PHI_ROLL,
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate RVM curve with discontinuities broken.
    
    Parameters
    ----------
    target_theta : float
        Target theta value (observer angle) for RVM curve.
    alpha_angle : float
        Magnetic inclination angle in degrees.
    phi_roll_deg : float, optional
        Additive roll applied to φ after RVM evaluation. Default is :data:`PHI_ROLL`.
    n_points : int, optional
        Number of points in the curve. Default is 1000.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (phi_grid, pa_grid) tuple with transformed and broken curves.
    """
    # Calculate RVM curve in original phi space (-180 to 180, sorted)
    phi_grid_orig = np.linspace(-180, 180, n_points) 
    pa_grid = PA_RVM(alpha_angle, target_theta, phi_grid_orig)
    
    # Detect PA discontinuities in original space (where phi is sorted)
    pa_diff = np.abs(np.diff(pa_grid))
    break_mask = pa_diff > 30  # True PA discontinuities
    
    # Transform phi after break detection
    phi_grid = phi_grid_orig + phi_roll_deg
    phi_grid[phi_grid>360] -= 360
    phi_grid[phi_grid<0] += 360
    
    phi_diff = np.abs(np.diff(phi_grid))
    phi_wrap = phi_diff > 180  # Wrapping point, not a real discontinuity
    break_mask = break_mask | (phi_wrap)
    
    # Apply breaks at detected discontinuities
    if np.any(break_mask):
        phi_grid_broken = phi_grid.copy()
        pa_grid_broken = pa_grid.copy()
        # Find indices where breaks occur (shift by 1 since diff is one element shorter)
        break_indices = np.where(break_mask)[0] + 1
        # Insert NaN before each break point
        for idx in reversed(break_indices):
            phi_grid_broken = np.insert(phi_grid_broken, idx, np.nan)
            pa_grid_broken = np.insert(pa_grid_broken, idx, np.nan)
        phi_grid = phi_grid_broken
        pa_grid = pa_grid_broken
    
    return phi_grid, pa_grid


def plot_RVM_curve(
    target_theta: float,
    alpha_angle: float,
    ax=None,
    theta_tolerance: Optional[float] = None,
    phi_roll_deg: float = PHI_ROLL,
    zorder: int = 1000,
) -> Tuple:
    """Plot RVM curve.
    
    Parameters
    ----------
    target_theta : float
        Target theta value (observer angle) for RVM curve.
    alpha_angle : float
        Magnetic inclination angle in degrees.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    theta_tolerance : Optional[float], optional
        Tolerance for theta. If provided, plots additional curves for 
        target_theta ± theta_tolerance. Default is None.
    phi_roll_deg : float, optional
        Additive roll applied to φ (must match scatter transform). Default is :data:`PHI_ROLL`.
    zorder : int, optional
        Z-order for the RVM curve. Default is 1000 (on top).
    
    Returns
    -------
    Tuple
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Calculate RVM curve
    phi_grid, pa_grid = calculate_RVM_curve(
        target_theta, alpha_angle, phi_roll_deg=phi_roll_deg
    )
    
    # Plot RVM curve
    ax.plot(phi_grid, pa_grid, color='k', lw=0.8, linestyle='--', zorder=zorder)
    
    """
    # Plot tolerance curves if provided
    if theta_tolerance is not None and theta_tolerance > 0:
        phi_grid_lower, pa_grid_lower = calculate_RVM_curve(target_theta - theta_tolerance, alpha_angle)
        phi_grid_upper, pa_grid_upper = calculate_RVM_curve(target_theta + theta_tolerance, alpha_angle)
        ax.plot(phi_grid_lower, pa_grid_lower, color='0.1', linewidth=1, zorder=100)
        ax.plot(phi_grid_upper, pa_grid_upper, color='0.1', linewidth=1, zorder=100)
    """
    return fig, ax


def plot_PA_scatter(
    pa_values: np.ndarray,
    phi_values: np.ndarray,
    *,
    phi_roll_deg: float = PHI_ROLL,
    ax=None,
    colors: Optional[np.ndarray] = None,
    **scatter_kwargs
) -> Tuple:
    """Plot PA scatter points.
    
    Parameters
    ----------
    pa_values : np.ndarray
        PA values in degrees.
    phi_values : np.ndarray
        Tangent phi values in degrees.
    phi_roll_deg : float, optional
        Additive roll applied after φ flip (``360 - φ``). Default is :data:`PHI_ROLL`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    colors : Optional[np.ndarray], optional
        Array of RGBA colors with shape (N, 4). If provided, overrides c/cmap/vmin/vmax.
        Default is None.
    **scatter_kwargs
        Additional keyword arguments passed to scatter.
    
    Returns
    -------
    Tuple
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    phi_sorted = _transform_scatter_phi(phi_values, phi_roll_deg)
    pa_sorted = pa_values
    
    # Use provided colors if available
    if colors is not None and len(colors) == len(pa_values):
        scatter_kwargs['c'] = colors
        scatter_kwargs.pop('cmap', None)  # Remove cmap since we're using explicit colors
        scatter_kwargs.pop('vmin', None)
        scatter_kwargs.pop('vmax', None)
    
    # Plot scatter (on top)
    ax.scatter(phi_sorted, pa_sorted, zorder=2, **scatter_kwargs)
    
    return fig, ax


def plot_PA_vs_phi(
    pa_values: np.ndarray,
    phi_values: np.ndarray,
    target_theta: float,
    alpha_angle: float,
    ax=None,
    colors: Optional[np.ndarray] = None,
    theta_tolerance: Optional[float] = None,
    phi_roll_deg: float = PHI_ROLL,
    phi_tick_step: float = 45.0,
    **plot_kwargs
) -> Tuple:
    """Plot PA vs phi for filtered data (combines RVM curve and scatter).
    
    Parameters
    ----------
    pa_values : np.ndarray
        PA values in degrees.
    phi_values : np.ndarray
        Tangent phi values in degrees.
    target_theta : float
        Target theta value (observer angle) for title and RVM curve.
    alpha_angle : float
        Magnetic inclination angle in degrees.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    colors : Optional[np.ndarray], optional
        Array of RGBA colors with shape (N, 4). If provided, overrides c/cmap/vmin/vmax.
        Default is None.
    theta_tolerance : Optional[float], optional
        Tolerance for theta. If provided, fills region between RVM curves for 
        target_theta ± theta_tolerance. Default is None.
    phi_roll_deg : float, optional
        Additive roll applied to φ for RVM curve and scatter. Default is :data:`PHI_ROLL`.
    phi_tick_step : float, optional
        Spacing of φ axis tick marks in degrees. Default is 45.
    **plot_kwargs
        Additional keyword arguments passed to scatter and plot.
    
    Returns
    -------
    Tuple
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Extract scatter plot kwargs
    scatter_kwargs = {k: v for k, v in plot_kwargs.items() if k in ['s', 'alpha', 'c', 'cmap', 'vmin', 'vmax', 'marker', 'edgecolors']}
    
    # Extract zorder for RVM curve (default 1000)
    rvm_zorder = plot_kwargs.get('zorder', 1000)
    
    # Plot RVM curve
    plot_RVM_curve(
        target_theta,
        alpha_angle,
        ax=ax,
        theta_tolerance=theta_tolerance,
        phi_roll_deg=phi_roll_deg,
        zorder=rvm_zorder,
    )

    # Plot scatter
    plot_PA_scatter(
        pa_values,
        phi_values,
        ax=ax,
        phi_roll_deg=phi_roll_deg,
        colors=colors,
        **scatter_kwargs,
    )
    
    # Set axis labels and formatting
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\mathrm{PA}$')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_ylim(-90, 90)
    ax.set_xlim(0, 360)
    x_ticks, x_tick_labels = rolled_phi_ticks(
        phi_roll_deg=phi_roll_deg, tick_step=float(phi_tick_step)
    )
    ax.set_xticks(x_ticks, [f"{int(round(label))}$\degree$" for label in x_tick_labels])
    # Set y-axis ticks with 15-degree spacing and add degree symbols
    y_ticks = np.arange(-90, 91, 30)
    ax.set_yticks(y_ticks, [f"{tick}$\degree$" for tick in y_ticks])
    ax.grid(True, alpha=0.3)
    
    return fig, ax


