"""OFLR map panel plotting (base map, emission overlay, HEALPix scatter)."""

from __future__ import annotations

from typing import Optional, Tuple

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.viz.los import overlay_observer_los

logger = get_logger(__name__, console_format_brief=True)

def plot_OFLR_map(
    oflr_data: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    figure_size: Tuple[float, float] = (10, 5),
    dpi: int = 150,
    ax=None,
    NScl: str = 'LightCyan',
    theta_obs: Optional[float] = None,
    cl_LOS: str = 'b',
    lw_LOS: float = 0.7,
    plot_contour: bool = False,
    **imshow_kwargs
) -> Tuple:
    """Plot OFLR map using imshow.
    
    Parameters
    ----------
    oflr_data : np.ndarray
        OFLR map image data (2D array, categorical).
    longitude_deg : np.ndarray
        Longitude coordinates in degrees.
    latitude_deg : np.ndarray
        Latitude coordinates in degrees.
    figure_size : Tuple[float, float], optional
        Figure size (width, height). Default is (10, 5).
    dpi : int, optional
        Figure resolution. Default is 150.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    NScl : str, optional
        Color for NaN values. Default is 'LightCyan'.
    theta_obs : Optional[float], optional
        Observer theta angle in degrees. If provided, plots a horizontal line.
    cl_LOS : str, optional
        Color for line of sight. Default is 'b'.
    lw_LOS : float, optional
        Line width for line of sight. Default is 1.0.
    plot_contour : bool, optional
        Whether to plot contours. Default is False.
    **imshow_kwargs
        Additional keyword arguments passed to imshow.
    
    Returns
    -------
    Tuple
        (fig, ax, im) tuple.
    """
    # Create or use existing axes
    if ax is None:
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    
    x_size, y_size = oflr_data.shape
    
    if plot_contour:
        oflr_data_for_contour = oflr_data.copy()    
        oflr_data_for_contour = np.roll(oflr_data_for_contour, int(x_size*3/4), axis=1)
        oflr_data_for_contour[np.isnan(oflr_data_for_contour)] = 0
    
    # Roll to match skymap convention
    oflr_data = np.roll(oflr_data, int(x_size*3/4), axis=1)
    oflr_data[~np.isnan(oflr_data)] = 1

    # Create colormap and set NaN color
    cmap = matplotlib.colormaps['Greys']
    cmap.set_bad(color=NScl)
    
    # Create coordinate meshes from original longitude/latitude arrays
    # These match the data before rolling - use the actual projection coordinates
    lon_mesh_orig, lat_mesh_orig = np.meshgrid(longitude_deg, latitude_deg)
    
    
    lon_mesh = lon_mesh_orig.copy() 
    lat_mesh = 90 - lat_mesh_orig.copy() 
    
    # Panel-level kwargs must not reach AxesImage.
    imshow_kwargs.pop("em_overlay", None)

    # Plot with imshow
    im = ax.imshow(
        oflr_data,
        aspect='auto',
        vmin=0,
        vmax=3,
        interpolation='nearest',
        cmap=cmap,
        extent=(0,360,180,0),
        origin='upper',
        **imshow_kwargs
    )
    
    # Plot contours using original categorical values (optional)
    if plot_contour:
        unique_levels = np.unique(oflr_data_for_contour[~np.isnan(oflr_data_for_contour)])
        if len(unique_levels) > 0:
            ax.contour(
                lon_mesh,
                lat_mesh,
                oflr_data_for_contour,
                levels=[1],
                colors=['r'],
                linewidths=0.5,
                zorder=10,
            )
    ax.set_ylim(180, 0)
    """
    phi_grid = np.arange(0, 360, 180)
    phi_grid_labels = np.roll(phi_grid, int(len(phi_grid)*0.75))
    # Add 90 at the right end (position 360)
    phi_grid = np.append(phi_grid, 360)
    phi_grid_labels = np.append(phi_grid_labels, 90)
    ax.set_xticks(phi_grid, [f"{label}$\degree$" for label in phi_grid_labels])
    ax.set_yticks(np.arange(0, 181, 30), [f"{tick}$\degree$" for tick in np.arange(0, 181, 30)])
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r'$\phi$')
    """
    ax.grid(True, alpha=0.3)
    
    overlay_observer_los(ax, theta_obs, color=cl_LOS, linewidth=lw_LOS)

    return fig, ax, im


def plot_OFLR_em_map(
    oflr_em_data: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    figure_size: Tuple[float, float] = (10, 5),
    dpi: int = 150,
    ax=None,
    NScl: str = 'LightCyan',
    theta_obs: Optional[float] = None,
    cl_LOS: str = 'b',
    lw_LOS: float = 0.7,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **imshow_kwargs
) -> Tuple:
    """Plot OFLR emission map using imshow.
    
    Parameters
    ----------
    oflr_em_data : np.ndarray
        OFLR emission map image data (2D array).
    longitude_deg : np.ndarray
        Longitude coordinates in degrees.
    latitude_deg : np.ndarray
        Latitude coordinates in degrees.
    figure_size : Tuple[float, float], optional
        Figure size (width, height). Default is (10, 5).
    dpi : int, optional
        Figure resolution. Default is 150.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    NScl : str, optional
        Color for NaN values. Default is 'LightCyan'.
    theta_obs : Optional[float], optional
        Observer angle in degrees. Default is None.
    cl_LOS : str, optional
        Color for line of sight. Default is 'b'.
    lw_LOS : float, optional
        Line width for line of sight. Default is 1.0.
    vmin : Optional[float], optional
        Minimum value for colormap. If None, uses data minimum.
    vmax : Optional[float], optional
        Maximum value for colormap. If None, uses data maximum.
    **imshow_kwargs
        Additional keyword arguments passed to imshow.
    
    Returns
    -------
    Tuple
        (fig, ax, im) tuple.
    """
    # Create or use existing axes
    if ax is None:
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    
    x_size, y_size = oflr_em_data.shape
    
    # Roll to match skymap convention
    oflr_em_data = np.roll(oflr_em_data, int(x_size*3/4), axis=1)
    oflr_em_data[oflr_em_data==0] = np.nan
    
    # Apply HSV colormap with 10-degree cycles
    # Transform colatitude values: each 10 degrees maps to full HSV cycle [0, 1]
    cycle_period = 10.0
    oflr_em_data_normalized = np.full_like(oflr_em_data, np.nan, dtype=float)
    valid_mask = ~np.isnan(oflr_em_data)
    oflr_em_data_normalized[valid_mask] = (oflr_em_data[valid_mask] % cycle_period) / cycle_period
    
    # Create colormap and set NaN to transparent
    cmap = matplotlib.colormaps['hsv'].copy()
    cmap.set_bad(color='None')  # Transparent for bad values
    
    # Plot with imshow
    im = ax.imshow(
        oflr_em_data_normalized,
        aspect='auto',
        vmin=0,
        vmax=1,
        interpolation='nearest',
        cmap=cmap,
        extent=(0,360,180,0),
        origin='upper',
        zorder=5,
        **imshow_kwargs
    )
    
    ax.set_ylim(180, 0)
    phi_grid = [0, 90, 180, 270, 360]
    phi_grid_labels = [90, 180, 270, 0, 90]
    # Add 90 at the right end (position 360)
    ax.set_xticks(phi_grid, [f"{label}$\degree$" for label in phi_grid_labels])
    ax.set_yticks(np.arange(0, 181, 30), [f"{tick}$\degree$" for tick in np.arange(0, 181, 30)])
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r'$\phi$')
    
    overlay_observer_los(ax, theta_obs, color=cl_LOS, linewidth=lw_LOS)

    return fig, ax, im


def overlay_oflr_emission_scatter(
    ax,
    oflr_em_map: np.ndarray,
    nside: int,
    *,
    phi_roll_deg: float = 270.0,
    cycle_period: float = 10.0,
    s: float = 0.8,
    alpha: float = 0.95,
    zorder: int = 6,
):
    """Rainbow scatter overlay for active OFLR footpoints (HSV, 10° hue cycles).

    The vertical coordinate is HEALPix colatitude in degrees, the same axis as
    ``overlay_observer_los``. It is not astronomical latitude.
    """
    em_map = np.asarray(oflr_em_map, dtype=np.float64)
    if em_map.size == 0 or not np.any(np.isfinite(em_map)):
        return None

    valid = np.flatnonzero(np.isfinite(em_map))
    if valid.size == 0:
        return None

    theta, phi = hp.pix2ang(int(nside), valid)
    lon_deg = np.mod(np.rad2deg(phi) + float(phi_roll_deg), 360.0)
    colatitude_deg = np.rad2deg(theta)
    vals = em_map[valid]
    cvals = np.mod(vals, float(cycle_period)) / float(cycle_period)
    return ax.scatter(
        lon_deg,
        colatitude_deg,
        c=cvals,
        cmap="hsv",
        vmin=0.0,
        vmax=1.0,
        s=float(s),
        alpha=float(alpha),
        edgecolors="none",
        zorder=int(zorder),
    )


def plot_healpix_scatter(
    oflr_em_map: np.ndarray,
    nside: int,
    ax=None,
    cmap='prism',
    vmin: float = 0.0,
    vmax: float = 180.0,
    s: float = 1.0,
    alpha: float = 1.0,
    figure_size: Tuple[float, float] = (10, 5),
    dpi: int = 150,
) -> Tuple:
    """Plot healpix map as scatter plot with non-NaN values.
    
    Parameters
    ----------
    oflr_em_map : np.ndarray
        Healpix map array (indexed by healpix pixel).
    nside : int
        Healpix nside parameter.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str, optional
        Colormap name. Default is 'prism'.
    vmin : float, optional
        Minimum value for colormap. Default is 0.0.
    vmax : float, optional
        Maximum value for colormap. Default is 180.0.
    s : float, optional
        Scatter point size. Default is 1.0.
    alpha : float, optional
        Scatter point transparency. Default is 1.0.
    figure_size : Tuple[float, float], optional
        Figure size (width, height). Default is (10, 5).
    dpi : int, optional
        Figure resolution. Default is 150.
    
    Returns
    -------
    Tuple
        (fig, ax, scatter) tuple.
    """
    # Create or use existing axes
    if ax is None:
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    
    # Get all healpix pixel indices
    npix = len(oflr_em_map)
    pixel_indices = np.arange(npix)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(oflr_em_map)
    valid_pixels = pixel_indices[valid_mask]
    valid_values = oflr_em_map[valid_mask]
    
    if len(valid_pixels) == 0:
        logger.warning("No valid (non-NaN) values in OFLR_em_map")
        return fig, ax, None
    
    # Convert healpix pixel indices to spherical coordinates
    colatitude, longitude = hp.pixelfunc.pix2ang(nside, valid_pixels)
    
    # Convert to degrees
    colatitude_deg = np.rad2deg(colatitude)
    longitude_deg = np.rad2deg(longitude)
    
    # Normalize values for colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]
    colors = cmap_obj(norm(valid_values))
    
    longitude_deg += 270
    longitude_deg[longitude_deg>360] -= 360
    longitude_deg[longitude_deg<0] += 360
    # Plot scatter
    scatter = ax.scatter(
        longitude_deg,
        colatitude_deg,
        c=valid_values,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        s=s,
        alpha=alpha,
        edgecolors='none',
    )
    
    #ax.set_xlabel(r'$\phi,\,^\circ$')
    #ax.set_ylabel(r'$\theta,\,^\circ$')
    
    phi_grid = np.arange(0, 360, 90)
    phi_grid_labels = np.roll(phi_grid, 3)
    # Add 90 at the right end (position 360)
    phi_grid = np.append(phi_grid, 360)
    phi_grid_labels = np.append(phi_grid_labels, 90)
    ax.set_xticks(phi_grid, phi_grid_labels)
    ax.set_xlim(0, 360)
    ax.set_ylim(180, 0)
    #ax.grid(True, alpha=0.3)
    
    return fig, ax, scatter

