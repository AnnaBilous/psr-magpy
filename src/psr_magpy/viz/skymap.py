"""Equirectangular skymap panel plotting."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from psr_magpy.viz.los import overlay_observer_los

def plot_skymap(
    skymap_data: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figure_size: Tuple[float, float] = (10, 5),
    dpi: int = 150,
    ax=None,
    theta_obs: Optional[float] = None,
    cl_LOS: str = 'b',
    lw_LOS: float = 1.0,
    **imshow_kwargs
) -> Tuple:
    """Plot skymap using imshow.
    
    Parameters
    ----------
    skymap_data : np.ndarray
        Skymap image data (2D array).
    longitude_deg : np.ndarray
        Longitude coordinates in degrees.
    latitude_deg : np.ndarray
        Latitude coordinates in degrees.
    vmin : Optional[float], optional
        Minimum value for colormap. If None, uses data minimum.
    vmax : Optional[float], optional
        Maximum value for colormap. If None, uses data maximum.
    figure_size : Tuple[float, float], optional
        Figure size (width, height). Default is (10, 5).
    dpi : int, optional
        Figure resolution. Default is 150.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
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
    
    x_size, y_size = skymap_data.shape
    # Determine extent from longitude and latitude
    skymap_data = np.roll(skymap_data, int(x_size*3/4), axis=1)
    
    # Plot with imshow
    im = ax.imshow(
        skymap_data,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        extent=(0,360,180,0),
        **imshow_kwargs
    )
    

    overlay_observer_los(ax, theta_obs, color=cl_LOS, linewidth=lw_LOS)

    phi_grid = np.arange(0, 360, 180)
    phi_grid_labels = np.roll(phi_grid, int(len(phi_grid)*0.75))
    # Add 90 at the right end (position 360)
    phi_grid = np.append(phi_grid, 360)
    phi_grid_labels = np.append(phi_grid_labels, 90)
    ax.set_xticks(phi_grid, [f"{label}$\degree$" for label in phi_grid_labels])
    ax.set_yticks(np.arange(0, 181, 30), [f"{tick}$\degree$" for tick in np.arange(0, 181, 30)])
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r'$\phi$')
    ax.set_xticks([])
    return fig, ax, im


