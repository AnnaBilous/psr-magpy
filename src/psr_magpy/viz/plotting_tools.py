"""Plotting tools for field line visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict, Callable
from numpy.typing import NDArray


def get_Nlayers(XZview: float, YtoX: float, K: float) -> int:
    """Calculate number of layers for field line plotting.

    Parameters
    ----------
    XZview : float
        View extent in XZ plane.
    YtoX : float
        Ratio of Y to X view extent.
    K : float
        Layer density factor.

    Returns
    -------
    int
        Number of layers.
    """
    Yview = XZview * YtoX
    Nlayers = int(2 * Yview * K)
    return Nlayers


def initialize_subplot(XZview: float, ZtoX: float) -> plt.Axes:
    """Initialize subplot for field line visualization.

    Parameters
    ----------
    XZview : float
        View extent in XZ plane.
    ZtoX : float
        Ratio of Z to X view extent.

    Returns
    -------
    plt.Axes
        Initialized axes object.
    """
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(-XZview, XZview)
    ax.set_ylim(-ZtoX * XZview, ZtoX * XZview)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def initialize_lines(
    ax: plt.Axes,
    linewidth_func: Callable,
    zorder_layer: Dict[int, int],
    Nlayers: int,
) -> List[LineCollection]:
    """Initialize line collections for field line plotting.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    linewidth_func : Callable
        Function that takes (layer_no, Nlayers) and returns line width.
    zorder_layer : Dict[int, int]
        Dictionary mapping layer number to z-order.
    Nlayers : int
        Total number of layers.

    Returns
    -------
    List[LineCollection]
        List of initialized line collections.
    """
    cur_line = []
    for k in zorder_layer:
        line_layer = LineCollection(
            np.empty((0, 2, 2)),
            zorder=zorder_layer[k],
            linewidths=linewidth_func(k, Nlayers),
            antialiaseds=1,
            alpha=0.8,
            capstyle="round",
            rasterized=True,
        )
        ax.add_collection(line_layer)
        cur_line.append(line_layer)

    return cur_line


def get_line_projection(
    FLs: List[NDArray], phi_rot: float, theta_rot: float
) -> List[NDArray]:
    """Project field lines using rotation angles.

    Parameters
    ----------
    FLs : List[NDArray]
        List of field line coordinate arrays (shape: (3, N)).
    phi_rot : float
        Longitude rotation angle in degrees.
    theta_rot : float
        Latitude rotation angle in degrees.

    Returns
    -------
    List[NDArray]
        List of projected field line coordinate arrays.
    """
    rotation_longitude = R.from_rotvec([np.deg2rad(phi_rot) * np.array([0, 0, 1])])
    rotation_latitude = R.from_rotvec([np.deg2rad(theta_rot) * np.array([1, 0, 0])])
    r = rotation_latitude * rotation_longitude

    FLs_proj = [r.apply(FL).T for FL in FLs]

    return FLs_proj


def prepare_lines_for_plotting(
    FLs: List[NDArray],
    phi_rot: float,
    theta_rot: float,
    XZview: float,
    YtoX: float,
    K: float,
) -> Tuple[NDArray, NDArray]:
    """Prepare field lines for plotting with projection and z-ordering.

    Parameters
    ----------
    FLs : List[NDArray]
        List of field line coordinate arrays (shape: (3, N)).
    phi_rot : float
        Longitude rotation angle in degrees.
    theta_rot : float
        Latitude rotation angle in degrees.
    XZview : float
        View extent in XZ plane.
    YtoX : float
        Ratio of Y to X view extent.
    K : float
        Layer density factor.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of (segments_all, zorder_all) where segments_all contains line segments
        and zorder_all contains z-order values for each segment.
    """
    # Project all field lines
    FLs_proj = get_line_projection(FLs, phi_rot, theta_rot)

    segments_list = []
    y_all = np.array([])

    for Xproj, Yproj, Zproj in FLs_proj:
        # Compute segments and midpoints
        points = np.stack([Xproj, Zproj], axis=1).reshape(-1, 1, 2)
        segments_list.append(np.concatenate([points[:-1], points[1:]], axis=1))
        y_all = np.append(y_all, 0.5 * (Yproj[1:] + Yproj[:-1]))

    # Combine segments and compute z-order
    segments_all = np.vstack(segments_list)
    zorder_all = ((y_all + XZview * YtoX) * K).astype(int)

    return segments_all, zorder_all


def compute_fraction_of_line_length(FLs: List[NDArray]) -> NDArray:
    """Compute fraction of line length along each field line.

    Parameters
    ----------
    FLs : List[NDArray]
        List of field line coordinate arrays (shape: (3, N)).

    Returns
    -------
    NDArray
        Array of fractional line lengths for each segment.
    """
    l_all = np.array([])

    for FL in FLs:
        X, Y, Z = FL.T
        R = np.sqrt(X[0] ** 2 + Y[0] ** 2 + Z[0] ** 2)
        R_next = np.sqrt(X[1] ** 2 + Y[1] ** 2 + Z[1] ** 2)

        # Determine direction based on radial distance
        l_min, l_max = (0, 1) if R_next > R else (1, 0)
        l_all = np.append(l_all, np.linspace(l_min, l_max, len(Y) - 1))

    return l_all


def fill_lines(
    lines: List[LineCollection],
    segments_all: NDArray,
    zorder_all: NDArray,
    color_param_all: NDArray,
    zorder_layer: Dict[int, int],
    colormap: Callable,
) -> List[LineCollection]:
    """Fill line collections with segments and colors.

    Parameters
    ----------
    lines : List[LineCollection]
        List of line collection objects.
    segments_all : NDArray
        Array of line segments.
    zorder_all : NDArray
        Array of z-order values for each segment.
    color_param_all : NDArray
        Array of color parameter values for each segment.
    zorder_layer : Dict[int, int]
        Dictionary mapping layer number to z-order.
    colormap : Callable
        Colormap function that maps color parameters to colors.

    Returns
    -------
    List[LineCollection]
        Updated line collections with segments and colors.
    """
    colors = colormap(color_param_all)

    # Update line collection data
    for k in zorder_layer:
        mask = zorder_all == k
        lines[k].set_segments(segments_all[mask])
        segment_colors = colors[mask]
        lines[k].set_color(segment_colors)

    return lines

