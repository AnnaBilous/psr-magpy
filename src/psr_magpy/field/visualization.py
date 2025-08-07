import logging
from typing import Optional, Tuple, Literal
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from psr_magpy.field.multipole import MultipoleField
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class PlotPlane(Enum):
    """Enumeration for 2D plot planes."""

    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class FieldVisualizer:
    """
    Visualization tools for magnetic field diagnostics.

    Provides methods for creating quiver plots, contour plots, and other
    visualizations of magnetic field behavior.

    Examples
    --------
    >>> from psr_magpy.field.components import DipoleFieldComponent
    >>> from psr_magpy.field.multipole import MultipoleField
    >>> field = MultipoleField([DipoleFieldComponent()])
    >>> visualizer = FieldVisualizer()
    >>> visualizer.quiver_2d(field, x_range=(-3, 3), y_range=(-3, 3), plane='xy')
    """

    def __init__(self):
        """Initialize FieldVisualizer."""
        logger.debug("Initializing FieldVisualizer")

    def quiver_2d(
        self,
        field: MultipoleField,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        n_points: int = 20,
        plane: Literal["xy", "yz", "xz"] = "xy",
        fixed_coord: float = 0.0,
        normalize: bool = True,
        skip_points: int = 1,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show_magnitude: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create 2D quiver plot of magnetic field.

        Parameters
        ----------
        field : MultipoleField
            Field to visualize
        x_range : Tuple[float, float]
            Range for first coordinate
        y_range : Tuple[float, float]
            Range for second coordinate
        n_points : int
            Number of grid points per dimension
        plane : str
            Plot plane ('xy', 'yz', 'xz')
        fixed_coord : float
            Value for the fixed coordinate
        normalize : bool
            Whether to normalize field vectors
        skip_points : int
            Skip every nth point for clarity
        figsize : Tuple[int, int]
            Figure size
        title : str, optional
            Plot title
        show_magnitude : bool
            Color arrows by field magnitude
        **kwargs
            Additional arguments passed to plt.quiver

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        logger.info(f"Creating 2D quiver plot on {plane} plane")

        # Create grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        # Set up coordinate arrays based on plane
        if plane == "xy":
            coords = [X, Y, np.full_like(X, fixed_coord)]
            xlabel, ylabel = "X", "Y"
        elif plane == "yz":
            coords = [np.full_like(X, fixed_coord), X, Y]
            xlabel, ylabel = "Y", "Z"
        elif plane == "xz":
            coords = [X, np.full_like(X, fixed_coord), Y]
            xlabel, ylabel = "X", "Z"
        else:
            error_msg = f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate field at each grid point
        field_vectors = np.zeros((X.shape[0], X.shape[1], 3))
        field_magnitudes = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array(
                    [coords[0][i, j], coords[1][i, j], coords[2][i, j]],
                    dtype=np.float32,
                )
                field_vec = field.get_total_field(pos, normalize=normalize)
                field_vectors[i, j] = field_vec
                field_magnitudes[i, j] = np.linalg.norm(field_vec)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Get vector components for the plane
        if plane == "xy":
            u, v = field_vectors[:, :, 0], field_vectors[:, :, 1]
        elif plane == "yz":
            u, v = field_vectors[:, :, 1], field_vectors[:, :, 2]
        else:  # xz
            u, v = field_vectors[:, :, 0], field_vectors[:, :, 2]

        # Create quiver plot
        if show_magnitude:
            quiver = ax.quiver(
                X[::skip_points, ::skip_points],
                Y[::skip_points, ::skip_points],
                u[::skip_points, ::skip_points],
                v[::skip_points, ::skip_points],
                field_magnitudes[::skip_points, ::skip_points],
                cmap="viridis",
                **kwargs,
            )
            plt.colorbar(quiver, ax=ax, label="Field Magnitude")
        else:
            ax.quiver(
                X[::skip_points, ::skip_points],
                Y[::skip_points, ::skip_points],
                u[::skip_points, ::skip_points],
                v[::skip_points, ::skip_points],
                **kwargs,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is None:
            title = f"Magnetic Field on {plane.upper()} Plane ({plane[0].upper()}={fixed_coord})"
        ax.set_title(title)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        logger.debug("2D quiver plot created successfully")
        return fig, ax

    def magnitude_contour(
        self,
        field: MultipoleField,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        n_points: int = 50,
        plane: Literal["xy", "yz", "xz"] = "xy",
        fixed_coord: float = 0.0,
        levels: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create contour plot of field magnitude.

        Parameters
        ----------
        field : MultipoleField
            Field to visualize
        x_range : Tuple[float, float]
            Range for first coordinate
        y_range : Tuple[float, float]
            Range for second coordinate
        n_points : int
            Number of grid points per dimension
        plane : str
            Plot plane ('xy', 'yz', 'xz')
        fixed_coord : float
            Value for the fixed coordinate
        levels : int
            Number of contour levels
        figsize : Tuple[int, int]
            Figure size
        title : str, optional
            Plot title
        **kwargs
            Additional arguments passed to plt.contourf

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        logger.info(f"Creating magnitude contour plot on {plane} plane")

        # Create grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        # Set up coordinate arrays based on plane
        if plane == "xy":
            coords = [X, Y, np.full_like(X, fixed_coord)]
            xlabel, ylabel = "X", "Y"
        elif plane == "yz":
            coords = [np.full_like(X, fixed_coord), X, Y]
            xlabel, ylabel = "Y", "Z"
        elif plane == "xz":
            coords = [X, np.full_like(X, fixed_coord), Y]
            xlabel, ylabel = "X", "Z"
        else:
            error_msg = f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate field magnitude at each grid point
        field_magnitudes = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array(
                    [coords[0][i, j], coords[1][i, j], coords[2][i, j]],
                    dtype=np.float32,
                )
                field_vec = field.get_total_field(pos, normalize=False)
                field_magnitudes[i, j] = np.linalg.norm(field_vec)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        contour = ax.contourf(
            X, Y, field_magnitudes, levels=levels, cmap="viridis", **kwargs
        )
        plt.colorbar(contour, ax=ax, label="Field Magnitude")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is None:
            title = f"Field Magnitude on {plane.upper()} Plane ({plane[0].upper()}={fixed_coord})"
        ax.set_title(title)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        logger.debug("Magnitude contour plot created successfully")
        return fig, ax

    def quiver_3d(
        self,
        field: MultipoleField,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        z_range: Tuple[float, float] = (-2, 2),
        n_points: int = 8,
        normalize: bool = True,
        skip_points: int = 1,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        show_magnitude: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Create 3D quiver plot of magnetic field.

        Parameters
        ----------
        field : MultipoleField
            Field to visualize
        x_range : Tuple[float, float]
            Range for x coordinate
        y_range : Tuple[float, float]
            Range for y coordinate
        z_range : Tuple[float, float]
            Range for z coordinate
        n_points : int
            Number of grid points per dimension
        normalize : bool
            Whether to normalize field vectors
        skip_points : int
            Skip every nth point for clarity
        figsize : Tuple[int, int]
            Figure size
        title : str, optional
            Plot title
        show_magnitude : bool
            Color arrows by field magnitude
        **kwargs
            Additional arguments passed to ax.quiver

        Returns
        -------
        Tuple[plt.Figure, Axes3D]
            Figure and 3D axes objects
        """
        logger.info("Creating 3D quiver plot")

        # Create 3D grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        z = np.linspace(z_range[0], z_range[1], n_points)
        X, Y, Z = np.meshgrid(x, y, z)

        # Calculate field at each grid point
        field_vectors = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))
        field_magnitudes = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    pos = np.array(
                        [X[i, j, k], Y[i, j, k], Z[i, j, k]], dtype=np.float32
                    )
                    field_vec = field.get_total_field(pos, normalize=normalize)
                    field_vectors[i, j, k] = field_vec
                    field_magnitudes[i, j, k] = np.linalg.norm(field_vec)

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Apply skip_points to all dimensions
        X_skip = X[::skip_points, ::skip_points, ::skip_points]
        Y_skip = Y[::skip_points, ::skip_points, ::skip_points]
        Z_skip = Z[::skip_points, ::skip_points, ::skip_points]
        U_skip = field_vectors[::skip_points, ::skip_points, ::skip_points, 0]
        V_skip = field_vectors[::skip_points, ::skip_points, ::skip_points, 1]
        W_skip = field_vectors[::skip_points, ::skip_points, ::skip_points, 2]
        M_skip = field_magnitudes[::skip_points, ::skip_points, ::skip_points]

        # Create 3D quiver plot
        if show_magnitude:
            quiver = ax.quiver(
                X_skip,
                Y_skip,
                Z_skip,
                U_skip,
                V_skip,
                W_skip,
                M_skip,
                cmap="viridis",
                **kwargs,
            )
            plt.colorbar(quiver, ax=ax, label="Field Magnitude")
        else:
            ax.quiver(X_skip, Y_skip, Z_skip, U_skip, V_skip, W_skip, **kwargs)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if title is None:
            title = "3D Magnetic Field"
        ax.set_title(title)

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        logger.debug("3D quiver plot created successfully")
        return fig, ax

    def quick_diagnostic(
        self, field: MultipoleField, plane: Literal["xy", "yz", "xz"] = "xy", **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Quick diagnostic plot combining quiver and magnitude contour.

        Parameters
        ----------
        field : MultipoleField
            Field to visualize
        plane : str
            Plot plane ('xy', 'yz', 'xz')
        **kwargs
            Additional arguments passed to quiver_2d

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        logger.info(f"Creating quick diagnostic plot on {plane} plane")

        # Create magnitude contour as background
        fig, ax = self.magnitude_contour(field, plane=plane, **kwargs)

        # Add quiver plot on top
        x_range = kwargs.get("x_range", (-3, 3))
        y_range = kwargs.get("y_range", (-3, 3))
        n_points = kwargs.get("n_points", 20)
        skip_points = kwargs.get("skip_points", 2)

        # Create grid for quiver
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        # Set up coordinate arrays based on plane
        if plane == "xy":
            coords = [X, Y, np.full_like(X, 0.0)]
        elif plane == "yz":
            coords = [np.full_like(X, 0.0), X, Y]
        else:  # xz
            coords = [X, np.full_like(X, 0.0), Y]

        # Calculate field vectors for quiver
        field_vectors = np.zeros((X.shape[0], X.shape[1], 3))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array(
                    [coords[0][i, j], coords[1][i, j], coords[2][i, j]],
                    dtype=np.float32,
                )
                field_vec = field.get_total_field(pos, normalize=True)
                field_vectors[i, j] = field_vec

        # Get vector components for the plane
        if plane == "xy":
            u, v = field_vectors[:, :, 0], field_vectors[:, :, 1]
        elif plane == "yz":
            u, v = field_vectors[:, :, 1], field_vectors[:, :, 2]
        else:  # xz
            u, v = field_vectors[:, :, 0], field_vectors[:, :, 2]

        # Add quiver plot
        ax.quiver(
            X[::skip_points, ::skip_points],
            Y[::skip_points, ::skip_points],
            u[::skip_points, ::skip_points],
            v[::skip_points, ::skip_points],
            color="white",
            alpha=0.7,
            linewidth=1,
        )

        title = f"Field Diagnostic on {plane.upper()} Plane"
        ax.set_title(title)

        logger.debug("Quick diagnostic plot created successfully")
        return fig, ax
