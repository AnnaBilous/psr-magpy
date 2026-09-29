"""2D field visualization module."""
# Not used now, for example purposes. 

from typing import Optional, Tuple, Dict, Any, Union, Literal, TYPE_CHECKING
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
# Ensure we use matplotlib's mpl_toolkits, not system namespace package
import sys
import os
_matplotlib_path = os.path.dirname(matplotlib.__file__)
_mpl_toolkits_path = os.path.join(_matplotlib_path, 'mpl_toolkits')
if _mpl_toolkits_path not in sys.path:
    sys.path.insert(0, _mpl_toolkits_path)
from mpl_toolkits.mplot3d import Axes3D

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyFieldError

if TYPE_CHECKING:
    from psr_magpy.field.multipole import MultipoleField

logger = get_logger(__name__, console_format_brief=True)

# Type alias for plot results
PlotResult = Tuple[Figure, Axes]


class Base2DFieldPlotter:
    """Base class for 2D field plotters."""

    def __init__(self, **defaults) -> None:
        """Initialize with default configuration."""
        self.config: Dict[str, Any] = {
            "title": "Vector Field",
            "xlabel": "x",
            "ylabel": "y",
            "grid": True,
            "grid_alpha": 0.3,
            "figsize": (6, 6),
            "dpi": 100,
            "circle_radius": 1.0,
            "circle_color": "Bisque",
            "circle_alpha": 0.9,
            "circle_fill": True,
            **defaults,
        }

    def _setup_figure_and_axes(
        self, ax: Optional[Axes], plot_config: Dict[str, Any]
    ) -> Tuple[Figure, Axes]:
        """Create figure/axes if not provided."""
        if ax is None:
            fig = plt.figure(figsize=plot_config["figsize"], dpi=plot_config["dpi"])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        return fig, ax

    def _add_circle(self, ax: Axes, plot_config: Dict[str, Any]) -> None:
        """Add circular patch if radius > 0."""
        if plot_config["circle_radius"] > 0:
            circle = Circle(
                (0, 0),
                plot_config["circle_radius"],
                color=plot_config["circle_color"],
                alpha=plot_config["circle_alpha"],
                fill=plot_config["circle_fill"],
            )
            ax.add_patch(circle)

    def _setup_layout(self, ax: Axes, plot_config: Dict[str, Any]) -> None:
        """Set labels, title, and grid."""
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        ax.set_title(plot_config["title"])
        if plot_config["grid"]:
            ax.grid(True, alpha=plot_config["grid_alpha"])


class Quiver2DFieldPlotter(Base2DFieldPlotter):
    """2D quiver plotter for vector fields."""

    def __init__(self, **defaults) -> None:
        """Initialize with default configuration."""
        super().__init__(**defaults)
        self.config.update(
            {
                "scale": 20,
                "alpha": 0.7,
                "color": "Navy",
                "width": 0.002,
                "headwidth": 3,
                "headlength": 5,
                "minshaft": 1,
                "minlength": 1,
                "pivot": "middle",
                **defaults,
            }
        )

    def plot(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot 2D vector field using quiver."""
        plot_config = {**self.config, **kwargs}
        fig, ax = self._setup_figure_and_axes(ax, plot_config)

        quiver_params = {
            k: v
            for k, v in plot_config.items()
            if k
            in [
                "scale",
                "alpha",
                "color",
                "width",
                "headwidth",
                "headlength",
                "minshaft",
                "minlength",
                "pivot",
            ]
        }

        ax.quiver(X, Y, U, V, **quiver_params)
        self._add_circle(ax, plot_config)
        self._setup_layout(ax, plot_config)

        return fig, ax


class Streamline2DFieldPlotter(Base2DFieldPlotter):
    """2D streamline plotter for vector fields."""

    def __init__(self, **defaults) -> None:
        """Initialize with default configuration."""
        super().__init__(**defaults)
        self.config.update(
            {
                "density": 1.0,
                "linewidth": 1.0,
                "arrowsize": 1.5,
                "arrowstyle": "->",
                "start_points": None,
                "maxlength": 4.0,
                "integration_direction": "both",
                "color": "red",
                "alpha": 0.7,
                "title": "Streamlines",
                "figsize": (8, 6),
                **defaults,
            }
        )

    def plot(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot 2D vector field using streamplot."""
        plot_config = {**self.config, **kwargs}
        fig, ax = self._setup_figure_and_axes(ax, plot_config)

        streamplot_params = {
            k: v
            for k, v in plot_config.items()
            if k
            in [
                "density",
                "color",
                "linewidth",
                "arrowsize",
                "arrowstyle",
                "start_points",
                "maxlength",
                "integration_direction",
            ]
        }

        ax.streamplot(X, Y, U, V, **streamplot_params)
        self._add_circle(ax, plot_config)
        self._setup_layout(ax, plot_config)

        return fig, ax


class Grid3DVisualizer:
    """3D grid visualizer with coordinate system, sphere, and plane grid."""

    def __init__(self, **defaults) -> None:
        """Initialize with default configuration."""
        self.config: Dict[str, Any] = {
            "sphere_radius": 1.0,
            "sphere_alpha": 0.3,
            "sphere_color": "lightblue",
            "grid_color": "red",
            "grid_alpha": 0.7,
            "grid_size": 0.1,
            "axis_length": 2.0,
            "axis_width": 2,
            "title": "3D Grid Visualization",
            "figsize": (10, 8),
            "dpi": 100,
            "elevation": 20,
            "azimuth": 45,
            **defaults,
        }

    def plot_grid_on_plane(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        limits_u: Tuple[float, float],
        limits_v: Tuple[float, float],
        n_points_u: int,
        n_points_v: int,
        ax: Optional[Axes3D] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes3D]:
        """Plot 3D visualization with coordinate system, sphere, and plane grid."""
        plot_config = {**self.config, **kwargs}

        if ax is None:
            fig = plt.figure(figsize=plot_config["figsize"], dpi=plot_config["dpi"])
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        assert isinstance(ax, Axes3D)
        assert isinstance(fig, Figure)

        # Check if points define a valid plane
        if np.allclose(p1, p2) or np.allclose(p1, p3) or np.allclose(p2, p3):
            raise PSRMagPyFieldError("At least two points coincide. Cannot define a plane.")

        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        if np.allclose(cross_product, 0):
            raise PSRMagPyFieldError("Points are collinear. Cannot define a plane.")

        # Calculate plane basis vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        v2_perp = v2_norm - np.dot(v2_norm, v1_norm) * v1_norm
        v2_perp = v2_perp / np.linalg.norm(v2_perp)

        # Create grid in plane coordinates
        u = np.linspace(limits_u[0], limits_u[1], n_points_u)
        v = np.linspace(limits_v[0], limits_v[1], n_points_v)
        U, V = np.meshgrid(u, v)

        # Convert to 3D coordinates
        X_3d = np.zeros((n_points_v, n_points_u, 3))
        for i in range(n_points_v):
            for j in range(n_points_u):
                X_3d[i, j] = p1 + U[i, j] * v1_norm + V[i, j] * v2_perp

        # Plot coordinate axes
        axis_length = plot_config["axis_length"]
        ax.quiver(
            0,
            0,
            0,
            axis_length,
            0,
            0,
            color="red",
            arrow_length_ratio=0.1,
            linewidth=plot_config["axis_width"],
        )
        ax.quiver(
            0,
            0,
            0,
            0,
            axis_length,
            0,
            color="green",
            arrow_length_ratio=0.1,
            linewidth=plot_config["axis_width"],
        )
        ax.quiver(
            0,
            0,
            0,
            0,
            0,
            axis_length,
            color="blue",
            arrow_length_ratio=0.1,
            linewidth=plot_config["axis_width"],
        )

        # Add axis labels
        ax.text(axis_length * 1.1, 0, 0, "X", color="red", fontsize=12)
        ax.text(0, axis_length * 1.1, 0, "Y", color="green", fontsize=12)
        ax.text(0, 0, axis_length * 1.1, "Z", color="blue", fontsize=12)

        # Plot sphere
        sphere_radius = plot_config["sphere_radius"]
        u_sphere = np.linspace(0, 2 * np.pi, 20)
        v_sphere = np.linspace(0, np.pi, 20)
        x_sphere = sphere_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y_sphere = sphere_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z_sphere = sphere_radius * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

        ax.plot_surface(
            x_sphere,
            y_sphere,
            z_sphere,
            alpha=plot_config["sphere_alpha"],
            color=plot_config["sphere_color"],
        )

        # Plot grid points on plane
        grid_size = plot_config["grid_size"]
        for i in range(n_points_v):
            for j in range(n_points_u):
                point = X_3d[i, j]
                ax.scatter(
                    point[0],
                    point[1],
                    point[2],
                    c=plot_config["grid_color"],
                    alpha=plot_config["grid_alpha"],
                    s=grid_size * 1000,
                )

        # Set view and labels
        ax.view_init(elev=plot_config["elevation"], azim=plot_config["azimuth"])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(plot_config["title"])

        # Set equal aspect ratio
        ax.set_box_aspect((1, 1, 1))

        return fig, ax


def plot_field_lines_2d(
    field,
    plane: Literal["xy", "yz", "xz"],
    fixed_coord: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int = 100,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """Plot field lines in 2D plane.

    Parameters
    ----------
    field : MultipoleField
        Field object.
    plane : str
        Plane to plot in ("xy", "yz", or "xz").
    fixed_coord : float
        Value of fixed coordinate.
    x_range : Tuple[float, float]
        Range for x-axis.
    y_range : Tuple[float, float]
        Range for y-axis.
    n_points : int, optional
        Number of points per dimension. Default is 100.
    ax : Optional[Axes], optional
        Optional matplotlib axes. Default is None.
    **kwargs : Any
        Additional plotting arguments.

    Returns
    -------
    Axes
        Matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)

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
        raise PSRMagPyFieldError(f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'")

    # Calculate field at each point
    field_values = np.zeros((n_points, n_points, 3))
    for i in range(n_points):
        for j in range(n_points):
            field_values[i, j] = field.get_total_field(
                [coords[0][i, j], coords[1][i, j], coords[2][i, j]]
            )

    # Plot streamlines
    ax.streamplot(X, Y, field_values[:, :, 0], field_values[:, :, 1], **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    return ax


class MultipoleFieldVisualizer:
    """Visualizer for MultipoleField instances.
    
    Separates visualization logic from field computation, allowing MultipoleField
    to focus solely on field calculations without matplotlib dependencies.
    
    Parameters
    ----------
    field : MultipoleField
        The multipole field instance to visualize
    """
    
    def __init__(self, field: "MultipoleField"):
        """Initialize visualizer with a MultipoleField instance."""
        self.field = field
        self._quiver_plotter: Optional[Quiver2DFieldPlotter] = None
        self._streamline_plotter: Optional[Streamline2DFieldPlotter] = None
    
    @property
    def quiver_plotter(self) -> Quiver2DFieldPlotter:
        """Get the quiver field plotter instance."""
        if self._quiver_plotter is None:
            self._quiver_plotter = Quiver2DFieldPlotter()
        return self._quiver_plotter
    
    @property
    def streamline_plotter(self) -> Streamline2DFieldPlotter:
        """Get the streamline field plotter instance."""
        if self._streamline_plotter is None:
            self._streamline_plotter = Streamline2DFieldPlotter()
        return self._streamline_plotter
    
    def _calculate_grid_field(
        self,
        plane_type: str = "standard",
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        n_points: int = 20,
        plane: str = "xy",
        fixed_coord: float = 0.0,
        normalize: bool = True,
        # For arbitrary planes
        p1: Optional[NDArray[np.float32]] = None,
        p2: Optional[NDArray[np.float32]] = None,
        p3: Optional[NDArray[np.float32]] = None,
        limits_u: Optional[Tuple[float, float]] = None,
        limits_v: Optional[Tuple[float, float]] = None,
        n_points_u: Optional[int] = None,
        n_points_v: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
        """
        Unified method to calculate grid and field vectors for both standard and arbitrary planes.

        Parameters
        ----------
        plane_type : str
            Type of plane: "standard" for xy/yz/xz or "arbitrary" for 3D points
        x_range, y_range : Tuple[float, float]
            Range for coordinates (standard planes only)
        n_points : int
            Number of grid points per dimension (standard planes only)
        plane : str
            Plot plane ('xy', 'yz', 'xz') for standard planes
        fixed_coord : float
            Fixed coordinate for the third dimension (standard planes only)
        normalize : bool
            Whether to normalize field vectors
        p1, p2, p3 : NDArray[np.float32], optional
            Three 3D points defining arbitrary plane
        limits_u, limits_v : Tuple[float, float], optional
            Range for plane coordinates (arbitrary planes only)
        n_points_u, n_points_v : int, optional
            Number of grid points for each dimension (arbitrary planes only)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]
            X, Y (grid coordinates), U, V (field components), xlabel, ylabel
        """
        if plane_type == "standard":
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
                raise PSRMagPyFieldError(f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'")

            # Calculate field at each grid point
            U = np.zeros_like(X)
            V = np.zeros_like(Y)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pos = np.array(
                        [coords[0][i, j], coords[1][i, j], coords[2][i, j]],
                        dtype=np.float32,
                    )
                    field_vec = self.field.get_total_field(pos, normalize=normalize)

                    if plane == "xy":
                        U[i, j], V[i, j] = field_vec[0], field_vec[1]
                    elif plane == "yz":
                        U[i, j], V[i, j] = field_vec[1], field_vec[2]
                    else:  # xz
                        U[i, j], V[i, j] = field_vec[0], field_vec[2]

            return X, Y, U, V, xlabel, ylabel

        elif plane_type == "arbitrary":
            if p1 is None or p2 is None or p3 is None:
                raise PSRMagPyFieldError("p1, p2, p3 must be provided for arbitrary planes")
            if limits_u is None or limits_v is None:
                raise PSRMagPyFieldError("limits_u, limits_v must be provided for arbitrary planes")
            if n_points_u is None or n_points_v is None:
                raise PSRMagPyFieldError(
                    "n_points_u, n_points_v must be provided for arbitrary planes"
                )

            # Check if points coincide
            if np.allclose(p1, p2) or np.allclose(p1, p3) or np.allclose(p2, p3):
                raise PSRMagPyFieldError("At least two points coincide. Cannot define a plane.")

            # Calculate plane basis vectors
            v1 = p2 - p1  # First direction vector
            v2 = p3 - p1  # Second direction vector

            # Check if points are collinear (cross product should be non-zero)
            cross_product = np.cross(v1, v2)
            if np.allclose(cross_product, 0):
                raise PSRMagPyFieldError("Points are collinear. Cannot define a plane.")

            # Normalize basis vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            # Ensure v2 is perpendicular to v1 (Gram-Schmidt)
            v2_perp = v2_norm - np.dot(v2_norm, v1_norm) * v1_norm
            v2_perp = v2_perp / np.linalg.norm(v2_perp)

            # Create grid in plane coordinates
            u = np.linspace(limits_u[0], limits_u[1], n_points_u)
            v = np.linspace(limits_v[0], limits_v[1], n_points_v)
            U, V = np.meshgrid(u, v)

            # Convert to 3D coordinates
            X_3d = np.zeros((n_points_v, n_points_u, 3))
            for i in range(n_points_v):
                for j in range(n_points_u):
                    X_3d[i, j] = p1 + U[i, j] * v1_norm + V[i, j] * v2_perp

            # Calculate field at each grid point
            field_vectors = np.zeros((n_points_v, n_points_u, 3))
            for i in range(n_points_v):
                for j in range(n_points_u):
                    pos = X_3d[i, j].astype(np.float32)
                    field_vectors[i, j] = self.field.get_total_field(pos, normalize=normalize)

            # Project field vectors onto plane
            U_field = np.zeros((n_points_v, n_points_u))
            V_field = np.zeros((n_points_v, n_points_u))

            for i in range(n_points_v):
                for j in range(n_points_u):
                    field_vec = field_vectors[i, j]
                    U_field[i, j] = np.dot(field_vec, v1_norm)
                    V_field[i, j] = np.dot(field_vec, v2_perp)

            xlabel = f"Plane coord 1 ({limits_u[0]:.1f} to {limits_u[1]:.1f})"
            ylabel = f"Plane coord 2 ({limits_v[0]:.1f} to {limits_v[1]:.1f})"
            return U, V, U_field, V_field, xlabel, ylabel
        else:
            raise PSRMagPyFieldError(
                f"Invalid plane_type: {plane_type}. Must be 'standard' or 'arbitrary'"
            )
    
    def plot_2d_quiver(
        self,
        # Standard plane parameters
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        n_points: int = 20,
        plane: str = "xy",
        fixed_coord: float = 0.0,
        # Arbitrary plane parameters
        p1: Optional[NDArray[np.float32]] = None,
        p2: Optional[NDArray[np.float32]] = None,
        p3: Optional[NDArray[np.float32]] = None,
        limits_u: Optional[Tuple[float, float]] = None,
        limits_v: Optional[Tuple[float, float]] = None,
        n_points_u: Optional[int] = None,
        n_points_v: Optional[int] = None,
        # Common parameters
        normalize: bool = True,
        ax: Optional[Axes] = None,
        # Arrow properties
        scale: Optional[float] = None,
        width: Optional[float] = None,
        headwidth: Optional[float] = None,
        headlength: Optional[float] = None,
        minshaft: Optional[float] = None,
        minlength: Optional[float] = None,
        pivot: Optional[Literal["tail", "middle", "tip"]] = None,
        # Color properties
        color: Optional[Union[str, Tuple[float, float, float]]] = None,
        alpha: Optional[float] = None,
        # Circle properties
        circle_radius: Optional[float] = None,
        circle_color: Optional[Union[str, Tuple[float, float, float]]] = None,
        circle_alpha: Optional[float] = None,
        circle_fill: Optional[bool] = None,
        # Layout properties
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: Optional[bool] = None,
        grid_alpha: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        **kwargs,
    ) -> PlotResult:
        """
        Plot 2D vector field using quiver arrows with support for both standard and arbitrary planes.

        Parameters
        ----------
        Standard Plane Parameters:
        -------------------------
        x_range, y_range : Tuple[float, float]
            Range for coordinates (standard planes only)
        n_points : int
            Number of grid points per dimension (standard planes only)
        plane : str
            Plot plane ('xy', 'yz', 'xz') for standard planes
        fixed_coord : float
            Fixed coordinate for the third dimension (standard planes only)

        Arbitrary Plane Parameters:
        ---------------------------
        p1, p2, p3 : NDArray[np.float32], optional
            Three 3D points defining arbitrary plane
        limits_u, limits_v : Tuple[float, float], optional
            Range for plane coordinates (arbitrary planes only)
        n_points_u, n_points_v : int, optional
            Number of grid points for each dimension (arbitrary planes only)

        Common Parameters:
        ------------------
        normalize : bool
            Whether to normalize field vectors
        ax : Axes, optional
            Existing axes to plot on

        Arrow Properties:
        ----------------
        scale : float, optional
            Scale factor for arrow length. Larger values = shorter arrows.
            Range: 0.1 to 100, typical: 10-50
        width : float, optional
            Width of arrow shaft. Range: 0.001 to 0.01
        headwidth : float, optional
            Width of arrow head relative to shaft. Range: 1-10
        headlength : float, optional
            Length of arrow head relative to shaft. Range: 1-10
        minshaft : float, optional
            Minimum length of arrow shaft. Range: 0.1-5
        minlength : float, optional
            Minimum length of entire arrow. Range: 0.1-5
        pivot : {'tail', 'middle', 'tip'}, optional
            Arrow pivot point. Options: 'tail', 'middle', 'tip'

        Color Properties:
        -----------------
        color : str or tuple, optional
            Arrow color. Can be named colors, hex codes, RGB tuples, or single letters
        alpha : float, optional
            Arrow transparency. Range: 0.0 (transparent) to 1.0 (opaque)

        Circle Properties:
        ------------------
        circle_radius : float, optional
            Radius of reference circle. Set to 0 to disable. Typical: 0.5-3.0
        circle_color : str or tuple, optional
            Circle color. Same options as arrow colors
        circle_alpha : float, optional
            Circle transparency. Range: 0.0 to 1.0
        circle_fill : bool, optional
            Whether to fill circle (True) or just outline (False)

        Layout Properties:
        ------------------
        title : str, optional
            Plot title text
        xlabel : str, optional
            X-axis label text
        ylabel : str, optional
            Y-axis label text
        grid : bool, optional
            Whether to show background grid
        grid_alpha : float, optional
            Grid transparency. Range: 0.0 to 1.0
        figsize : tuple, optional
            Figure size as (width, height) in inches
        dpi : int, optional
            Figure resolution in dots per inch. Higher = sharper images

        **kwargs
            Additional arguments passed to plotter

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects

        Examples
        --------
        # Standard plane
        >>> from psr_magpy.viz import MultipoleFieldVisualizer
        >>> visualizer = MultipoleFieldVisualizer(field)
        >>> fig, ax = visualizer.plot_2d_quiver(plane="xy")

        # Arbitrary plane
        >>> p1 = np.array([0, 0, 0], dtype=np.float32)
        >>> p2 = np.array([1, 1, 0], dtype=np.float32)
        >>> p3 = np.array([0, 0, 1], dtype=np.float32)
        >>> fig, ax = visualizer.plot_2d_quiver(
        ...     p1=p1, p2=p2, p3=p3, limits_u=(-2, 2), limits_v=(-2, 2),
        ...     n_points_u=20, n_points_v=20
        ... )
        """
        # Determine plane type and calculate grid
        if p1 is not None and p2 is not None and p3 is not None:
            # Arbitrary plane
            if limits_u is None or limits_v is None:
                raise PSRMagPyFieldError(
                    "limits_u and limits_v must be provided for arbitrary planes"
                )
            if n_points_u is None or n_points_v is None:
                raise PSRMagPyFieldError(
                    "n_points_u and n_points_v must be provided for arbitrary planes"
                )

            X, Y, U, V, default_xlabel, default_ylabel = self._calculate_grid_field(
                plane_type="arbitrary",
                p1=p1,
                p2=p2,
                p3=p3,
                limits_u=limits_u,
                limits_v=limits_v,
                n_points_u=n_points_u,
                n_points_v=n_points_v,
                normalize=normalize,
            )
        else:
            # Standard plane
            X, Y, U, V, default_xlabel, default_ylabel = self._calculate_grid_field(
                plane_type="standard",
                x_range=x_range,
                y_range=y_range,
                n_points=n_points,
                plane=plane,
                fixed_coord=fixed_coord,
                normalize=normalize,
            )

        # Build plot kwargs from provided parameters
        plot_kwargs: Dict[str, Any] = {}
        
        # Arrow properties
        if scale is not None:
            plot_kwargs["scale"] = scale
        if width is not None:
            plot_kwargs["width"] = width
        if headwidth is not None:
            plot_kwargs["headwidth"] = headwidth
        if headlength is not None:
            plot_kwargs["headlength"] = headlength
        if minshaft is not None:
            plot_kwargs["minshaft"] = minshaft
        if minlength is not None:
            plot_kwargs["minlength"] = minlength
        if pivot is not None:
            plot_kwargs["pivot"] = pivot
        
        # Color properties
        if color is not None:
            plot_kwargs["color"] = color
        if alpha is not None:
            plot_kwargs["alpha"] = alpha
        
        # Circle properties
        if circle_radius is not None:
            plot_kwargs["circle_radius"] = circle_radius
        if circle_color is not None:
            plot_kwargs["circle_color"] = circle_color
        if circle_alpha is not None:
            plot_kwargs["circle_alpha"] = circle_alpha
        if circle_fill is not None:
            plot_kwargs["circle_fill"] = circle_fill
        
        # Layout properties
        if title is not None:
            plot_kwargs["title"] = title
        if xlabel is not None:
            plot_kwargs["xlabel"] = xlabel
        elif default_xlabel:
            plot_kwargs["xlabel"] = default_xlabel
        if ylabel is not None:
            plot_kwargs["ylabel"] = ylabel
        elif default_ylabel:
            plot_kwargs["ylabel"] = default_ylabel
        if grid is not None:
            plot_kwargs["grid"] = grid
        if grid_alpha is not None:
            plot_kwargs["grid_alpha"] = grid_alpha
        if figsize is not None:
            plot_kwargs["figsize"] = figsize
        if dpi is not None:
            plot_kwargs["dpi"] = dpi
        
        # Merge with additional kwargs
        plot_kwargs.update(kwargs)

        # Use plotter with all settings
        return self.quiver_plotter.plot(X, Y, U, V, ax=ax, **plot_kwargs)
    
    def plot_2d_streamlines(
        self,
        # Standard plane parameters
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        n_points: int = 20,
        plane: str = "xy",
        fixed_coord: float = 0.0,
        # Arbitrary plane parameters
        p1: Optional[NDArray[np.float32]] = None,
        p2: Optional[NDArray[np.float32]] = None,
        p3: Optional[NDArray[np.float32]] = None,
        limits_u: Optional[Tuple[float, float]] = None,
        limits_v: Optional[Tuple[float, float]] = None,
        n_points_u: Optional[int] = None,
        n_points_v: Optional[int] = None,
        # Common parameters
        normalize: bool = True,
        ax: Optional[Axes] = None,
        # Streamline properties
        density: Optional[float] = None,
        linewidth: Optional[float] = None,
        arrowsize: Optional[float] = None,
        arrowstyle: Optional[str] = None,
        start_points: Optional[NDArray[np.float32]] = None,
        maxlength: Optional[float] = None,
        integration_direction: Optional[Literal["forward", "backward", "both"]] = None,
        # Color properties
        color: Optional[Union[str, Tuple[float, float, float]]] = None,
        alpha: Optional[float] = None,
        # Circle properties
        circle_radius: Optional[float] = None,
        circle_color: Optional[Union[str, Tuple[float, float, float]]] = None,
        circle_alpha: Optional[float] = None,
        circle_fill: Optional[bool] = None,
        # Layout properties
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: Optional[bool] = None,
        grid_alpha: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        **kwargs,
    ) -> PlotResult:
        """
        Plot 2D streamlines with support for both standard and arbitrary planes.

        Parameters
        ----------
        Standard Plane Parameters:
        -------------------------
        x_range, y_range : Tuple[float, float]
            Range for coordinates (standard planes only)
        n_points : int
            Number of grid points per dimension (standard planes only)
        plane : str
            Plot plane ('xy', 'yz', 'xz') for standard planes
        fixed_coord : float
            Fixed coordinate for the third dimension (standard planes only)

        Arbitrary Plane Parameters:
        ---------------------------
        p1, p2, p3 : NDArray[np.float32], optional
            Three 3D points defining arbitrary plane
        limits_u, limits_v : Tuple[float, float], optional
            Range for plane coordinates (arbitrary planes only)
        n_points_u, n_points_v : int, optional
            Number of grid points for each dimension (arbitrary planes only)

        Common Parameters:
        ------------------
        normalize : bool
            Whether to normalize field vectors
        ax : Axes, optional
            Existing axes to plot on

        Streamline Properties:
        ---------------------
        density : float, optional
            Density of streamlines. Higher values = more streamlines.
            Range: 0.1 to 10.0, typical: 0.5-3.0
        linewidth : float, optional
            Width of streamline lines. Range: 0.1 to 5.0
        arrowsize : float, optional
            Size of arrows along streamlines. Range: 0.5 to 3.0
        arrowstyle : str, optional
            Style of arrows. Options: ``->``, ``-|>``, ``->>``, ``->|``.
        start_points : NDArray[np.float32], optional
            Custom starting points for streamlines (Nx2 array)
        maxlength : float, optional
            Maximum length of streamlines. Range: 1.0 to 20.0
        integration_direction : {'forward', 'backward', 'both'}, optional
            Direction of streamline integration

        Color Properties:
        -----------------
        color : str or tuple, optional
            Streamline color. Can be named colors, hex codes, RGB tuples, or single letters
        alpha : float, optional
            Streamline transparency. Range: 0.0 (transparent) to 1.0 (opaque)

        Circle Properties:
        ------------------
        circle_radius : float, optional
            Radius of reference circle. Set to 0 to disable. Typical: 0.5-3.0
        circle_color : str or tuple, optional
            Circle color. Same options as arrow colors
        circle_alpha : float, optional
            Circle transparency. Range: 0.0 to 1.0
        circle_fill : bool, optional
            Whether to fill circle (True) or just outline (False)

        Layout Properties:
        ------------------
        title : str, optional
            Plot title text
        xlabel : str, optional
            X-axis label text
        ylabel : str, optional
            Y-axis label text
        grid : bool, optional
            Whether to show background grid
        grid_alpha : float, optional
            Grid transparency. Range: 0.0 to 1.0
        figsize : tuple, optional
            Figure size as (width, height) in inches
        dpi : int, optional
            Figure resolution in dots per inch. Higher = sharper images

        **kwargs
            Additional arguments passed to plotter

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects

        Examples
        --------
        # Standard plane
        >>> from psr_magpy.viz import MultipoleFieldVisualizer
        >>> visualizer = MultipoleFieldVisualizer(field)
        >>> fig, ax = visualizer.plot_2d_streamlines(plane="xy")

        # Arbitrary plane
        >>> p1 = np.array([0, 0, 0], dtype=np.float32)
        >>> p2 = np.array([1, 1, 0], dtype=np.float32)
        >>> p3 = np.array([0, 0, 1], dtype=np.float32)
        >>> fig, ax = visualizer.plot_2d_streamlines(
        ...     p1=p1, p2=p2, p3=p3, limits_u=(-2, 2), limits_v=(-2, 2),
        ...     n_points_u=20, n_points_v=20
        ... )
        """
        # Determine plane type and calculate grid
        if p1 is not None and p2 is not None and p3 is not None:
            # Arbitrary plane
            if limits_u is None or limits_v is None:
                raise PSRMagPyFieldError(
                    "limits_u and limits_v must be provided for arbitrary planes"
                )
            if n_points_u is None or n_points_v is None:
                raise PSRMagPyFieldError(
                    "n_points_u and n_points_v must be provided for arbitrary planes"
                )

            X, Y, U, V, default_xlabel, default_ylabel = self._calculate_grid_field(
                plane_type="arbitrary",
                p1=p1,
                p2=p2,
                p3=p3,
                limits_u=limits_u,
                limits_v=limits_v,
                n_points_u=n_points_u,
                n_points_v=n_points_v,
                normalize=normalize,
            )
        else:
            # Standard plane
            X, Y, U, V, default_xlabel, default_ylabel = self._calculate_grid_field(
                plane_type="standard",
                x_range=x_range,
                y_range=y_range,
                n_points=n_points,
                plane=plane,
                fixed_coord=fixed_coord,
                normalize=normalize,
            )

        # Build plot kwargs from provided parameters
        plot_kwargs: Dict[str, Any] = {}
        
        # Streamline properties
        if density is not None:
            plot_kwargs["density"] = density
        if linewidth is not None:
            plot_kwargs["linewidth"] = linewidth
        if arrowsize is not None:
            plot_kwargs["arrowsize"] = arrowsize
        if arrowstyle is not None:
            plot_kwargs["arrowstyle"] = arrowstyle
        if start_points is not None:
            plot_kwargs["start_points"] = start_points
        if maxlength is not None:
            plot_kwargs["maxlength"] = maxlength
        if integration_direction is not None:
            plot_kwargs["integration_direction"] = integration_direction
        
        # Color properties
        if color is not None:
            plot_kwargs["color"] = color
        if alpha is not None:
            plot_kwargs["alpha"] = alpha
        
        # Circle properties
        if circle_radius is not None:
            plot_kwargs["circle_radius"] = circle_radius
        if circle_color is not None:
            plot_kwargs["circle_color"] = circle_color
        if circle_alpha is not None:
            plot_kwargs["circle_alpha"] = circle_alpha
        if circle_fill is not None:
            plot_kwargs["circle_fill"] = circle_fill
        
        # Layout properties
        if title is not None:
            plot_kwargs["title"] = title
        if xlabel is not None:
            plot_kwargs["xlabel"] = xlabel
        elif default_xlabel:
            plot_kwargs["xlabel"] = default_xlabel
        if ylabel is not None:
            plot_kwargs["ylabel"] = ylabel
        elif default_ylabel:
            plot_kwargs["ylabel"] = default_ylabel
        if grid is not None:
            plot_kwargs["grid"] = grid
        if grid_alpha is not None:
            plot_kwargs["grid_alpha"] = grid_alpha
        if figsize is not None:
            plot_kwargs["figsize"] = figsize
        if dpi is not None:
            plot_kwargs["dpi"] = dpi
        
        # Merge with additional kwargs
        plot_kwargs.update(kwargs)

        # Use plotter with all settings
        return self.streamline_plotter.plot(X, Y, U, V, ax=ax, **plot_kwargs)
