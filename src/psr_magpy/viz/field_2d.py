"""2D field visualization module."""

from typing import Optional, Tuple, Dict, Any, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


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
            'sphere_radius': 1.0,
            'sphere_alpha': 0.3,
            'sphere_color': 'lightblue',
            'grid_color': 'red',
            'grid_alpha': 0.7,
            'grid_size': 0.1,
            'axis_length': 2.0,
            'axis_width': 2,
            'title': '3D Grid Visualization',
            'figsize': (10, 8),
            'dpi': 100,
            'elevation': 20,
            'azimuth': 45,
            **defaults
        }
    
    def plot_grid_on_plane(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                          limits_u: Tuple[float, float], limits_v: Tuple[float, float],
                          n_points_u: int, n_points_v: int,
                          ax: Optional[Axes3D] = None, **kwargs) -> Tuple[Figure, Axes3D]:
        """Plot 3D visualization with coordinate system, sphere, and plane grid."""
        plot_config = {**self.config, **kwargs}
        
        if ax is None:
            fig = plt.figure(figsize=plot_config['figsize'], dpi=plot_config['dpi'])
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        assert isinstance(ax, Axes3D)
        assert isinstance(fig, Figure)
        
        # Check if points define a valid plane
        if np.allclose(p1, p2) or np.allclose(p1, p3) or np.allclose(p2, p3):
            error_msg = "At least two points coincide. Cannot define a plane."
            raise ValueError(error_msg)
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        if np.allclose(cross_product, 0):
            error_msg = "Points are collinear. Cannot define a plane."
            raise ValueError(error_msg)
        
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
        axis_length = plot_config['axis_length']
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=plot_config['axis_width'])
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=plot_config['axis_width'])
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=plot_config['axis_width'])
        
        # Add axis labels
        ax.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=12)
        ax.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=12)
        ax.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=12)
        
        # Plot sphere
        sphere_radius = plot_config['sphere_radius']
        u_sphere = np.linspace(0, 2 * np.pi, 20)
        v_sphere = np.linspace(0, np.pi, 20)
        x_sphere = sphere_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y_sphere = sphere_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z_sphere = sphere_radius * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                       alpha=plot_config['sphere_alpha'], 
                       color=plot_config['sphere_color'])
        
        # Plot grid points on plane
        grid_size = plot_config['grid_size']
        for i in range(n_points_v):
            for j in range(n_points_u):
                point = X_3d[i, j]
                ax.scatter(point[0], point[1], point[2], 
                          c=plot_config['grid_color'], 
                          alpha=plot_config['grid_alpha'],
                          s=grid_size*1000)
        
        # Set view and labels
        ax.view_init(elev=plot_config['elevation'], azim=plot_config['azimuth'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(plot_config['title'])
        
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

    Args:
        field: Field object
        plane: Plane to plot in ("xy", "yz", or "xz")
        fixed_coord: Value of fixed coordinate
        x_range: Range for x-axis
        y_range: Range for y-axis
        n_points: Number of points per dimension
        ax: Optional matplotlib axes
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axes
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
        error_msg = f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate field at each point
    field_values = np.zeros((n_points, n_points, 3))
    for i in range(n_points):
        for j in range(n_points):
            field_values[i, j] = field.get_field(
                [coords[0][i, j], coords[1][i, j], coords[2][i, j]]
            )

    # Plot streamlines
    ax.streamplot(X, Y, field_values[:, :, 0], field_values[:, :, 1], **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    return ax
