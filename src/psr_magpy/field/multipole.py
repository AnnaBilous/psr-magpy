import logging
from typing import List, Optional, Tuple, Union, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator
import matplotlib.pyplot as plt

from psr_magpy.field.components import FieldComponent
from psr_magpy.viz import Quiver2DFieldPlotter, Streamline2DFieldPlotter
from psr_magpy.utils.logging_config import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = get_logger(__name__, console_format_brief=True)

# Type aliases for better IDE support
PlotResult = Tuple["Figure", "Axes"]


class MultipoleFieldParams(BaseModel):
    """Parameters for multipole field configuration."""

    field_components: List[FieldComponent] = Field(
        description="List of field components that make up this multipole field"
    )

    @field_validator("field_components")
    @classmethod
    def validate_field_components(cls, v):
        """Validate that all components are FieldComponent instances and list is not empty."""
        if not v:
            raise ValueError("Field components list cannot be empty")

        for i, component in enumerate(v):
            if not isinstance(component, FieldComponent):
                raise ValueError(
                    f"Component {i+1} is not a FieldComponent instance, got {type(component).__name__}"
                )

        return v

    model_config = {"arbitrary_types_allowed": True}


class MultipoleField:
    """
    Magnetic field composed of multiple field components.

    This class manages a collection of field components (dipole, quadrupole, etc.) and
    provides methods to calculate their combined field effect.

    Parameters
    ----------
    field_components : List[FieldComponent]
        List of field components that make up this multipole field.
        Each component must be a FieldComponent instance.

    Raises
    ------
    ValueError
        If field_components list is empty or contains invalid components.

    Examples
    --------
    >>> from psr_magpy.field.components import DipoleFieldComponent
    >>> dipole = DipoleFieldComponent(B_magnitude=1.0)
    >>> field = MultipoleField([dipole])
    >>> field.get_total_field(np.array([1.0, 0.0, 0.0]))
    array([...])
    """

    def __init__(self, field_components: List[FieldComponent]):
        """Initialize MultipoleField with validated field components."""
        logger.info(
            f"Initializing MultipoleField with {len(field_components)} components"
        )

        # Validate components using Pydantic
        try:
            self.params = MultipoleFieldParams(field_components=field_components)
            self.field_components = self.params.field_components
        except Exception as e:
            error_msg = f"Failed to initialize MultipoleField parameters: {e}"
            logger.error(error_msg)
            raise

        # Initialize plotters
        self._quiver_plotter = None
        self._streamline_plotter = None
        self._streamline_3d_plotter = None

        # Log component details at debug level
        if logger.isEnabledFor(logging.DEBUG):
            for i, comp in enumerate(self.field_components):
                logger.debug(f"Component {i+1}: {comp}")

        logger.debug("MultipoleField initialized successfully")

    @property
    def n_components(self) -> int:
        """Get number of field components."""
        return len(self.field_components)

    def get_numba_config(self) -> List[dict]:
        """Get Numba-compatible configuration for all field components.
        
        Collects Numba-compatible parameters from all components and validates
        that all components support Numba compilation.
        
        Returns
        -------
        List[dict]
            List of parameter dictionaries, one per component. Each dict contains:
            - 'B_magnitude': float
            - 'rotation_matrix': NDArray[float64] (3x3, magn to spin frame)
            - 'rotation_matrix_inv': NDArray[float64] (3x3, spin to magn frame)
            - 'offset_vector': NDArray[float64] (3-element)
            - 'moment_type': str ("dipole" or "quadrupole")
        
        Raises
        ------
        ValueError
            If any component does not support Numba compilation (missing get_numba_params method).
        """
        config = []
        for i, component in enumerate(self.field_components):
            if not hasattr(component, "get_numba_params"):
                error_msg = (
                    f"Component {i+1} ({type(component).__name__}) does not support "
                    "Numba compilation. Missing get_numba_params() method."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                params = component.get_numba_params()
                config.append(params)
            except Exception as e:
                error_msg = (
                    f"Failed to get Numba parameters from component {i+1} "
                    f"({type(component).__name__}): {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        logger.debug(f"Generated Numba config for {len(config)} components")
        return config

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

    def get_total_field(
        self, r: NDArray[np.float32], K: int = 1, normalize: bool = True
    ) -> NDArray[np.float32]:
        """
        Fast calculation of total magnetic field at position r.
        Assumes r is a valid 3-element float32 array. No validation or error handling.

        Parameters
        ----------
        r : NDArray[np.float32]
            Position vector (3-element array)
        K : int, optional
            Field direction multiplier (1 or -1). Default is 1.
        normalize : bool, optional
            Whether to normalize the field vector. Default is True.

        Returns
        -------
        NDArray[np.float32]
            Total magnetic field vector
        """
        F = np.zeros(3, dtype=np.float32)
        for component in self.field_components:
            F += component.get_field(r)
        F *= K
        if normalize:
            mag = np.linalg.norm(F)
            if mag > 0:
                F /= mag

        return F

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
                error_msg = f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Calculate field at each grid point
            U = np.zeros_like(X)
            V = np.zeros_like(Y)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pos = np.array(
                        [coords[0][i, j], coords[1][i, j], coords[2][i, j]],
                        dtype=np.float32,
                    )
                    field_vec = self.get_total_field(pos, normalize=normalize)

                    if plane == "xy":
                        U[i, j], V[i, j] = field_vec[0], field_vec[1]
                    elif plane == "yz":
                        U[i, j], V[i, j] = field_vec[1], field_vec[2]
                    else:  # xz
                        U[i, j], V[i, j] = field_vec[0], field_vec[2]

            return X, Y, U, V, xlabel, ylabel

        elif plane_type == "arbitrary":
            if p1 is None or p2 is None or p3 is None:
                error_msg = "p1, p2, p3 must be provided for arbitrary planes"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if limits_u is None or limits_v is None:
                error_msg = "limits_u, limits_v must be provided for arbitrary planes"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if n_points_u is None or n_points_v is None:
                error_msg = (
                    "n_points_u, n_points_v must be provided for arbitrary planes"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check if points coincide
            if np.allclose(p1, p2) or np.allclose(p1, p3) or np.allclose(p2, p3):
                error_msg = "At least two points coincide. Cannot define a plane."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Calculate plane basis vectors
            v1 = p2 - p1  # First direction vector
            v2 = p3 - p1  # Second direction vector

            # Check if points are collinear (cross product should be non-zero)
            cross_product = np.cross(v1, v2)
            if np.allclose(cross_product, 0):
                error_msg = "Points are collinear. Cannot define a plane."
                logger.error(error_msg)
                raise ValueError(error_msg)

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
                    field_vectors[i, j] = self.get_total_field(pos, normalize=normalize)

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
            error_msg = (
                f"Invalid plane_type: {plane_type}. Must be 'standard' or 'arbitrary'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        ax: Optional["Axes"] = None,
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
        fig, ax = field.plot_2d_quiver(plane="xy")

        # Arbitrary plane
        p1 = np.array([0, 0, 0], dtype=np.float32)
        p2 = np.array([1, 1, 0], dtype=np.float32)
        p3 = np.array([0, 0, 1], dtype=np.float32)
        fig, ax = field.plot_2d_quiver(
            p1=p1, p2=p2, p3=p3, limits_u=(-2, 2), limits_v=(-2, 2),
            n_points_u=20, n_points_v=20
        )

        # With customization
        fig, ax = field.plot_2d_quiver(
            scale=30, color='red', circle_radius=1.5,
            title='Magnetic Field'
        )
        """
        # Determine plane type and calculate grid
        if p1 is not None and p2 is not None and p3 is not None:
            # Arbitrary plane
            if limits_u is None or limits_v is None:
                error_msg = (
                    "limits_u and limits_v must be provided for arbitrary planes"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            if n_points_u is None or n_points_v is None:
                error_msg = (
                    "n_points_u and n_points_v must be provided for arbitrary planes"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

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

        # Configure plotter with all options
        plotter = self.quiver_plotter

        # Set arrow properties if provided
        if any(
            v is not None
            for v in [scale, width, headwidth, headlength, minshaft, minlength, pivot]
        ):
            plotter.set_arrows(
                scale=scale,
                width=width,
                headwidth=headwidth,
                headlength=headlength,
                minshaft=minshaft,
                minlength=minlength,
                pivot=pivot,
            )

        # Set color properties if provided
        if color is not None or alpha is not None:
            plotter.set_colors(color=color, alpha=alpha)

        # Set circle properties if provided
        if any(
            v is not None
            for v in [circle_radius, circle_color, circle_alpha, circle_fill]
        ):
            plotter.set_circle(
                circle_radius=circle_radius,
                circle_color=circle_color,
                circle_alpha=circle_alpha,
                circle_fill=circle_fill,
            )

        # Set layout properties if provided
        if any(
            v is not None
            for v in [title, xlabel, ylabel, grid, grid_alpha, figsize, dpi]
        ):
            plotter.set_layout(
                title=title,
                xlabel=xlabel or default_xlabel,
                ylabel=ylabel or default_ylabel,
                grid=grid,
                grid_alpha=grid_alpha,
                figsize=figsize,
                dpi=dpi,
            )

        # Use plotter with all settings
        return plotter.plot(X, Y, U, V, ax=ax, **kwargs)

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
        ax: Optional["Axes"] = None,
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
            Style of arrows. Options: '->', '-|>', '->>', '->|', etc.
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
        fig, ax = field.plot_2d_streamlines(plane="xy")

        # Arbitrary plane
        p1 = np.array([0, 0, 0], dtype=np.float32)
        p2 = np.array([1, 1, 0], dtype=np.float32)
        p3 = np.array([0, 0, 1], dtype=np.float32)
        fig, ax = field.plot_2d_streamlines(
            p1=p1, p2=p2, p3=p3, limits_u=(-2, 2), limits_v=(-2, 2),
            n_points_u=20, n_points_v=20
        )

        # With customization
        fig, ax = field.plot_2d_streamlines(
            density=2.0, color='blue', linewidth=2.0,
            title='Flow Streamlines'
        )
        """
        # Determine plane type and calculate grid
        if p1 is not None and p2 is not None and p3 is not None:
            # Arbitrary plane
            if limits_u is None or limits_v is None:
                error_msg = (
                    "limits_u and limits_v must be provided for arbitrary planes"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            if n_points_u is None or n_points_v is None:
                error_msg = (
                    "n_points_u and n_points_v must be provided for arbitrary planes"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

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

        # Configure plotter with all options
        plotter = self.streamline_plotter

        # Set streamline properties if provided
        if any(
            v is not None
            for v in [
                density,
                linewidth,
                arrowsize,
                arrowstyle,
                start_points,
                maxlength,
                integration_direction,
            ]
        ):
            plotter.set_streamlines(
                density=density,
                linewidth=linewidth,
                arrowsize=arrowsize,
                arrowstyle=arrowstyle,
                start_points=start_points,
                maxlength=maxlength,
                integration_direction=integration_direction,
            )

        # Set color properties if provided
        if color is not None or alpha is not None:
            plotter.set_colors(color=color, alpha=alpha)

        # Set circle properties if provided
        if any(
            v is not None
            for v in [circle_radius, circle_color, circle_alpha, circle_fill]
        ):
            plotter.set_circle(
                circle_radius=circle_radius,
                circle_color=circle_color,
                circle_alpha=circle_alpha,
                circle_fill=circle_fill,
            )

        # Set layout properties if provided
        if any(
            v is not None
            for v in [title, xlabel, ylabel, grid, grid_alpha, figsize, dpi]
        ):
            plotter.set_layout(
                title=title,
                xlabel=xlabel or default_xlabel,
                ylabel=ylabel or default_ylabel,
                grid=grid,
                grid_alpha=grid_alpha,
                figsize=figsize,
                dpi=dpi,
            )

        # Use plotter with all settings
        return plotter.plot(X, Y, U, V, ax=ax, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the multipole field."""
        component_types = [type(c).__name__ for c in self.field_components]
        return f"MultipoleField(components=[{', '.join(component_types)}], n_components={self.n_components})"
