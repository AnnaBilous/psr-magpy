import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, TypeVar, Type, Union, cast, List, Dict
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as rot

from astropy import units as u
from astropy.coordinates import Angle, PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from astropy.io import fits as pf
from astropy.nddata import CCDData

from psr_magpy.utils.logging_config import get_logger
from pydantic import BaseModel, Field, model_validator

logger = get_logger(__name__, console_format_brief=True)

T = TypeVar("T", bound="FootpointGrid")


class GridType(Enum):
    """Grid type enumeration."""

    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"
    CUSTOM = "custom"


class FootpointGrid(ABC):
    """Abstract base class for footpoint grids.

    Provides common functionality for all grid types including
    coordinate conversion, subsetting, and I/O operations.
    """

    def __init__(self) -> None:
        """Initialize grid attributes."""
        self.partial: bool = False
        self.footpoint_coordinates: Optional[CartesianRepresentation] = None
        self.nline_ori: Optional[int] = None
        self.indices: Optional[np.ndarray] = None
        self.nline: Optional[int] = None
        self.grid_type: Optional[GridType] = None

    @classmethod
    def load_from_fits(cls: Type[T], fitsname: str) -> T:
        """Load grid from FITS file."""
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.data_io import GridIO

        # Ensure registration
        if cls not in get_io_factory()._data_registry:
            get_io_factory().register_data_io(cls, GridIO)

        return get_io_factory().load_data(cls, fitsname)

    @classmethod
    def _create_grid_from_header(
        cls: Type[T], header: pf.Header, grid_type: GridType
    ) -> T:
        """Create grid instance from header based on type."""
        grid_classes = {
            GridType.SPHERICAL: SphericalGrid,
            GridType.HEALPIX: HealpixGrid,
            GridType.RANDOM: RandomGrid,
            GridType.CUSTOM: CustomGrid,
        }
        grid_class = grid_classes.get(grid_type)
        if grid_class is None:
            error_msg = f"Unknown grid type: {grid_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return cast(T, grid_class.from_header(header))

    @property
    def colatitude(self) -> np.ndarray:
        """Get colatitude values in degrees."""
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.theta.to(u.deg).value

    @property
    def longitude(self) -> np.ndarray:
        """Get longitude values in degrees."""
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.phi.to(u.deg).value

    @abstractmethod
    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Create footpoint coordinates."""
        pass

    def _set_up_grid(self) -> None:
        """Set up grid coordinates and indices."""
        try:
            logger.debug(
                f"Preparing footpoint coordinates for {self.grid_type.value} grid"
            )
            start_time = time.time()

            self.footpoint_coordinates = self._prepare_footpoints()
            self.nline_ori = len(self.footpoint_coordinates.x)
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori

            duration = time.time() - start_time
            logger.info(f"Grid setup: {self.nline} points in {duration:.3f}s")
            logger.debug(f"Grid type: {self.grid_type.value}, partial: {self.partial}")

        except Exception as e:
            error_msg = f"Failed to set up grid: {str(e)}"
            logger.error(error_msg)
            raise

    def subset_grid(self, grid_indices: np.ndarray) -> "FootpointGrid":
        """Create subset of grid points."""
        logger.debug(f"Creating subset: {len(grid_indices)}/{len(self.indices)} points")

        # Check for empty indices
        if len(grid_indices) == 0:
            error_msg = "Cannot create subset with empty indices"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate indices
        if not all(0 <= i < len(self.indices) for i in grid_indices):
            error_msg = f"Invalid indices. Must be between 0 and {len(self.indices)-1}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create new grid using copy
        new_grid = self.copy()

        # Set subset properties
        new_grid.partial = True
        new_grid.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
        new_grid.indices = grid_indices
        new_grid.nline = len(grid_indices)

        logger.info(f"Subset created: {new_grid.nline} points")
        return new_grid

    def save_to_fits(self, filepath: Union[str, Path]) -> None:
        """Save grid to FITS file."""
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.data_io import GridIO

        # Ensure registration
        if type(self) not in get_io_factory()._data_registry:
            get_io_factory().register_data_io(type(self), GridIO)

        get_io_factory().save_data(self, filepath)

    def plot_footpoint_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = ".",
        color: str = "r",
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot footpoint coordinates."""
        try:
            if ax is None:
                _, ax = plt.subplots()

            ax.plot(self.longitude, self.colatitude, marker, color=color, **kwargs)

            ax.set_xlim(0, 360)
            ax.set_xlabel("Longitude, deg")
            ax.set_ylim(180, 0)
            ax.set_ylabel("Latitude, deg")

            return ax
        except Exception as e:
            error_msg = f"Failed to plot footpoint coordinates: {e}"
            logger.error(error_msg)
            raise

    @abstractmethod
    def copy(self) -> "FootpointGrid":
        """Create a copy of the grid."""
        pass

    def _copy_attributes(self, copied: "FootpointGrid") -> None:
        """Copy common attributes to the copied grid."""
        copied.footpoint_coordinates = self.footpoint_coordinates
        copied.indices = self.indices.copy() if self.indices is not None else None
        copied.partial = self.partial
        copied.nline = self.nline

    @classmethod
    def _extract_params(
        cls, header: pf.Header, required_params: List[str]
    ) -> Dict[str, Any]:
        """Extract parameters from header with consistent error handling."""
        logger.debug(f"Extracting parameters: {required_params}")
        params = {}
        for param in required_params:
            try:
                params[param] = header[param]
                logger.debug(f"Found parameter {param}: {header[param]}")
            except KeyError:
                error_msg = f"Missing {param} parameter in header"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return params

    def _get_header_params(self) -> Dict[str, Any]:
        """Get parameters to add to header. Override in subclasses."""
        return {}

    def _get_header_comments(self) -> Dict[str, str]:
        """Get comments for header parameters. Override in subclasses."""
        return {}

    def _add_params_to_header(self, header: pf.Header) -> None:
        """Add grid parameters to FITS header."""
        params = self._get_header_params()
        comments = self._get_header_comments()

        for key, value in params.items():
            comment = comments.get(key, "")
            header.set(key, value, comment)


class SphericalGridParams(BaseModel):
    """Parameters for spherical grid."""

    nlon: int = Field(gt=0, description="Number of longitude divisions")
    nlat: int = Field(gt=0, description="Number of latitude divisions")
    pole_lat: float = Field(default=0.0, description="Latitude of grid pole")


class SphericalGrid(FootpointGrid):
    """Regular spherical grid with uniform spacing in longitude and latitude."""

    def __init__(self, nlon: int, nlat: int, pole_lat: float = 0):
        """Initialize spherical grid.

        Args:
            nlon: Number of longitude divisions
            nlat: Number of latitude divisions
            pole_lat: Latitude of grid pole in degrees (default: 0)
        """
        super().__init__()
        self.grid_type = GridType.SPHERICAL

        # Pydantic validation
        params = SphericalGridParams(nlon=nlon, nlat=nlat, pole_lat=pole_lat)

        self.nlon = params.nlon
        self.nlat = params.nlat
        self.pole_lat = Angle(params.pole_lat * u.deg)

        logger.debug(f"Initializing SphericalGrid: {nlon}x{nlat}, pole_lat={pole_lat}°")
        self._set_up_grid()

    def __repr__(self) -> str:
        return (
            f"SphericalGrid(nlon={self.nlon}, nlat={self.nlat}, "
            f"pole_lat={self.pole_lat.deg:.1f} deg)"
        )

    @classmethod
    def from_header(cls, header: pf.Header) -> "SphericalGrid":
        """Create SphericalGrid from FITS header."""
        params = cls._extract_params(header, ["NLON", "NLAT"])
        # Map header names to constructor names
        return cls(nlon=params["NLON"], nlat=params["NLAT"])

    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Create spherical grid coordinates."""
        # Create grid
        phi = np.linspace(0, 2 * np.pi, self.nlon, endpoint=False)
        theta = np.linspace(0, np.pi, self.nlat)

        # Create mesh and remove duplicate poles
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        phi_flat, theta_flat = phi_mesh.ravel(), theta_mesh.ravel()

        # Remove duplicate pole points
        pole_mask = (phi_flat > 0) & ((theta_flat == 0) | (theta_flat == np.pi))
        phi_clean = phi_flat[~pole_mask]
        theta_clean = theta_flat[~pole_mask]

        # Convert to Cartesian
        coords = PhysicsSphericalRepresentation(
            phi=Angle(phi_clean * u.rad), theta=Angle(theta_clean * u.rad), r=1
        )
        cart_coords = coords.represent_as(CartesianRepresentation)

        # Apply rotation if needed
        if self.pole_lat != 0:
            rotation = rot.from_euler("yzx", [self.pole_lat.rad, 0, 0])
            rotated_xyz = rotation.apply(cart_coords.xyz.value.T).T
            cart_coords = CartesianRepresentation(
                rotated_xyz, unit=cart_coords.xyz.unit
            )

        return cart_coords

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NLON", self.nlon, "Number of longitude divisions")
        header.set("NLAT", self.nlat, "Number of latitude divisions")
        header.set("POLELAT", self.pole_lat.deg, "Latitude of grid pole, degree")

    def copy(self) -> "SphericalGrid":
        """Create a copy of the grid."""
        copied = SphericalGrid(self.nlon, self.nlat, self.pole_lat.deg)
        self._copy_attributes(copied)
        return copied


class HealpixGridParams(BaseModel):
    """Parameters for HEALPix grid."""

    nside: int = Field(gt=0, description="HEALPix resolution parameter")

    @model_validator(mode="after")
    def validate_power_of_two(self):
        if not (self.nside & (self.nside - 1) == 0):
            error_msg = f"nside must be a power of 2, got {self.nside}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self


class HealpixGrid(FootpointGrid):
    """HEALPix grid for uniform spherical sampling."""

    def __init__(self, nside: int):
        """Initialize HEALPix grid.

        Args:
            nside: HEALPix resolution parameter (must be power of 2)
        """
        super().__init__()
        self.grid_type = GridType.HEALPIX

        # Pydantic validation
        params = HealpixGridParams(nside=nside)
        self.nside = params.nside
        self._set_up_grid()

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self.nside})"

    def _get_header_params(self) -> Dict[str, Any]:
        return {"NSIDE": self.nside}

    def _get_header_comments(self) -> Dict[str, str]:
        return {"NSIDE": "Healpix parameter nside"}

    def _add_params_to_header(self, header: pf.Header) -> None:
        """Add grid parameters to FITS header."""
        params = self._get_header_params()
        comments = self._get_header_comments()

        for key, value in params.items():
            comment = comments.get(key, "")
            header.set(key, value, comment)

    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Create HEALPix grid coordinates."""
        npix = hp.nside2npix(self.nside)
        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(npix))

        coords = PhysicsSphericalRepresentation(
            phi=Angle(phi_val * u.rad), theta=Angle(theta_val * u.rad), r=1
        )
        return coords.represent_as(CartesianRepresentation)

    @classmethod
    def from_header(cls, header: pf.Header) -> "HealpixGrid":
        """Create HealpixGrid from FITS header."""
        params = cls._extract_params(header, ["NSIDE"])
        return cls(nside=params["NSIDE"])

    def copy(self) -> "HealpixGrid":
        """Create a copy of the grid."""
        copied = HealpixGrid(self.nside)
        self._copy_attributes(copied)
        return copied


class RandomGridParams(BaseModel):
    """Parameters for random grid."""

    nline_ori: int = Field(gt=0, description="Number of random points")


class RandomGrid(FootpointGrid):
    """Random uniform distribution of points on a sphere."""

    def __init__(self, nline_ori: int):
        """Initialize random grid.

        Args:
            nline_ori: Number of random points to generate
        """
        super().__init__()
        self.grid_type = GridType.RANDOM

        # Pydantic validation
        params = RandomGridParams(nline_ori=nline_ori)
        self.nline_ori = params.nline_ori

        # Issue warning about saving to file
        logger.warning(
            "RandomGrid created: This grid must be saved to a file and referenced by path in experiment configs. "
            "Use grid.save_to_fits('path/to/grid.fits') to save the grid."
        )

        self._set_up_grid()

    def __repr__(self) -> str:
        return f"RandomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for random grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        # Muller method for random points on a sphere
        u: np.ndarray = np.random.normal(loc=0, scale=1, size=(3, self.nline_ori))
        norm: np.ndarray = (np.sum(u**2, axis=0)) ** 0.5
        x, y, z = u / norm
        return CartesianRepresentation(x, y, z)

    @classmethod
    def from_header(cls, header: pf.Header) -> "RandomGrid":
        """Create RandomGrid from FITS header."""
        params = cls._extract_params(header, ["NLINE"])
        return cls(nline_ori=params["NLINE"])

    def copy(self) -> "RandomGrid":
        """Create a copy of the grid."""
        copied = RandomGrid(self.nline_ori)
        self._copy_attributes(copied)
        return copied


class CustomGridParams(BaseModel):
    """Parameters for custom grid."""

    nline_ori: int = Field(gt=0, description="Number of points")
    coordinates: Optional[CartesianRepresentation] = Field(
        default=None, description="Initial coordinates"
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_coordinates_length(self):
        if self.coordinates is not None:
            if len(self.coordinates.x) != self.nline_ori:
                error_msg = f"Coordinates length {len(self.coordinates.x)} != nline_ori {self.nline_ori}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return self


class CustomGrid(FootpointGrid):
    """Custom grid with user-provided coordinates."""

    def __init__(
        self, nline_ori: int, coordinates: Optional[CartesianRepresentation] = None
    ):
        """Initialize custom grid.

        Args:
            nline_ori: Number of points in the grid
            coordinates: Initial coordinates (optional, defaults to NaN)
        """
        super().__init__()
        self.grid_type = GridType.CUSTOM

        # Pydantic validation
        params = CustomGridParams(nline_ori=nline_ori, coordinates=coordinates)

        self.nline_ori = params.nline_ori
        self.set_coordinates(params.coordinates)
        self.indices = np.arange(self.nline_ori)
        self.nline = self.nline_ori

        # Issue warning about saving to file
        logger.warning(
            "CustomGrid created: This grid must be saved to a file and referenced by path in experiment configs. "
            "Use grid.save_to_fits('path/to/grid.fits') to save the grid."
        )

    def __repr__(self) -> str:
        return f"CustomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for custom grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Return current footpoint coordinates."""
        return self.footpoint_coordinates

    def set_coordinates(self, coordinates: Optional[CartesianRepresentation]) -> None:
        """Set new coordinates for grid points.

        Args:
            coordinates: New coordinates for grid points (can be None)

        Raises:
            ValueError: If coordinates length doesn't match nline_ori
        """
        if coordinates is None:
            # Create NaN coordinates
            nan_coords = np.full(self.nline_ori, np.nan)
            self.footpoint_coordinates = CartesianRepresentation(
                nan_coords, nan_coords, nan_coords
            )
        else:
            if len(coordinates.x) != self.nline_ori:
                error_msg = f"Coordinates length {len(coordinates.x)} != nline_ori {self.nline_ori}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.footpoint_coordinates = coordinates

    @classmethod
    def from_header(cls, header: pf.Header) -> "CustomGrid":
        """Create CustomGrid from FITS header."""
        params = cls._extract_params(header, ["NLINE"])
        return cls(nline_ori=params["NLINE"])

    def copy(self) -> "CustomGrid":
        """Create a copy of the grid."""
        copied = CustomGrid(self.nline_ori)
        self._copy_attributes(copied)
        return copied


"""Grid classes for magnetic field line tracing.

Provides different grid types for sampling points on a sphere:
- SphericalGrid: Regular lat/lon grid
- HealpixGrid: Uniform spherical sampling  
- RandomGrid: Random uniform distribution
- CustomGrid: User-defined coordinates
"""
