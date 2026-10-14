import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, TypeVar, Type, Union, cast, List, Dict, Tuple, TYPE_CHECKING
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
from psr_magpy.utils.exceptions import PSRMagPyGridError, PSRMagPyValidationError, PSRMagPyIOError, PSRMagPyExperimentError
from pydantic import BaseModel, Field, model_validator, field_validator

if TYPE_CHECKING:
    from psr_magpy.experiment.grid_config import GridConfig
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.line.field_line import FieldLine

logger = get_logger(__name__, console_format_brief=True)

T = TypeVar("T", bound="FootpointGrid")


class GridType(Enum):
    """Grid type enumeration."""

    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"
    CUSTOM = "custom"


class SubsetGridParams(BaseModel):
    """Parameters for creating a grid subset with validation."""

    grid_indices: Union[np.ndarray, List[int]]
    max_index: int

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("grid_indices", mode="before")
    @classmethod
    def convert_to_array(cls, v):
        """Convert list to numpy array if needed."""
        if isinstance(v, list):
            return np.array(v, dtype=np.int64)
        return v

    @model_validator(mode="after")
    def validate_indices(self):
        """Validate that indices are non-empty and within bounds."""
        if len(self.grid_indices) == 0:
            raise ValueError("Cannot create subset with empty indices")

        if not all(0 <= i < self.max_index for i in self.grid_indices):
            raise ValueError(
                f"Invalid indices. Must be between 0 and {self.max_index - 1}"
            )

        return self


class FootpointGrid(ABC):
    """Abstract base class for footpoint grids.

    Provides common functionality for all grid types including
    coordinate conversion, subsetting, and I/O operations.
    """

    def __init__(self) -> None:
        """Initialize grid attributes.
        
        Note: Grid coordinates are not computed until setup() is called
        or footpoint_coordinates is accessed (lazy initialization).
        """
        self.partial: bool = False
        self._footpoint_coordinates: Optional[CartesianRepresentation] = None
        self._initialized: bool = False
        self.nline_ori: Optional[int] = None
        self.indices: Optional[np.ndarray] = None
        self.nline: Optional[int] = None
        self.grid_type: Optional[GridType] = None

    @classmethod
    def load_from_fits(cls: Type[T], fitsname: str) -> T:
        """Load grid from FITS file."""
        from psr_magpy.io import load_data

        result = load_data(cls, fitsname)
        return cast(T, result)

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
            raise PSRMagPyGridError(f"Unknown grid type: {grid_type}")

        # Type ignore: from_header is a class method on grid classes, not visible to mypy on ABCMeta
        return cast(T, grid_class.from_header(header))  # type: ignore[attr-defined]

    @property
    def footpoint_coordinates(self) -> CartesianRepresentation:
        """Get footpoint coordinates (lazy initialization).
        
        Coordinates are computed on first access if grid is not yet initialized.
        
        Returns
        -------
        CartesianRepresentation
            Footpoint coordinates
            
        Raises
        ------
        PSRMagPyGridError
            If grid setup fails
        """
        if not self._initialized:
            self.setup()
        if self._footpoint_coordinates is None:
            raise PSRMagPyGridError("Grid coordinates are None after setup")
        return self._footpoint_coordinates
    
    @footpoint_coordinates.setter
    def footpoint_coordinates(self, value: CartesianRepresentation) -> None:
        """Set footpoint coordinates directly."""
        self._footpoint_coordinates = value
        if value is not None:
            self._initialized = True

    @property
    def colatitude(self) -> np.ndarray:
        """Get colatitude values in degrees."""
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return np.asarray(spherical.theta.to(u.deg).value)

    @property
    def longitude(self) -> np.ndarray:
        """Get longitude values in degrees."""
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return np.asarray(spherical.phi.to(u.deg).value)

    @abstractmethod
    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Create footpoint coordinates."""
        pass

    def setup(self) -> None:
        """Explicitly initialize grid coordinates and indices.
        
        This method computes footpoint coordinates by calling the abstract
        _prepare_footpoints() method. It can be called explicitly or will
        be called automatically when footpoint_coordinates is first accessed.
        
        Raises
        ------
        PSRMagPyGridError
            If grid setup fails
        """
        if self._initialized:
            logger.debug("Grid already initialized, skipping setup")
            return
            
        try:
            grid_type_str = self.grid_type.value if self.grid_type is not None else "unknown"
            logger.debug(
                f"Preparing footpoint coordinates for {grid_type_str} grid"
            )
            start_time = time.time()

            self._footpoint_coordinates = self._prepare_footpoints()
            self.nline_ori = len(self._footpoint_coordinates.x)
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori
            self._initialized = True

            duration = time.time() - start_time
            logger.info(f"Grid setup: {self.nline} points in {duration:.3f}s")
            grid_type_str = self.grid_type.value if self.grid_type is not None else "unknown"
            logger.debug(f"Grid type: {grid_type_str}, partial: {self.partial}")

        except Exception as e:
            raise PSRMagPyGridError(f"Failed to set up grid: {str(e)}") from e
    
    def _set_up_grid(self) -> None:
        """Set up grid (internal method for backward compatibility). Use setup() instead."""
        self.setup()

    def subset_grid(self, grid_indices: np.ndarray) -> "FootpointGrid":
        """Create subset of grid points."""
        if self.indices is None:
            raise PSRMagPyGridError("Cannot create subset: indices is None")
        logger.debug(f"Creating subset: {len(grid_indices)}/{len(self.indices)} points")

        # Validate using Pydantic model
        
        SubsetGridParams(
            grid_indices=grid_indices,
            max_index=len(self.indices)
        )

        # Create new grid using copy
        new_grid = self.copy()

        # Set subset properties
        new_grid.partial = True
        new_grid.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
        # Preserve original footpoint IDs by indexing into self.indices
        new_grid.indices = self.indices[grid_indices]
        new_grid.nline = len(grid_indices)

        logger.info(f"Subset created: {new_grid.nline} points")
        return new_grid

    def save_to_fits(self, filepath: Union[str, Path]) -> None:
        """Save grid to FITS file."""
        from psr_magpy.io import save_data

        save_data(self, filepath)

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
            raise PSRMagPyGridError(f"Failed to plot footpoint coordinates: {e}") from e

    @abstractmethod
    def copy(self) -> "FootpointGrid":
        """Create a copy of the grid."""
        pass

    def _copy_attributes(self, copied: "FootpointGrid") -> None:
        """Copy common attributes to the copied grid."""
        copied._footpoint_coordinates = self._footpoint_coordinates
        copied._initialized = self._initialized
        copied.indices = self.indices.copy() if self.indices is not None else None
        copied.partial = self.partial
        copied.nline = self.nline
        copied.nline_ori = self.nline_ori

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
                raise PSRMagPyIOError(f"Missing {param} parameter in header")
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

    def __init__(self, nlon: int, nlat: int, pole_lat: float = 0, auto_setup: bool = True):
        """Initialize spherical grid.

        Parameters
        ----------
        nlon : int
            Number of longitude divisions.
        nlat : int
            Number of latitude divisions.
        pole_lat : float, optional
            Latitude of grid pole in degrees. Default is 0.
        auto_setup : bool, optional
            If True, automatically call setup(). Default is True for backward compatibility.
        """
        super().__init__()
        self.grid_type = GridType.SPHERICAL

        # Pydantic validation
        params = SphericalGridParams(nlon=nlon, nlat=nlat, pole_lat=pole_lat)

        self.nlon = params.nlon
        self.nlat = params.nlat
        self.pole_lat = Angle(params.pole_lat * u.deg)

        logger.debug(f"Initializing SphericalGrid: {nlon}x{nlat}, pole_lat={pole_lat}°")
        if auto_setup:
            self.setup()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SphericalGrid(nlon={self.nlon}, nlat={self.nlat}, "
            f"pole_lat={self.pole_lat.deg:.1f} deg)"
        )

    @classmethod
    def from_header(cls, header: pf.Header) -> "SphericalGrid":
        """Create SphericalGrid from FITS header."""
        params = cls._extract_params(header, ["NLON", "NLAT"])
        # Map header names to constructor names
        grid = cls(nlon=params["NLON"], nlat=params["NLAT"], auto_setup=False)
        # Coordinates will be loaded from FITS file, so mark as initialized
        grid._initialized = True
        return grid

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
        """Validate that nside is a power of 2.
        
        Raises
        ------
        ValueError
            If nside is not a power of 2.
        """
        if not (self.nside & (self.nside - 1) == 0):
            raise ValueError(f"nside must be a power of 2, got {self.nside}")
        return self


class HealpixGrid(FootpointGrid):
    """HEALPix grid for uniform spherical sampling."""

    def __init__(self, nside: int, auto_setup: bool = True):
        """Initialize HEALPix grid.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter (must be power of 2).
        auto_setup : bool, optional
            If True, automatically call setup(). Default is True for backward compatibility.
        """
        super().__init__()
        self.grid_type = GridType.HEALPIX

        # Pydantic validation
        params = HealpixGridParams(nside=nside)
        self.nside = params.nside
        if auto_setup:
            self.setup()

    def __repr__(self) -> str:
        """Return string representation."""
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
        return cls(nside=params["NSIDE"], auto_setup=False)

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

    def __init__(self, nline_ori: int, auto_setup: bool = True):
        """Initialize random grid.

        Parameters
        ----------
        nline_ori : int
            Number of random points to generate.
        auto_setup : bool, optional
            If True, automatically call setup(). Default is True for backward compatibility.
        """
        super().__init__()
        self.grid_type = GridType.RANDOM

        # Pydantic validation
        params = RandomGridParams(nline_ori=nline_ori)
        self.nline_ori = params.nline_ori

        # Issue warning about saving to file
        if auto_setup:
            logger.warning(
                "RandomGrid created: This grid must be saved to a file and referenced by path in experiment configs. "
                "Use grid.save_to_fits('path/to/grid.fits') to save the grid."
            )
            self.setup()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RandomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for random grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        # Muller method for random points on a sphere
        if self.nline_ori is None:
            raise PSRMagPyGridError("Cannot prepare footpoints: nline_ori is None")
        u: np.ndarray = np.random.normal(loc=0, scale=1, size=(3, self.nline_ori))
        norm: np.ndarray = (np.sum(u**2, axis=0)) ** 0.5
        x, y, z = u / norm
        return CartesianRepresentation(x, y, z)

    @classmethod
    def from_header(cls, header: pf.Header) -> "RandomGrid":
        """Create RandomGrid from FITS header."""
        params = cls._extract_params(header, ["NLINE"])
        nline = params["NLINE"]
        if not isinstance(nline, int):
            raise PSRMagPyGridError(f"Invalid NLINE value: {nline}")
        grid = cls(nline_ori=nline, auto_setup=False)
        # Coordinates will be loaded from FITS file, so mark as initialized
        grid._initialized = True
        return grid

    def copy(self) -> "RandomGrid":
        """Create a copy of the grid."""
        if self.nline_ori is None:
            raise PSRMagPyGridError("Cannot copy grid: nline_ori is None")
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
        """Validate that coordinates length matches nline_ori.
        
        Raises
        ------
        ValueError
            If coordinates length doesn't match nline_ori.
        """
        if self.coordinates is not None:
            if len(self.coordinates.x) != self.nline_ori:
                raise ValueError(
                    f"Coordinates length {len(self.coordinates.x)} != nline_ori {self.nline_ori}"
                )
        return self


class CustomGrid(FootpointGrid):
    """Custom grid with user-provided coordinates."""

    def __init__(
        self, nline_ori: int, coordinates: Optional[CartesianRepresentation] = None, auto_setup: bool = True
    ):
        """Initialize custom grid.

        Parameters
        ----------
        nline_ori : int
            Number of points in the grid.
        coordinates : Optional[CartesianRepresentation], optional
            Initial coordinates. Defaults to NaN if not provided.
        auto_setup : bool, optional
            If True, automatically set up indices. Default is True for backward compatibility.
        """
        super().__init__()
        self.grid_type = GridType.CUSTOM

        # Pydantic validation
        params = CustomGridParams(nline_ori=nline_ori, coordinates=coordinates)

        self.nline_ori = params.nline_ori
        self.set_coordinates(params.coordinates)
        if auto_setup:
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori
            logger.warning(
                "CustomGrid created: This grid must be saved to a file and referenced by path in experiment configs. "
                "Use grid.save_to_fits('path/to/grid.fits') to save the grid."
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CustomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for custom grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Return current footpoint coordinates."""
        return self.footpoint_coordinates

    def set_coordinates(self, coordinates: Optional[CartesianRepresentation]) -> None:
        """Set new coordinates for grid points.

        Parameters
        ----------
        coordinates : Optional[CartesianRepresentation]
            New coordinates for grid points. Can be None.

        Raises
        ------
        ValueError
            If coordinates length doesn't match nline_ori.
        """
        if coordinates is None:
            # Create NaN coordinates
            if self.nline_ori is None:
                raise PSRMagPyGridError("Cannot create NaN coordinates: nline_ori is None")
            nan_coords = np.full(self.nline_ori, np.nan)
            self.footpoint_coordinates = CartesianRepresentation(
                nan_coords, nan_coords, nan_coords
            )
        else:
            if len(coordinates.x) != self.nline_ori:
                raise PSRMagPyGridError(
                    f"Coordinates length {len(coordinates.x)} != nline_ori {self.nline_ori}"
                )
            self.footpoint_coordinates = coordinates

    @classmethod
    def from_header(cls, header: pf.Header) -> "CustomGrid":
        """Create CustomGrid from FITS header."""
        params = cls._extract_params(header, ["NLINE"])
        nline = params["NLINE"]
        if not isinstance(nline, int):
            raise PSRMagPyGridError(f"Invalid NLINE value: {nline}")
        grid = cls(nline_ori=nline, auto_setup=False)
        # Coordinates will be loaded from FITS file, so mark as initialized
        grid._initialized = True
        return grid

    def copy(self) -> "CustomGrid":
        """Create a copy of the grid."""
        if self.nline_ori is None:
            raise PSRMagPyGridError("Cannot copy grid: nline_ori is None")
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


def upgrade_healpix_footpoints(
    coarse_nside: int, fine_nside: int, footpoint_indices: np.ndarray
) -> np.ndarray:
    """Upgrade healpix footpoint indices from coarse to fine resolution.

    Creates a boolean map at coarse resolution marking footpoint locations,
    upgrades it to fine resolution using healpix ud_grade, and returns the
    fine resolution indices.

    Parameters
    ----------
    coarse_nside : int
        Healpix nside parameter for coarse resolution.
    fine_nside : int
        Healpix nside parameter for fine resolution.
    footpoint_indices : np.ndarray
        Array of pixel indices at coarse resolution.

    Returns
    -------
    np.ndarray
        Array of pixel indices at fine resolution corresponding to upgraded footpoints.

    Examples
    --------
    >>> coarse_indices = np.array([100, 200, 300])
    >>> fine_indices = upgrade_healpix_footpoints(128, 256, coarse_indices)
    >>> len(fine_indices)  # Typically 4x more pixels at higher resolution
    """
    # Create boolean map: True where OFL footpoints are located
    coarse_map = np.zeros(hp.nside2npix(coarse_nside), dtype=bool)
    coarse_map[footpoint_indices] = True

    # Upgrade the boolean map to fine resolution
    fine_map = hp.pixelfunc.ud_grade(coarse_map, fine_nside)

    # Get indices where the fine map is True
    fine_footpoint_indices = np.where(fine_map)[0]

    logger.debug(
        f"Upgraded {len(footpoint_indices)} coarse footpoints (nside={coarse_nside}) "
        f"to {len(fine_footpoint_indices)} fine footpoints (nside={fine_nside})"
    )

    return fine_footpoint_indices


def create_fine_resolution_subset_from_coarse_OFL(
    coarse_experiment: "ExperimentConfig",
    fine_experiment: "ExperimentConfig",
    R_LC: float,
    coarse_lines_dir: Union[str, Path],
    fine_grids_dir: Union[str, Path],
    h_LC_coeff: float = 3**1.5 / 2.0,
    template: Optional["FieldLine"] = None,
) -> Tuple[Path, "GridConfig"]:
    """Create fine resolution subset grid from coarse experiment OFL footpoints.

    This function:
    1. Loads OFL data from coarse experiment
    2. Extracts OFL footpoints
    3. Creates fine resolution grid
    4. Upgrades footpoints from coarse to fine resolution
    5. Creates and saves subset grid
    6. Returns grid filepath and updated grid config

    Parameters
    ----------
    coarse_experiment : ExperimentConfig
        Coarse resolution experiment config.
    fine_experiment : ExperimentConfig
        Fine resolution experiment config.
    R_LC : float
        Light cylinder radius for OFL calculation.
    coarse_lines_dir : Union[str, Path]
        Directory containing coarse experiment batch files.
    fine_grids_dir : Union[str, Path]
        Directory where fine subset grid will be saved.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    template : Optional[FieldLine], optional
        Optional FieldLine template. Defaults to template_OFL_subset.

    Returns
    -------
    Tuple[Path, GridConfig]
        Tuple of (grid_filepath, updated_grid_config).

    Raises
    ------
    ValueError
        If OFL data cannot be loaded or grids cannot be created.
    """
    from psr_magpy.line.field_line_template import FieldLineTemplate
    from psr_magpy.experiment.grid_config import GridConfig, HealpixGridConfig

    # Use minimal template if not provided
    if template is None:
        template = FieldLineTemplate.template_OFL_subset()

    # Load OFL data from coarse experiment
    logger.info(f"Loading OFL data from coarse experiment: {coarse_experiment.name}")
    OFL_data = coarse_experiment.load_combined_batch_data(template, str(coarse_lines_dir))

    if OFL_data is None:
        raise PSRMagPyExperimentError(f"No OFL data found for experiment {coarse_experiment.name}")

    # Calculate OFL footpoints
    OFL_footpoints = OFL_data.get_OFL_footpoints(R_LC, h_LC_coeff=h_LC_coeff)
    logger.info(f"Found {len(OFL_footpoints)} OFL footpoints")

    if len(OFL_footpoints) == 0:
        raise PSRMagPyExperimentError(f"No OFL footpoints found for R_LC={R_LC:.2f}")

    # Create coarse grid to get nside
    coarse_grid_initial = coarse_experiment.grid.create_grid()
    if coarse_grid_initial is None:
        raise PSRMagPyExperimentError(f"Could not create coarse grid for experiment {coarse_experiment.name}")

    # Create fine grid
    fine_grid_initial = fine_experiment.grid.create_grid()
    if fine_grid_initial is None:
        raise PSRMagPyExperimentError(f"Could not create fine grid for experiment {fine_experiment.name}")

    # Verify both grids are healpix
    if not hasattr(coarse_grid_initial, "nside") or not hasattr(fine_grid_initial, "nside"):
        raise PSRMagPyGridError("Both grids must be HealpixGrid for footpoint upgrade")

    # Upgrade footpoints from coarse to fine resolution
    # Type ignore: nside exists on HealpixGrid subclass, verified by hasattr check above
    coarse_nside = coarse_grid_initial.nside  # type: ignore[attr-defined]
    fine_nside = fine_grid_initial.nside  # type: ignore[attr-defined]
    OFL_footpoints_fine = upgrade_healpix_footpoints(
        coarse_nside, fine_nside, OFL_footpoints
    )

    logger.info(
        f"Upgraded {len(OFL_footpoints)} coarse footpoints to "
        f"{len(OFL_footpoints_fine)} fine footpoints"
    )

    # Create subset grid
    fine_grid_subset = fine_grid_initial.subset_grid(OFL_footpoints_fine)

    # Save subset grid to file
    fine_grids_dir = Path(fine_grids_dir)
    fine_grids_dir.mkdir(parents=True, exist_ok=True)
    grid_filename = f"{fine_experiment.name}_subset_grid.fits"
    grid_filepath = fine_grids_dir / grid_filename

    # Remove existing file if it exists
    if grid_filepath.exists():
        grid_filepath.unlink()
        logger.debug(f"Deleted existing grid file: {grid_filepath}")

    fine_grid_subset.save_to_fits(str(grid_filepath))
    logger.info(f"Saved subset grid: {grid_filepath}")

    # Create updated grid config
    subset_healpix_config = HealpixGridConfig(
        nside=fine_nside,
        grid_file_path=str(grid_filepath),
        subset=True,
    )
    updated_grid_config = GridConfig(healpix=subset_healpix_config)

    return grid_filepath, updated_grid_config


def save_subset_grid(
    grid: FootpointGrid,
    experiment_name: str,
    grids_dir: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """Save subset grid to file with standardized naming.

    Parameters
    ----------
    grid : FootpointGrid
        The subset grid to save.
    experiment_name : str
        Name of the experiment (used for filename).
    grids_dir : Union[str, Path]
        Directory where grid will be saved.
    overwrite : bool, optional
        If True, delete existing file before saving. Default is True.

    Returns
    -------
    Path
        Path to saved grid file.

    Examples
    --------
        >>> grid = coarse_grid.subset_grid(indices)
        >>> grid_path = save_subset_grid(grid, "experiment_001", "/path/to/grids")
    """
    grids_dir = Path(grids_dir)
    grids_dir.mkdir(parents=True, exist_ok=True)

    grid_filename = f"{experiment_name}_subset_grid.fits"
    grid_filepath = grids_dir / grid_filename

    # Remove existing file if it exists and overwrite is True
    if overwrite and grid_filepath.exists():
        grid_filepath.unlink()
        logger.debug(f"Deleted existing grid file: {grid_filepath}")

    grid.save_to_fits(str(grid_filepath))
    logger.info(f"Saved subset grid: {grid_filepath}")

    return grid_filepath


def create_subset_grid_config(
    original_grid: FootpointGrid,
    subset_indices: np.ndarray,
    experiment_name: str,
    grid_type: str,
) -> "GridConfig":
    """Create a grid config for a subset grid that will be automatically saved.

    Parameters
    ----------
    original_grid : FootpointGrid
        The original grid to create a subset from.
    subset_indices : np.ndarray
        Indices to include in the subset.
    experiment_name : str
        Name of the experiment (used for filename).
    grid_type : str
        Type of grid ('spherical' or 'healpix' only).

    Returns
    -------
    GridConfig
        Configuration for the subset grid.

    Examples
    --------
    >>> indices = np.array([0, 1, 4])
    >>> config = create_subset_grid_config(grid, indices, "dipole_000", "healpix")
    """
    from psr_magpy.experiment.grid_config import (
        GridConfig,
        SphericalGridConfig,
        HealpixGridConfig,
    )

    # Create subset grid
    subset_grid = original_grid.subset_grid(subset_indices)

    # Generate filename
    grid_filename = f"{experiment_name}_subset_grid.fits"
    grid_path = f"grids/{grid_filename}"

    # Create appropriate config based on grid type
    # Type ignore: nlon, nlat, pole_lat exist on SphericalGrid subclass
    # Type ignore: nside exists on HealpixGrid subclass
    if grid_type.lower() == "spherical":
        config: Union[SphericalGridConfig, HealpixGridConfig] = SphericalGridConfig(
            nlon=original_grid.nlon,  # type: ignore[attr-defined]
            nlat=original_grid.nlat,  # type: ignore[attr-defined]
            pole_lat=original_grid.pole_lat.deg,  # type: ignore[attr-defined]
            subset=True,
            grid_file_path=grid_path,
        )
    elif grid_type.lower() == "healpix":
        config = HealpixGridConfig(
            nside=original_grid.nside, subset=True, grid_file_path=grid_path  # type: ignore[attr-defined]
        )
    else:
        raise ValueError(
            f"Subset grids only supported for 'spherical' and 'healpix' types, got: {grid_type}"
        )

    return GridConfig(**{grid_type: config})
