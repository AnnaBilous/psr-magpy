import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, TypeVar, Type, Union, cast

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
from pydantic import BaseModel, Field, validator, model_validator

logger = get_logger(__name__, console_format_brief=True)

T = TypeVar("T", bound="FootpointGrid")


class GridType(Enum):
    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"
    CUSTOM = "custom"


class FootpointGrid(ABC):
    """Abstract base class for different types of spherical grid point distributions.

    This class provides a framework for creating and managing different types of point
    distributions on a unit sphere, including regular spherical grids, HEALPix grids,
    and random distributions.

    Attributes
    ----------
    partial : bool
        Whether this is a subset of a full grid
    footpoint_coordinates : CartesianRepresentation, optional
        Cartesian coordinates of grid points
    nline_ori : int, optional
        Original number of points in the full grid
    indices : ndarray, optional
        Array of point indices
    nline : int, optional
        Current number of points in the grid
    grid_type : GridType, optional
        Type of grid (SPHERICAL, HEALPIX, or RANDOM)
    """

    def __init__(self) -> None:
        self.partial: bool = False
        self.footpoint_coordinates: Optional[CartesianRepresentation] = None
        self.nline_ori: Optional[int] = None
        self.indices: Optional[np.ndarray] = None
        self.nline: Optional[int] = None
        self.grid_type: Optional[GridType] = None

    @classmethod
    def load_from_fits(cls: Type[T], fitsname: str) -> T:
        """Load grid from a FITS file."""
        try:
            with pf.open(fitsname) as hdulist:
                header_main = hdulist[0].header
                grid_type = GridType(header_main["GRIDTYPE"])

                # Delegate creation to specific grid class
                grid = cls._create_grid_from_header(header_main, grid_type)

                # Load coordinates
                footpoint_coordinates = hdulist["FTP_CRD"].data
                grid.footpoint_coordinates = CartesianRepresentation(
                    footpoint_coordinates
                )

                # Load indices
                grid.indices = np.array([row[0] for row in hdulist["FTP_ID"].data])
                grid.partial = header_main["PARTIAL"]
                grid.nline = len(grid.indices)
                
                return grid

        except Exception as e:
            logger.error(f"Failed to load grid from {fitsname}: {e}")
            raise

    @classmethod
    def _create_grid_from_header(
        cls: Type[T], header: pf.Header, grid_type: GridType
    ) -> T:
        """Factory method to create appropriate grid instance."""
        grid_classes = {
            GridType.SPHERICAL: SphericalGrid,
            GridType.HEALPIX: HealpixGrid,
            GridType.RANDOM: RandomGrid,
            GridType.CUSTOM: CustomGrid,
        }
        grid_class = grid_classes.get(grid_type)
        if grid_class is None:
            logger.error(f"Unknown grid type: {grid_type}")
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        return cast(T, grid_class.from_header(header))

    @property
    def colatitude(self) -> np.ndarray:
        """Get the colatitude of all grid points.

        Returns
        -------
        ndarray
            Colatitude values in degrees, ranging from 0 (north pole) to 180 (south pole)
        """
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.theta.to(u.deg).value

    @property
    def longitude(self) -> np.ndarray:
        """Get the longitude of all grid points.

        Returns
        -------
        ndarray
            Longitude values in degrees, ranging from 0 to 360
        """
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.phi.to(u.deg).value

    @abstractmethod
    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Prepare the footpoint coordinates for this grid type."""
        pass

    def _set_up_grid(self) -> None:
        """Calculate grid points and initialize grid properties."""
        try:
            self.footpoint_coordinates = self._prepare_footpoints()
            self.nline_ori = len(self.footpoint_coordinates.x)
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori
        except Exception as e:
            logger.error(f"Failed to set up grid: {str(e)}")
            raise

    def subset_grid(self, grid_indices: np.ndarray) -> "FootpointGrid":
        """Create a new grid instance with subset of indices."""
        # Validate indices
        if not all(0 <= i < len(self.indices) for i in grid_indices):
            raise ValueError(f"Invalid indices. Must be between 0 and {len(self.indices)-1}")
        
        # Create new grid using copy
        new_grid = self.copy()
        
        # Set subset properties
        new_grid.partial = True
        new_grid.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
        new_grid.indices = grid_indices
        new_grid.nline = len(grid_indices)
        
        return new_grid

    def save_to_fits(self, fitsname: str) -> None:
        """Save grid to a FITS file."""
        logger.info(f"Saving grid to {fitsname}")
        try:
            hdu_main = pf.PrimaryHDU()
            header_main = hdu_main.header
            header_main.set("GRIDTYPE", self.grid_type.value, "Grid type")
            header_main.set(
                "NLINE", self.nline_ori, "Number of field lines in a full grid"
            )
            header_main.set("PARTIAL", self.partial, "Partial grid")

            # Add grid-specific parameters to header
            self._add_params_to_header(header_main)

            # Save coordinates
            data = CCDData(
                self.footpoint_coordinates.xyz, unit=self.footpoint_coordinates.xyz.unit
            )
            hdu_footpoints = data.to_hdu(as_image_hdu=True)[0]
            hdu_footpoints.name = "FTP_CRD"

            # Save indices
            col = pf.Column(name="footpoint_id", format="K", array=self.indices)
            hdu_indices = pf.BinTableHDU.from_columns([col], name="FTP_ID")

            # Write file
            hdul = pf.HDUList([hdu_main, hdu_indices, hdu_footpoints])
            hdul.writeto(fitsname, overwrite=True)
            logger.info(f"Grid successfully saved to {fitsname}")

        except Exception as e:
            logger.error(f"Failed to save grid to {fitsname}: {e}")
            raise

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
            logger.error(f"Failed to plot footpoint coordinates: {e}")
            raise

    @abstractmethod
    def copy(self) -> "FootpointGrid":
        """Create a copy of the grid with same parameters."""
        pass


class SphericalGridParams(BaseModel):
    nlon: int = Field(gt=0, description="Number of longitude divisions")
    nlat: int = Field(gt=0, description="Number of latitude divisions")
    pole_lat: float = Field(default=0.0, description="Latitude of grid pole")


class SphericalGrid(FootpointGrid):
    """Regular spherical grid with uniform spacing in longitude and latitude.

    Creates a grid of points on a unit sphere with regular spacing in longitude and
    latitude. Optionally, the grid can be rotated to have its pole at a specified latitude.

    Parameters
    ----------
    nlon : int
        Number of longitude divisions
    nlat : int
        Number of latitude divisions
    pole_lat : float, optional
        Latitude of the grid's pole in degrees, default is 0

    Attributes
    ----------
    nlon : int
        Number of longitude divisions
    nlat : int
        Number of latitude divisions
    pole_lat : Angle
        Latitude of grid's pole

    Raises
    ------
    ValueError
        If nlon or nlat are not positive integers
    """

    def __init__(self, nlon: int, nlat: int, pole_lat: float = 0):
        super().__init__()
        self.grid_type = GridType.SPHERICAL
        
        # Pydantic validation
        params = SphericalGridParams(nlon=nlon, nlat=nlat, pole_lat=pole_lat)
        
        self.nlon = params.nlon
        self.nlat = params.nlat
        self.pole_lat = Angle(params.pole_lat * u.deg)
        self._set_up_grid()

    def __repr__(self) -> str:
        return (
            f"SphericalGrid(nlon={self.nlon}, nlat={self.nlat}, "
            f"pole_lat={self.pole_lat.deg:.1f} deg)"
        )

    @classmethod
    def from_header(cls, header: pf.Header) -> "SphericalGrid":
        """Create SphericalGrid from FITS header."""
        return cls(nlon=header["NLON"], nlat=header["NLAT"], pole_lat=header["POLELAT"])

    def _prepare_footpoints(self) -> CartesianRepresentation:
        # Create grid
        phi_grid = np.linspace(0, 2 * np.pi, self.nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, self.nlat)
        lat_resolution = np.rad2deg(theta_grid[1] - theta_grid[0])

        # Create mesh and handle poles
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)

        duplicate = np.where(
            (phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi))
        )

        theta_pruned = np.delete(theta_mesh, duplicate)
        phi_pruned = np.delete(phi_mesh, duplicate)

        # Convert to Cartesian
        theta = Angle(theta_pruned * u.rad)
        phi = Angle(phi_pruned * u.rad)
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        footpoints_cart = footpoints.represent_as(CartesianRepresentation)

        # Apply rotation if needed
        if not (self.pole_lat == 0):
            rotation_operator = rot.from_euler("yzx", [self.pole_lat.rad, 0, 0])
            rotated_xyz = rotation_operator.apply(footpoints_cart.xyz.value.T).T
            footpoints_cart = CartesianRepresentation(
                rotated_xyz, unit=footpoints_cart.xyz.unit
            )

        return footpoints_cart

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NLON", self.nlon, "Number of longitude divisions")
        header.set("NLAT", self.nlat, "Number of latitude divisions")
        header.set("POLELAT", self.pole_lat.deg, "Latitude of grid pole, degree")

    def copy(self) -> "SphericalGrid":
        return SphericalGrid(self.nlon, self.nlat, self.pole_lat.deg)


class HealpixGridParams(BaseModel):
    nside: int = Field(gt=0, description="HEALPix resolution parameter")
    
    @validator('nside')
    def validate_power_of_two(cls, v):
        if not (v & (v - 1) == 0):
            raise ValueError(f"nside must be a power of 2, got {v}")
        return v


class HealpixGrid(FootpointGrid):
    """HEALPix grid for uniform spherical sampling.

    Creates a grid using the Hierarchical Equal Area isoLatitude Pixelization (HEALPix)
    scheme, which provides approximately uniform sampling of a sphere.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter, must be a power of 2

    Attributes
    ----------
    nside : int
        HEALPix resolution parameter

    Notes
    -----
    The number of pixels is 12 * nside^2. Resolution at given nside is
    approximately sqrt(4π/N) where N is the number of pixels.

    Raises
    ------
    ValueError
        If nside is not a positive power of 2
    """

    def __init__(self, nside: int):
        super().__init__()
        self.grid_type = GridType.HEALPIX
        
        # Pydantic validation
        params = HealpixGridParams(nside=nside)
        self.nside = params.nside
        self._set_up_grid()

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self.nside})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NSIDE", self.nside, "Healpix parameter nside")

    def _prepare_footpoints(self) -> CartesianRepresentation:
        npix = hp.nside2npix(self.nside)
        resolution_deg = hp.nside2resol(self.nside, arcmin=True) / 60
        
        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(npix))
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)

        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        return footpoints.represent_as(CartesianRepresentation)

    @classmethod
    def from_header(cls, header: pf.Header) -> "HealpixGrid":
        """Create HealpixGrid from FITS header."""
        return cls(nside=header["NSIDE"])

    def copy(self) -> "HealpixGrid":
        return HealpixGrid(self.nside)


class RandomGridParams(BaseModel):
    nline_ori: int = Field(gt=0, description="Number of random points")


class RandomGrid(FootpointGrid):
    """Random uniform distribution of points on a sphere.

    Creates a grid of randomly distributed points that are approximately uniformly
    distributed on a unit sphere using the Muller method.

    Parameters
    ----------
    nline_ori : int
        Number of random points to generate

    Notes
    -----
    Uses the Muller method which generates points by:
    1. Creating 3D normal distributions
    2. Normalizing to unit vectors
    This produces a uniform distribution on the sphere.

    Raises
    ------
    ValueError
        If nline_ori is not a positive integer
    """

    def __init__(self, nline_ori: int):
        super().__init__()
        self.grid_type = GridType.RANDOM
        
        # Pydantic validation
        params = RandomGridParams(nline_ori=nline_ori)
        self.nline_ori = params.nline_ori
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
        return cls(nline_ori=header["NLINE"])

    def copy(self) -> "RandomGrid":
        return RandomGrid(self.nline_ori)


class CustomGridParams(BaseModel):
    nline_ori: int = Field(gt=0, description="Number of points")
    coordinates: Optional[CartesianRepresentation] = Field(default=None, description="Initial coordinates")
    
    @model_validator(mode='after')
    def validate_coordinates_length(self):
        if self.coordinates is not None:
            if len(self.coordinates.x) != self.nline_ori:
                raise ValueError(f"Coordinates length {len(self.coordinates.x)} != nline_ori {self.nline_ori}")
        return self


class CustomGrid(FootpointGrid):
    """Custom grid with user-provided coordinates.

    Creates a grid with coordinates specified by the user. Initializes with NaN values
    that must be set either during initialization or via set_coordinates method.

    Parameters
    ----------
    nline_ori : int
        Number of points in the grid
    coordinates : CartesianRepresentation, optional
        Initial coordinates for grid points. If not provided, coordinates will be NaN.

    Notes
    -----
    User must provide valid coordinates before using the grid for calculations.
    Coordinates should be on a unit sphere.

    Raises
    ------
    ValueError
        If nline_ori is not positive or if provided coordinates have wrong shape
    """

    def __init__(self, nline_ori: int, coordinates: Optional[CartesianRepresentation] = None):
        super().__init__()
        self.grid_type = GridType.CUSTOM
        
        # Pydantic validation
        params = CustomGridParams(nline_ori=nline_ori, coordinates=coordinates)
        
        self.nline_ori = params.nline_ori
        self._set_coordinates(params.coordinates)
        self.indices = np.arange(self.nline_ori)
        self.nline = self.nline_ori

    def __repr__(self) -> str:
        return f"CustomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for custom grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        """Return current footpoint coordinates."""
        return self.footpoint_coordinates

    def set_coordinates(self, coordinates: CartesianRepresentation) -> None:
        """Set new coordinates for grid points.

        Parameters
        ----------
        coordinates : CartesianRepresentation
            New coordinates for grid points

        Raises
        ------
        ValueError
            If coordinates length doesn't match nline_ori
        """
        if len(coordinates.x) != self.nline_ori:
            error_msg = f"Coordinates length {len(coordinates.x)} != nline_ori {self.nline_ori}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.footpoint_coordinates = coordinates

    @classmethod
    def from_header(cls, header: pf.Header) -> "CustomGrid":
        """Create CustomGrid from FITS header."""
        return cls(nline_ori=header["NLINE"])

    def copy(self) -> "CustomGrid":
        return CustomGrid(self.nline_ori)