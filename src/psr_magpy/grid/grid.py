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

logger = get_logger(__name__, console_format_brief=True)

T = TypeVar("T", bound="FootpointGrid")


class GridType(Enum):
    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"


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
        logger.debug("Initializing FootpointGrid")
        self.partial: bool = False
        self.footpoint_coordinates: Optional[CartesianRepresentation] = None
        self.nline_ori: Optional[int] = None
        self.indices: Optional[np.ndarray] = None
        self.nline: Optional[int] = None
        self.grid_type: Optional[GridType] = None
        logger.debug("FootpointGrid initialized")

    @classmethod
    def load_from_fits(cls: Type[T], fitsname: str) -> T:
        """Load grid from a FITS file.

        Args:
            fitsname: Path to FITS file

        Returns:
            Instance of the specific grid class that called this method
        """
        logger.info(f"Loading grid from {fitsname}")
        try:
            with pf.open(fitsname) as hdulist:
                header_main = hdulist[0].header
                grid_type = GridType(header_main["GRIDTYPE"])
                logger.debug(f"Grid type from file: {grid_type}")

                # Delegate creation to specific grid class
                grid = cls._create_grid_from_header(header_main, grid_type)

                # Load coordinates
                logger.debug("Loading coordinates from FITS file")
                footpoint_coordinates = hdulist["FTP_CRD"].data
                grid.footpoint_coordinates = CartesianRepresentation(
                    footpoint_coordinates
                )

                # Load indices
                logger.debug("Loading indices from FITS file")
                grid.indices = np.array([row[0] for row in hdulist["FTP_ID"].data])
                grid.partial = header_main["PARTIAL"]
                grid.nline = len(grid.indices)
                
                logger.debug(f"Successfully loaded grid with {grid.nline} points")
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
        }
        grid_class = grid_classes.get(grid_type)
        if grid_class is None:
            logger.error(f"Unknown grid type: {grid_type}")
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        logger.debug(f"Creating {grid_class.__name__} from header")
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
        logger.info(f"Setting up grid of type {self.grid_type}")
        try:
            start_time = time.perf_counter()
            self.footpoint_coordinates = self._prepare_footpoints()
            self.nline_ori = len(self.footpoint_coordinates.x)
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori
            elapsed = time.perf_counter() - start_time
            logger.debug(
                f"Grid setup complete: {self.nline} points in {elapsed:.2f}s"
            )
            logger.debug(f"Grid details: type={self.grid_type}, points={self.nline}")
        except Exception as e:
            logger.error(f"Failed to set up grid: {str(e)}")
            logger.debug("Grid setup error details", exc_info=True)
            raise

    def subset_grid(
        self, grid_indices: np.ndarray
    ) -> Union["SphericalGrid", "HealpixGrid", "RandomGrid"]:
        """Create a new grid instance with subset of indices.

        Args:
            grid_indices: Array of indices to include in the subset

        Returns:
            FootpointGrid: New grid instance with subset of points

        Raises:
            ValueError: If any indices are invalid
        """
        logger.info(f"Creating subset with {len(grid_indices)} points")
        logger.debug(f"Original grid size: {self.nline_ori} points")
        
        if logger.isEnabledFor(logging.DEBUG):
            # Only log indices in debug mode to avoid wall of text
            logger.debug(f"Subset indices: {grid_indices[:5]}... (showing first 5)")

        invalid_indices: set[int] = set(grid_indices) - set(self.indices)
        if invalid_indices:
            error_msg = (
                f"Invalid indices found: {invalid_indices}. "
                f"Indices must be between 0 and {len(self.indices)-1}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Create a new instance with same parameters
            if isinstance(self, SphericalGrid):
                logger.debug(
                    f"Creating SphericalGrid subset with nlon={self.nlon}, nlat={self.nlat}"
                )
                new_grid = SphericalGrid(self.nlon, self.nlat, self.pole_lat.deg)
            elif isinstance(self, HealpixGrid):
                logger.debug(f"Creating HealpixGrid subset with nside={self.nside}")
                new_grid = HealpixGrid(self.nside)
            else:  # RandomGrid
                logger.debug(
                    f"Creating RandomGrid subset with nline_ori={self.nline_ori}"
                )
                new_grid = RandomGrid(self.nline_ori)

            # Set subset properties
            new_grid.partial = True
            new_grid.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
            new_grid.indices = grid_indices
            new_grid.nline = len(grid_indices)

            logger.debug(f"Subset grid created with {new_grid.nline} points")
            return new_grid

        except Exception as e:
            logger.error(f"Failed to create subset grid: {str(e)}")
            logger.debug("Subset creation error details", exc_info=True)
            raise

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
            logger.debug("Adding grid-specific parameters to header")
            self._add_params_to_header(header_main)

            # Save coordinates
            logger.debug("Saving coordinates to FITS file")
            data = CCDData(
                self.footpoint_coordinates.xyz, unit=self.footpoint_coordinates.xyz.unit
            )
            hdu_footpoints = data.to_hdu(as_image_hdu=True)[0]
            hdu_footpoints.name = "FTP_CRD"

            # Save indices
            logger.debug("Saving indices to FITS file")
            col = pf.Column(name="footpoint_id", format="K", array=self.indices)
            hdu_indices = pf.BinTableHDU.from_columns([col], name="FTP_ID")

            # Write file
            hdul = pf.HDUList([hdu_main, hdu_indices, hdu_footpoints])
            hdul.writeto(fitsname, overwrite=True)
            logger.info(f"Grid successfully saved to {fitsname}")

        except Exception as e:
            logger.error(f"Failed to save grid to {fitsname}: {e}")
            logger.debug("Save error details", exc_info=True)
            raise

    def plot_footpoint_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = ".",
        color: str = "r",
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot footpoint coordinates."""
        logger.debug("Plotting footpoint coordinates")
        try:
            if ax is None:
                _, ax = plt.subplots()

            ax.plot(self.longitude, self.colatitude, marker, color=color, **kwargs)

            ax.set_xlim(0, 360)
            ax.set_xlabel("Longitude, deg")
            ax.set_ylim(180, 0)
            ax.set_ylabel("Latitude, deg")

            logger.debug("Footpoint coordinates plotted successfully")
            return ax
        except Exception as e:
            logger.error(f"Failed to plot footpoint coordinates: {e}")
            logger.debug("Plot error details", exc_info=True)
            raise


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
        logger.info(
            f"Initializing SphericalGrid: nlon={nlon}, nlat={nlat}, pole_lat={pole_lat}"
        )
        super().__init__()
        self.grid_type = GridType.SPHERICAL
        if nlon <= 0 or nlat <= 0:
            error_msg = (
                f"Grid dimensions must be positive, got nlon={nlon}, nlat={nlat}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.nlon = nlon
        self.nlat = nlat
        self.pole_lat = Angle(pole_lat * u.deg)
        self._set_up_grid()
        logger.debug("SphericalGrid initialization complete")

    def __repr__(self) -> str:
        return (
            f"SphericalGrid(nlon={self.nlon}, nlat={self.nlat}, "
            f"pole_lat={self.pole_lat.deg:.1f} deg)"
        )

    @classmethod
    def from_header(cls, header: pf.Header) -> "SphericalGrid":
        """Create SphericalGrid from FITS header."""
        logger.debug(f"Creating SphericalGrid from header: NLON={header['NLON']}, NLAT={header['NLAT']}")
        return cls(nlon=header["NLON"], nlat=header["NLAT"], pole_lat=header["POLELAT"])

    def _prepare_footpoints(self) -> CartesianRepresentation:
        logger.debug("Preparing SphericalGrid footpoints")
        start_time = time.perf_counter()

        # Create grid
        phi_grid = np.linspace(0, 2 * np.pi, self.nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, self.nlat)
        lat_resolution = np.rad2deg(theta_grid[1] - theta_grid[0])
        logger.debug(f"Latitude resolution: {lat_resolution:.2f} degrees")

        # Create mesh and handle poles
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)

        duplicate = np.where(
            (phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi))
        )
        logger.debug(f"Removing {len(duplicate[0])} duplicate pole points")

        theta_pruned = np.delete(theta_mesh, duplicate)
        phi_pruned = np.delete(phi_mesh, duplicate)

        # Convert to Cartesian
        theta = Angle(theta_pruned * u.rad)
        phi = Angle(phi_pruned * u.rad)
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        footpoints_cart = footpoints.represent_as(CartesianRepresentation)

        # Apply rotation if needed
        if not (self.pole_lat == 0):
            logger.debug(
                f"Applying rotation for pole_lat={self.pole_lat.deg:.2f} degrees"
            )
            rotation_operator = rot.from_euler("yzx", [self.pole_lat.rad, 0, 0])
            rotated_xyz = rotation_operator.apply(footpoints_cart.xyz.value.T).T
            footpoints_cart = CartesianRepresentation(
                rotated_xyz, unit=footpoints_cart.xyz.unit
            )

        elapsed = time.perf_counter() - start_time
        logger.debug(f"Footpoint preparation completed in {elapsed:.2f} seconds")
        return footpoints_cart

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NLON", self.nlon, "Number of longitude divisions")
        header.set("NLAT", self.nlat, "Number of latitude divisions")
        header.set("POLELAT", self.pole_lat.deg, "Latitude of grid pole, degree")


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
        logger.debug(f"Initializing HealpixGrid with nside={nside}")
        super().__init__()
        self.grid_type = GridType.HEALPIX
        if nside <= 0 or not (nside & (nside - 1) == 0):
            logger.error(f"Invalid NSIDE value: {nside} (must be a positive power of 2)")
            raise ValueError("NSIDE must be a positive power of 2")

        self.nside = nside
        self._set_up_grid()
        logger.debug("HealpixGrid initialization complete")

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self.nside})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NSIDE", self.nside, "Healpix parameter nside")

    def _prepare_footpoints(self) -> CartesianRepresentation:
        logger.debug(f"Preparing HealpixGrid footpoints with nside={self.nside}")
        npix = hp.nside2npix(self.nside)
        resolution_deg = hp.nside2resol(self.nside, arcmin=True) / 60
        
        logger.info(f"HEALPix grid: nside={self.nside}, resolution={resolution_deg:.2f}°, points={npix}")

        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(npix))
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)

        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        return footpoints.represent_as(CartesianRepresentation)

    @classmethod
    def from_header(cls, header: pf.Header) -> "HealpixGrid":
        """Create HealpixGrid from FITS header."""
        logger.debug(f"Creating HealpixGrid from header: NSIDE={header['NSIDE']}")
        return cls(nside=header["NSIDE"])


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
        logger.info(f"Initializing RandomGrid with {nline_ori} points")
        super().__init__()
        self.grid_type = GridType.RANDOM
        if nline_ori <= 0:
            logger.error(f"Invalid number of points: {nline_ori} (must be positive)")
            raise ValueError("Number of random points must be positive")

        self.nline_ori = nline_ori
        self._set_up_grid()
        logger.debug("RandomGrid initialization complete")

    def __repr__(self) -> str:
        return f"RandomGrid(nline_ori={self.nline_ori})"

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for random grid

    def _prepare_footpoints(self) -> CartesianRepresentation:
        logger.debug(f"Generating {self.nline_ori} random points on sphere")
        # Muller method for random points on a sphere
        u: np.ndarray = np.random.normal(loc=0, scale=1, size=(3, self.nline_ori))
        norm: np.ndarray = (np.sum(u**2, axis=0)) ** 0.5
        x, y, z = u / norm
        return CartesianRepresentation(x, y, z)

    @classmethod
    def from_header(cls, header: pf.Header) -> "RandomGrid":
        """Create RandomGrid from FITS header."""
        logger.debug(f"Creating RandomGrid from header: NLINE={header['NLINE']}")
        return cls(nline_ori=header["NLINE"])
