from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Any
import logging

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle, PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from astropy.io import fits as pf
from astropy.nddata import CCDData
from scipy.spatial.transform import Rotation as rot

logger = logging.getLogger(__name__)


class GridType(Enum):
    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"


@dataclass
class SphericalGridParams:
    nlon: int
    nlat: int
    pole_lat: float = 0  # optional with default


@dataclass
class HealpixGridParams:
    nside: int


@dataclass
class RandomGridParams:
    nline_ori: int


# Type alias for parameter types
GridParams = Union[SphericalGridParams, HealpixGridParams, RandomGridParams]


class BaseGrid(ABC):
    def __init__(self, params: GridParams):
        self.params = params  # Store the params

    @abstractmethod
    def prepare_footpoints(self) -> CartesianRepresentation:
        pass


class GridConfigValidator:
    @staticmethod
    def validate_spherical_params(params: SphericalGridParams) -> None:
        if params.nlon <= 0 or params.nlat <= 0:
            raise ValueError("Grid dimensions must be positive")

    @staticmethod
    def validate_healpix_params(params: HealpixGridParams) -> None:
        if params.nside <= 0 or not (params.nside & (params.nside - 1) == 0):
            raise ValueError("NSIDE must be a positive power of 2")

    @staticmethod
    def validate_random_params(params: RandomGridParams) -> None:
        if params.nline_ori <= 0:
            raise ValueError("Number of random points must be positive")


class SphericalGrid(BaseGrid):
    def __init__(self, params: SphericalGridParams):
        super().__init__(params)  # Call parent constructor
        GridConfigValidator.validate_spherical_params(params)
        self.nlon = params.nlon
        self.nlat = params.nlat
        self.pole_lat = Angle(params.pole_lat * u.deg)

    def prepare_footpoints(self) -> CartesianRepresentation:
        phi_grid = np.linspace(0, 2 * np.pi, self.nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, self.nlat)
        print(
            f"Latitude resolution {np.rad2deg(theta_grid[1] - theta_grid[0])} degrees"
        )

        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)

        # Remove duplicate points at poles
        duplicate = np.where(
            (phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi))
        )
        theta_pruned = np.delete(theta_mesh, duplicate)
        phi_pruned = np.delete(phi_mesh, duplicate)

        theta = Angle(theta_pruned * u.rad)
        phi = Angle(phi_pruned * u.rad)

        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        footpoints_cart = footpoints.represent_as(CartesianRepresentation)

        if not (self.pole_lat == 0):
            rotation_operator = rot.from_euler("yzx", [self.pole_lat.rad, 0, 0])
            xyz = rotation_operator.apply(footpoints_cart.xyz.T)
            return CartesianRepresentation(xyz.T)

        return footpoints_cart

    def __repr__(self) -> str:
        return (
            f"SphericalGrid(nlon={self.nlon}, nlat={self.nlat}, "
            f"pole_lat={self.pole_lat.deg:.1f} deg)"
        )


class HealpixGrid(BaseGrid):
    def __init__(self, params: HealpixGridParams):
        super().__init__(params)  # Call parent constructor
        self.nside = params.nside

    def prepare_footpoints(self) -> CartesianRepresentation:
        npix = hp.nside2npix(self.nside)
        #print(
        #    f" Approx res at NSIDE {self.nside} is {hp.nside2resol(self.nside, arcmin=True) / 60:.2f} deg ({npix} points)"
        #)

        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(npix))
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)

        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        return footpoints.represent_as(CartesianRepresentation)

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self.nside})"


class RandomGrid(BaseGrid):
    def __init__(self, params: RandomGridParams):
        super().__init__(params)  # Call parent constructor
        self.nline_ori = params.nline_ori

    def prepare_footpoints(self) -> CartesianRepresentation:
        # Muller method for random points on a sphere
        u = np.random.normal(loc=0, scale=1, size=(3, self.nline_ori))
        norm = (np.sum(u**2, axis=0)) ** 0.5
        (x, y, z) = u / norm
        return CartesianRepresentation(x, y, z)

    def __repr__(self) -> str:
        return f"RandomGrid(nline_ori={self.nline_ori})"


class FootpointGrid:
    """A class for managing grids of footpoints on a sphere."""

    def __init__(self, grid_type: Union[GridType, str], params: GridParams) -> None:
        logger.info(f"Initializing FootpointGrid with type {grid_type}")

        # Convert string to GridType if needed
        if isinstance(grid_type, str):
            try:
                grid_type = GridType(grid_type.lower())
            except ValueError:
                logger.error(f"Invalid grid type: {grid_type}")
                raise ValueError(
                    f"Invalid grid type: {grid_type}. Must be one of {[t.value for t in GridType]}"
                )

        self.grid_type: GridType = grid_type
        try:
            self.grid: BaseGrid = self.__create_grid(grid_type, params)
        except Exception as e:
            logger.error(f"Failed to create grid: {e}")
            raise
        self.partial: bool = False
        self.calculate_grid()

    def __create_grid(self, grid_type: GridType, params: GridParams) -> BaseGrid:
        grid_classes: dict[GridType, type[BaseGrid]] = {
            GridType.SPHERICAL: SphericalGrid,
            GridType.HEALPIX: HealpixGrid,
            GridType.RANDOM: RandomGrid,
        }
        grid_class = grid_classes.get(grid_type)
        if not grid_class:
            logger.error(f"Unknown grid type: {grid_type}")
            raise ValueError(f"Unknown grid type: {grid_type}")
        logger.debug(f"Creating grid of type {grid_class.__name__}")
        return grid_class(params)

    def calculate_grid(self) -> None:
        logger.info("Calculating grid points")
        try:
            self.footpoint_coordinates: CartesianRepresentation = (
                self.grid.prepare_footpoints()
            )
            self.nline_ori: int = len(self.footpoint_coordinates.x)
            self.indices: np.ndarray = np.arange(self.nline_ori)
            self.nline: int = self.nline_ori
            logger.info(f"Successfully created grid with {self.nline} points")
        except Exception as e:
            logger.error(f"Failed to calculate grid points: {e}")
            raise

    def create_subset_grid(self, grid_indices: np.ndarray) -> "FootpointGrid":
        """Create a new FootpointGrid containing only the specified indices.

        Args:
            grid_indices: Array of indices to include in the subset

        Returns:
            A new FootpointGrid instance containing only the specified points

        Raises:
            ValueError: If invalid indices are provided
        """
        logger.info(f"Creating subset grid from {self.nline_ori} points")
        invalid_indices: set[int] = set(grid_indices) - set(self.indices)
        if invalid_indices:
            error_msg = (
                f"Invalid indices found: {invalid_indices}. "
                f"Indices must be between 0 and {len(self.indices)-1}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Create new grid with same parameters but subset of points
            subset = FootpointGrid(grid_type=self.grid_type, params=self.grid.params)

            # Update subset grid attributes
            subset.partial = True
            subset.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
            subset.indices = grid_indices
            subset.nline = len(grid_indices)
            subset.nline_ori = self.nline_ori  # Keep original size for reference

            logger.info(f"Successfully created subset grid with {subset.nline} points")
            return subset

        except Exception as e:
            logger.error(f"Failed to create subset grid: {e}")
            raise

    def plot_footpoint_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = ".",
        color: str = "r",
        **kwargs: Any,
    ) -> plt.Axes:
        logger.debug("Plotting footpoint coordinates")
        try:
            if ax is None:
                _, ax = plt.subplots()

            ax.plot(self.longitude, self.colatitude, marker, color=color, **kwargs)

            ax.set_xlim(0, 360)
            ax.set_xlabel("Longitude, deg")
            ax.set_ylim(180, 0)
            ax.set_ylabel("Latitude, deg")

            logger.debug("Successfully plotted footpoint coordinates")
            return ax
        except Exception as e:
            logger.error(f"Failed to plot footpoint coordinates: {e}")
            raise

    @property
    def colatitude(self) -> np.ndarray:
        """Latitude of footpoints in degrees.

        Returns:
            np.ndarray: Array of latitude values from 0 to 180 degrees
        """
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.theta.to(u.deg)

    @property
    def longitude(self) -> np.ndarray:
        """Longitude of footpoints in degrees.

        Returns:
            np.ndarray: Array of longitude values from 0 to 360 degrees
        """
        spherical = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        return spherical.phi.to(u.deg)

    """
    @property
    def xyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Cartesian coordinates of footpoints.
        
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, z coordinates
        
        return (
            self.footpoint_coordinates.x.value,
            self.footpoint_coordinates.y.value,
            self.footpoint_coordinates.z.value
        )
    """

    def __repr__(self) -> str:
        status = "partial" if self.partial else "full"
        points = f"{self.nline}/{self.nline_ori}" if self.partial else str(self.nline)
        return (
            f"FootpointGrid(type={self.grid_type.value}, "
            f"points={points}, "
            f"status={status})"
        )

    def save_to_fits(self, fitsname: str) -> None:
        """Save grid to a FITS file.

        Args:
            fitsname: Path to the output FITS file

        Raises:
            OSError: If file cannot be written
        """
        logger.info(f"Saving grid to {fitsname}")
        try:
            hdu_main = pf.PrimaryHDU()
            header_main = hdu_main.header
            header_main.set("GRIDTYPE", self.grid_type.value, "Grid type")
            header_main.set(
                "NLINE", self.nline_ori, "Number of field lines in a full grid"
            )
            header_main.set("PARTIAL", self.partial, "Partial grid")

            # Add grid-specific parameters
            if self.grid_type == GridType.SPHERICAL:
                header_main.set("NLON", self.grid.nlon, "Number of longitude divisions")
                header_main.set("NLAT", self.grid.nlat, "Number of latitude divisions")
                header_main.set(
                    "POLELAT", self.grid.pole_lat.deg, "Latitude of grid pole, degree"
                )
            elif self.grid_type == GridType.HEALPIX:
                header_main.set("NSIDE", self.grid.nside, "Healpix parameter nside")

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
            logger.info(f"Successfully saved grid to {fitsname}")

        except Exception as e:
            logger.error(f"Failed to save grid to {fitsname}: {e}")
            raise

    def _load_coordinates(self, data: np.ndarray) -> CartesianRepresentation:
        """Convert array data to CartesianRepresentation.

        Args:
            data: Array of coordinate data from FITS file

        Returns:
            CartesianRepresentation: Coordinates in Cartesian form
        """
        try:
            return CartesianRepresentation(data)
        except Exception as e:
            logger.error(f"Failed to convert coordinates: {e}")
            raise ValueError("Invalid coordinate data in FITS file") from e

    @classmethod
    def from_fits(cls, fitsname: str) -> "FootpointGrid":
        """Load grid from a FITS file.

        Args:
            fitsname: Path to the FITS file

        Returns:
            FootpointGrid: Loaded grid object

        Raises:
            OSError: If file cannot be read
            ValueError: If file format is invalid
        """
        logger.info(f"Loading grid from {fitsname}")
        try:
            with pf.open(fitsname) as hdulist:
                header_main = hdulist[0].header

                try:
                    grid_type = GridType(header_main["GRIDTYPE"])
                except ValueError as e:
                    logger.error(
                        f"Invalid grid type in {fitsname}: {header_main['GRIDTYPE']}"
                    )
                    raise ValueError(
                        f"Invalid grid type in FITS file: {header_main['GRIDTYPE']}"
                    ) from e

                # Create parameters dictionary based on grid type
                parameters_dict = {}
                if grid_type == GridType.SPHERICAL:
                    parameters_dict = {
                        "nlon": header_main["NLON"],
                        "nlat": header_main["NLAT"],
                        "pole_lat": header_main["POLELAT"],
                    }
                elif grid_type == GridType.HEALPIX:
                    parameters_dict = {"nside": header_main["NSIDE"]}
                else:  # GridType.RANDOM
                    parameters_dict = {"nline_ori": header_main["NLINE"]}

                # Create appropriate parameter object
                if grid_type == GridType.SPHERICAL:
                    params = SphericalGridParams(**parameters_dict)
                elif grid_type == GridType.HEALPIX:
                    params = HealpixGridParams(**parameters_dict)
                else:  # random
                    params = RandomGridParams(**parameters_dict)

                # Create grid instance
                grid = cls(grid_type, params)

                # Load coordinates using helper method
                footpoint_coordinates = hdulist["FTP_CRD"].data
                grid.footpoint_coordinates = grid._load_coordinates(
                    footpoint_coordinates
                )

                # Load indices
                # There shouldn't be a need for this conversion. Investigate later.
                grid.indices = np.array([row[0] for row in hdulist["FTP_ID"].data])
                grid.partial = header_main["PARTIAL"]

                logger.info(f"Successfully loaded grid from {fitsname}")
                return grid

        except Exception as e:
            logger.error(f"Failed to load grid from {fitsname}: {e}")
            raise
