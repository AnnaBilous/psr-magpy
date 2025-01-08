from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Any

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle, PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from scipy.spatial.transform import Rotation as rot
import logging


# Setup logger at the top of the file
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
    @abstractmethod
    def prepare_footpoints(self) -> CartesianRepresentation:
        pass

class SphericalGrid(BaseGrid):
    def __init__(self, params: SphericalGridParams):
        self.nlon = params.nlon
        self.nlat = params.nlat
        self.pole_lat = Angle(params.pole_lat * u.deg)

    def prepare_footpoints(self) -> CartesianRepresentation:
        phi_grid = np.linspace(0, 2 * np.pi, self.nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, self.nlat)
        print(f"Latitude resolution {np.rad2deg(theta_grid[1] - theta_grid[0])} degrees")

        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        
        # Remove duplicate points at poles
        duplicate = np.where((phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi)))
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
        self.nside = params.nside

    def prepare_footpoints(self) -> CartesianRepresentation:
        npix = hp.nside2npix(self.nside)
        print(
            f" Approx res at NSIDE {self.nside} is {hp.nside2resol(self.nside, arcmin=True) / 60:.2f} deg ({npix} points)"
        )
        
        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(npix))
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)
        
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=1)
        return footpoints.represent_as(CartesianRepresentation)

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self.nside})"

class RandomGrid(BaseGrid):
    def __init__(self, params: RandomGridParams):
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
    
    def __init__(self, grid_type: GridType, params: GridParams) -> None:
        logger.info(f"Initializing FootpointGrid with type {grid_type}")
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
            GridType.RANDOM: RandomGrid
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
            self.footpoint_coordinates: CartesianRepresentation = self.grid.prepare_footpoints()
            self.nline_ori: int = len(self.footpoint_coordinates.x)
            self.indices: np.ndarray = np.arange(self.nline_ori)
            self.nline: int = self.nline_ori
            logger.info(f"Successfully created grid with {self.nline} points")
        except Exception as e:
            logger.error(f"Failed to calculate grid points: {e}")
            raise

    def subset_grid(self, grid_indices: np.ndarray) -> None:
        logger.info(f"Subsetting grid from {self.nline_ori} points")
        invalid_indices: set[int] = set(grid_indices) - set(self.indices)
        if invalid_indices:
            error_msg = (
                f"Invalid indices found: {invalid_indices}. "
                f"Indices must be between 0 and {len(self.indices)-1}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            self.partial = True
            self.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
            self.indices = grid_indices
            self.nline = len(grid_indices)
            logger.info(f"Successfully subset grid to {self.nline} points")
        except Exception as e:
            logger.error(f"Failed to subset grid: {e}")
            raise

    def plot_footpoint_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = ".",
        color: str = "r",
        **kwargs: Any
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
        spherical = self.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
        return spherical.theta.to(u.deg)

    @property
    def longitude(self) -> np.ndarray:
        """Longitude of footpoints in degrees.
        
        Returns:
            np.ndarray: Array of longitude values from 0 to 360 degrees
        """
        spherical = self.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
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
