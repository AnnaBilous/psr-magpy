from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Any, TypeVar, Type
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

T = TypeVar('T', bound='FootpointGrid')

class GridType(Enum):
    SPHERICAL = "spherical"
    HEALPIX = "healpix"
    RANDOM = "random"

class FootpointGrid(ABC):
    """Base class for all grid types."""
    
    def __init__(self) -> None:
        self.partial: bool = False
        self.footpoint_coordinates: Optional[CartesianRepresentation] = None
        self.nline_ori: Optional[int] = None
        self.indices: Optional[np.ndarray] = None
        self.nline: Optional[int] = None
        self.grid_type: Optional[GridType] = None
        
    @abstractmethod
    def prepare_footpoints(self) -> CartesianRepresentation:
        pass

    def calculate_grid(self) -> None:
        """Calculate grid points and initialize grid properties."""
        logger.info("Calculating grid points")
        try:
            self.footpoint_coordinates = self.prepare_footpoints()
            self.nline_ori = len(self.footpoint_coordinates.x)
            self.indices = np.arange(self.nline_ori)
            self.nline = self.nline_ori
            logger.info(f"Successfully created grid with {self.nline} points")
        except Exception as e:
            logger.error(f"Failed to calculate grid points: {e}")
            raise

    def subset_grid(self, grid_indices: np.ndarray) -> 'FootpointGrid':
        """Create a new grid instance with subset of indices.
        
        Args:
            grid_indices: Array of indices to include in the subset
            
        Returns:
            FootpointGrid: New grid instance with subset of points
            
        Raises:
            ValueError: If any indices are invalid
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
            # Create a new instance with same parameters
            if isinstance(self, SphericalGrid):
                new_grid = SphericalGrid(self.nlon, self.nlat, self.pole_lat.deg)
            elif isinstance(self, HealpixGrid):
                new_grid = HealpixGrid(self.nside)
            else:  # RandomGrid
                new_grid = RandomGrid(self.nline_ori)
            
            # Set subset properties
            new_grid.partial = True
            new_grid.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
            new_grid.indices = grid_indices
            new_grid.nline = len(grid_indices)
            
            logger.info(f"Successfully created subset grid with {new_grid.nline} points")
            return new_grid
            
        except Exception as e:
            logger.error(f"Failed to create subset grid: {e}")
            raise

    def plot_footpoint_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = ".",
        color: str = "r",
        **kwargs: Any
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
            
            logger.debug("Successfully plotted footpoint coordinates")
            return ax
        except Exception as e:
            logger.error(f"Failed to plot footpoint coordinates: {e}")
            raise

    @property
    def colatitude(self) -> np.ndarray:
        """Latitude of footpoints in degrees."""
        spherical = self.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
        return spherical.theta.to(u.deg)

    @property
    def longitude(self) -> np.ndarray:
        """Longitude of footpoints in degrees."""
        spherical = self.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
        return spherical.phi.to(u.deg)

    def save_to_fits(self, fitsname: str) -> None:
        """Save grid to a FITS file."""
        logger.info(f"Saving grid to {fitsname}")
        try:
            hdu_main = pf.PrimaryHDU()
            header_main = hdu_main.header
            header_main.set("GRIDTYPE", self.grid_type.value, "Grid type")
            header_main.set("NLINE", self.nline_ori, "Number of field lines in a full grid")
            header_main.set("PARTIAL", self.partial, "Partial grid")

            # Add grid-specific parameters to header
            self._add_params_to_header(header_main)

            # Save coordinates
            data = CCDData(self.footpoint_coordinates.xyz, unit=self.footpoint_coordinates.xyz.unit)
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

    @abstractmethod
    def _add_params_to_header(self, header: pf.Header) -> None:
        """Add grid-specific parameters to FITS header."""
        pass

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
                
                # Delegate creation to specific grid class
                grid = cls._create_grid_from_header(header_main, grid_type)
                
                # Load coordinates
                footpoint_coordinates = hdulist["FTP_CRD"].data
                grid.footpoint_coordinates = CartesianRepresentation(footpoint_coordinates)
                
                # Load indices
                grid.indices = np.array([row[0] for row in hdulist["FTP_ID"].data])
                grid.partial = header_main["PARTIAL"]
                grid.nline = len(grid.indices)
                
                return grid
                    
        except Exception as e:
            logger.error(f"Failed to load grid from {fitsname}: {e}")
            raise

    @staticmethod
    def _create_grid_from_header(header: pf.Header, grid_type: GridType) -> 'FootpointGrid':
        """Factory method to create appropriate grid instance."""
        grid_classes = {
            GridType.SPHERICAL: SphericalGrid,
            GridType.HEALPIX: HealpixGrid,
            GridType.RANDOM: RandomGrid
        }
        grid_class = grid_classes.get(grid_type)
        if grid_class is None:
            raise ValueError(f"Unknown grid type: {grid_type}")
        return grid_class.from_header(header)

class SphericalGrid(FootpointGrid):
    def __init__(self, nlon: int, nlat: int, pole_lat: float = 0):
        super().__init__()
        self.grid_type = GridType.SPHERICAL
        if nlon <= 0 or nlat <= 0:
            raise ValueError("Grid dimensions must be positive")
            
        self.nlon = nlon
        self.nlat = nlat
        self.pole_lat = Angle(pole_lat * u.deg)
        self.calculate_grid()

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NLON", self.nlon, "Number of longitude divisions")
        header.set("NLAT", self.nlat, "Number of latitude divisions")
        header.set("POLELAT", self.pole_lat.deg, "Latitude of grid pole, degree")

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

    @classmethod
    def from_header(cls, header: pf.Header) -> 'SphericalGrid':
        """Create SphericalGrid from FITS header."""
        return cls(
            nlon=header["NLON"],
            nlat=header["NLAT"],
            pole_lat=header["POLELAT"]
        )

class HealpixGrid(FootpointGrid):
    def __init__(self, nside: int):
        super().__init__()
        self.grid_type = GridType.HEALPIX
        if nside <= 0 or not (nside & (nside - 1) == 0):
            raise ValueError("NSIDE must be a positive power of 2")
            
        self.nside = nside
        self.calculate_grid()

    def _add_params_to_header(self, header: pf.Header) -> None:
        header.set("NSIDE", self.nside, "Healpix parameter nside")

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

    @classmethod
    def from_header(cls, header: pf.Header) -> 'HealpixGrid':
        """Create HealpixGrid from FITS header."""
        return cls(nside=header["NSIDE"])

class RandomGrid(FootpointGrid):
    def __init__(self, nline_ori: int):
        super().__init__()
        self.grid_type = GridType.RANDOM
        if nline_ori <= 0:
            raise ValueError("Number of random points must be positive")
            
        self.nline_ori = nline_ori
        self.calculate_grid()

    def _add_params_to_header(self, header: pf.Header) -> None:
        pass  # No additional parameters needed for random grid

    def prepare_footpoints(self) -> CartesianRepresentation:
        # Muller method for random points on a sphere
        u = np.random.normal(loc=0, scale=1, size=(3, self.nline_ori))
        norm = (np.sum(u**2, axis=0)) ** 0.5
        (x, y, z) = u / norm
        return CartesianRepresentation(x, y, z)

    def __repr__(self) -> str:
        return f"RandomGrid(nline_ori={self.nline_ori})"

    @classmethod
    def from_header(cls, header: pf.Header) -> 'RandomGrid':
        """Create RandomGrid from FITS header."""
        return cls(nline_ori=header["NLINE"])
