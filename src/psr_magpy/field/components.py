from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation

logger = logging.getLogger(__name__)

class MultipoleType(Enum):
    """Enumeration for supported multipole field types.
    
    Attributes:
        DIPOLE: Dipole field component
        QUADRUPOLE: Quadrupole field component 
    """
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


class Parameters(NamedTuple):
    """Parameters defining a magnetic field component.
    
    Attributes:
        B_magnitude: Overall field strength scaling factor
        moment_orientation_latitude_degree: Latitude of magnetic axis (degrees)
        moment_orientation_longitude_degree: Longitude of magnetic axis (degrees)
        offset_r_ns: Offset of the multipole center [x, y, z] in NS radii
        moment_type: Type of multipole (DIPOLE or QUADRUPOLE)
    """
    B_magnitude: float
    moment_orientation_latitude_degree: float
    moment_orientation_longitude_degree: float
    offset_r_ns: NDArray[np.float32]
    moment_type: MultipoleType


class FieldComponent(ABC):
    """Abstract base class for magnetic field components.
    
    This class provides the framework for implementing various magnetic field components,
    handling coordinate transformations, and managing component parameters.
    
    Attributes:
        params (Parameters): Component parameters including magnitude, orientation, and offset
        moment_orientation (PhysicsSphericalRepresentation): Orientation in spherical coordinates
        moment_offset_vector (NDArray[np.float32]): Offset vector in Cartesian coordinates
        rotate_magn_to_spin_frame (Rotation): Rotation matrix from magnetic to spin frame
        rotate_spin_to_magn_frame (Rotation): Rotation matrix from spin to magnetic frame
    
    Implementation Notes:
        - Subclasses must implement _calculate_field() and get_moment_type()
        - All vector operations are performed using numpy arrays
        - Rotations are performed using scipy.spatial.transform.Rotation
    """

    params: Parameters
    moment_orientation: PhysicsSphericalRepresentation
    moment_offset_vector: NDArray[np.float32]
    rotate_magn_to_spin_frame: Rotation
    rotate_spin_to_magn_frame: Rotation

    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ) -> None:
        """Initialize FieldComponent with given parameters."""
        logger.info(
            "Initializing %s with B=%g, lat=%g°, lon=%g°, offset=%s",
            self.__class__.__name__,
            B_magnitude,
            moment_orientation_latitude_degree,
            moment_orientation_longitude_degree,
            offset_r_ns
        )

        try:
            self.params = Parameters(
                B_magnitude=B_magnitude,
                moment_orientation_latitude_degree=moment_orientation_latitude_degree,
                moment_orientation_longitude_degree=moment_orientation_longitude_degree,
                offset_r_ns=np.asarray(offset_r_ns, dtype=np.float32),
                moment_type=self.get_moment_type()
            )
        except Exception as e:
            logger.error("Failed to initialize parameters: %s", e)
            raise

        logger.debug("Creating moment orientation representation")
        try:
            self.moment_orientation = PhysicsSphericalRepresentation(
                phi=self.params.moment_orientation_longitude_degree * u.deg,
                theta=self.params.moment_orientation_latitude_degree * u.deg,
                r=0,
            )
        except Exception as e:
            logger.error("Failed to create moment orientation: %s", e)
            raise

        self.moment_offset_vector = self.params.offset_r_ns
        
        logger.debug(
            "Moment orientation details - phi: %s, theta: %s, r: %s",
            self.moment_orientation.phi,
            self.moment_orientation.theta,
            self.moment_orientation.r
        )

        try:
            self.rotate_magn_to_spin_frame = Rotation.from_euler(
                "yzx",
                [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
            )
            self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()
            logger.debug(
                "Rotation matrices created - magn_to_spin: %s, spin_to_magn: %s",
                self.rotate_magn_to_spin_frame.as_matrix(),
                self.rotate_spin_to_magn_frame.as_matrix()
            )
        except Exception as e:
            logger.error("Failed to create rotation matrices: %s", e)
            raise

        logger.info("Successfully initialized %s", self.__class__.__name__)

    @abstractmethod
    def _calculate_field(self, r_magn_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate magnetic field in the magnetic frame.
        
        This abstract method must be implemented by concrete subclasses to define
        the specific magnetic field configuration.
        
        Args:
            r_magn_frame: Position vector in the magnetic frame (x, y, z coordinates)
                         in units of neutron star radii
        
        Returns:
            NDArray[np.float32]: The magnetic field vector (Bx, By, Bz) in the magnetic frame

        """
        pass

    @abstractmethod
    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        pass

    def _transform_to_magnetic_frame(self, r_spin_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform coordinates from spin frame to magnetic frame.
        
        Performs the coordinate transformation in two steps:
        1. Applies the offset correction to center on the multipole
        2. Rotates the coordinates into the magnetic frame
        
        Args:
            r_spin_frame: Position vector [x, y, z] in the spin frame
        
        Returns:
            NDArray[np.float32]: Position vector in the magnetic frame

        """
        logger.debug("Transforming position from spin frame: %s", r_spin_frame)
        
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        logger.debug("Position after offset correction: %s", r_offset_spin_frame)
        
        r_magn_frame = self.rotate_spin_to_magn_frame.apply(r_offset_spin_frame)
        logger.debug("Position in magnetic frame: %s", r_magn_frame)
        
        return r_magn_frame

    def get_field(self, r_spin_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate the magnetic field at a given position in the spin frame.
        
        This is the main interface method for field calculations. It handles:
        - Input validation
        - Coordinate transformations
        - Field calculation in magnetic frame
        - Error handling and logging
        
        Args:
            r_spin_frame: Position vector [x, y, z] in the spin frame
                         (in units of neutron star radii)
        
        Returns:
            NDArray[np.float32]: Magnetic field vector [Bx, By, Bz] in the spin frame
        
        Raises:
            ValueError: If input contains non-finite values or calculation fails
            Exception: For other calculation errors

        """
        logger.debug(
            "Calculating %s field at position %s", 
            self.params.moment_type.value, 
            r_spin_frame
        )
        
        # Input validation
        if not np.isfinite(r_spin_frame).all():
            err_msg = f"Invalid position vector containing non-finite values: {r_spin_frame}"
            logger.error(err_msg)
            raise ValueError(err_msg)
            
        try:
            r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
            field = self._calculate_field(r_magn_frame)
            
            # Validate output
            if not np.isfinite(field).all():
                logger.error("Calculated field contains non-finite values: %s", field)
                raise ValueError(f"Invalid field calculation result: {field}")
                
            logger.debug("Field in magnetic frame: %s", field)
            logger.debug("Successfully calculated field at position %s", r_spin_frame)
            return field
            
        except Exception as e:
            logger.error("Error calculating field: %s", e)
            raise

    def __repr__(self) -> str:
        """Return string representation of the field component.

        Returns:
            str: A string representation showing key parameters of the component.
        """
        return (
            f"{self.__class__.__name__}("
            f"type={self.params.moment_type.value}, "
            f"B={self.params.B_magnitude:.3g}, "
            f"lat={self.params.moment_orientation_latitude_degree:.2f}°, "
            f"lon={self.params.moment_orientation_longitude_degree:.2f}°, "
            f"offset={list(self.params.offset_r_ns)})"
        )


class DipoleFieldComponent(FieldComponent):
    """Dipole magnetic field component implementation.
    
    This class implements a magnetic dipole field with arbitrary orientation and offset.
    The field follows the standard dipole form.

    
    Example:
        ```python
        # Create centered dipole aligned with z-axis
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        
        # Create offset tilted dipole
        dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=45.0,
            offset_r_ns=np.array([0.0, 0.0, 0.1])
        )
        ```
    """
    
    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ) -> None:
        """Initialize DipoleFieldComponent with given parameters."""
        super().__init__(
            B_magnitude=B_magnitude,
            moment_orientation_latitude_degree=moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=moment_orientation_longitude_degree,
            offset_r_ns=offset_r_ns,
        )

    def _calculate_field(self, r_magn_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate dipole magnetic field in the magnetic frame.
                
        Args:
            r_magn_frame: Position vector in magnetic frame (NS radii)
        
        Returns:
            NDArray[np.float32]: Magnetic field vector in magnetic frame
        
        """
        logger.debug("Calculating dipole field at position: %s", r_magn_frame)
        
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        r_cyl = (x**2 + y**2) ** 0.5
        
        logger.debug("Spherical radius: %g, cylindrical radius: %g", r, r_cyl)
        
        if r < 1e-10:  # Avoid singularity at origin
            logger.warning("Position too close to origin (r=%g), field undefined", r)
            return np.zeros_like(r_magn_frame)

        try:
            B_magnetic_frame = (
                self.params.B_magnitude
                * r**-5
                * np.array(
                    [
                        1.5 * z * x,
                        1.5 * z * y,
                        z**2 - 0.5 * r_cyl**2,
                    ]
                )
            )
            
            field = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)
            logger.debug("Calculated dipole field: %s", field)
            return field
            
        except Exception as e:
            logger.error("Error in dipole field calculation: %s", e)
            raise

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.DIPOLE


class QuadrupoleFieldComponent(FieldComponent):
    """Quadrupole magnetic field component implementation.
    
    This class implements a magnetic quadrupole field with arbitrary orientation and offset.
   
    """
    
    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ) -> None:
        """Initialize QuadrupoleFieldComponent with given parameters."""
        super().__init__(
            B_magnitude=B_magnitude,
            moment_orientation_latitude_degree=moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=moment_orientation_longitude_degree,
            offset_r_ns=offset_r_ns,
        )

    def _calculate_field(self, r_magn_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate quadrupole magnetic field in the magnetic frame.
        
        Implements a simplified quadrupole field configuration with:
        - 1/r⁴ radial falloff
        - Axisymmetric angular dependence
        - Continuous field everywhere except origin
        
        Args:
            r_magn_frame: Position vector in magnetic frame (NS radii)
        
        Returns:
            NDArray[np.float32]: Magnetic field vector in magnetic frame
        
        Notes:
            - More complex angular structure than dipole
            - Useful for modeling field deviations
            - Returns zero field at origin
        """
        logger.debug("Calculating quadrupole field at position: %s", r_magn_frame)
        
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        
        if r < 1e-10:  # Avoid singularity at origin
            logger.warning("Position too close to origin (r=%g), field undefined", r)
            return np.zeros_like(r_magn_frame)
            
        try:
            denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
            logger.debug("Denominator value: %g", denom)
            
            B_magnetic_frame = (
                self.params.B_magnitude
                * r**-4
                * np.array(
                    [
                        x / denom,
                        y / denom,
                        z / denom - z / r,
                    ]
                )
            )
            
            field = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)
            logger.debug("Calculated quadrupole field: %s", field)
            return field
            
        except Exception as e:
            logger.error("Error in quadrupole field calculation: %s", e)
            raise

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.QUADRUPOLE

