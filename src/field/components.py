from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, NamedTuple, Any, Protocol, runtime_checkable
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation

# Initialize logger
logger = logging.getLogger(__name__)


class MultipoleType(Enum):
    """Enumeration for multipole types."""
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


@runtime_checkable
class FieldComponentProtocol(Protocol):
    """Protocol defining the interface for field components."""
    def get_field(self, r_spin_frame: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def get_moment_type(self) -> MultipoleType: ...


class Parameters(NamedTuple):
    """Parameters for field components."""
    B_magnitude: float
    moment_orientation_latitude_degree: float
    moment_orientation_longitude_degree: float
    offset_r_ns: NDArray[np.float32]
    moment_type: MultipoleType


class FieldComponent(ABC):
    """Abstract base class for field components."""

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
        """Initialize FieldComponent with given parameters.

        Args:
            B_magnitude: Magnitude of the magnetic field.
            moment_orientation_latitude_degree: Latitude of the moment orientation in degrees.
            moment_orientation_longitude_degree: Longitude of the moment orientation in degrees.
            offset_r_ns: Offset vector [x, y, z] in neutron star radii.
        """
        logger.info("Initializing %s", self.__class__.__name__)

        self.params = Parameters(
            B_magnitude=B_magnitude,
            moment_orientation_latitude_degree=moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=moment_orientation_longitude_degree,
            offset_r_ns=np.asarray(offset_r_ns, dtype=np.float32),
            moment_type=self.get_moment_type()
        )

        self.moment_orientation = PhysicsSphericalRepresentation(
            phi=self.params.moment_orientation_longitude_degree * u.deg,
            theta=self.params.moment_orientation_latitude_degree * u.deg,
            r=0,
        )
        self.moment_offset_vector = self.params.offset_r_ns

        logger.debug(
            "Moment orientation: phi=%s, theta=%s",
            self.moment_orientation.phi,
            self.moment_orientation.theta
        )
        logger.debug("Moment offset vector: %s", self.moment_offset_vector)

        self.rotate_magn_to_spin_frame = Rotation.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )
        logger.debug("rotate_magn_to_spin_frame: %s", self.rotate_magn_to_spin_frame)
        self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()

    @abstractmethod
    def _calculate_field(self, r_magn_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate magnetic field in the magnetic frame.

        Args:
            r_magn_frame: Position vector in the magnetic frame.

        Returns:
            np.ndarray: The magnetic field vector in the magnetic frame.
        """
        pass

    @abstractmethod
    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        pass

    def _transform_to_magnetic_frame(self, r_spin_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform coordinates from spin frame to magnetic frame.

        Args:
            r_spin_frame: Position vector in the spin frame.

        Returns:
            np.ndarray: Position vector in the magnetic frame.
        """
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        r_magn_frame = self.rotate_spin_to_magn_frame.apply(r_offset_spin_frame)
        return r_magn_frame

    def get_field(self, r_spin_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate the magnetic field at a given position in the spin frame.

        Args:
            r_spin_frame: Position vector in the spin frame.

        Returns:
            np.ndarray: The magnetic field vector in the spin frame.
        """
        logger.debug("Calculating %s field at position %s", 
                    self.params.moment_type.value, r_spin_frame)
        r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
        field = self._calculate_field(r_magn_frame)
        logger.debug("Calculated %s field: %s", 
                    self.params.moment_type.value, field)
        return field

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
    """Concrete implementation of a dipole field component."""
    
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
            r_magn_frame: Position vector in the magnetic frame.

        Returns:
            np.ndarray: The magnetic field vector in the magnetic frame.
        """
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        r_cyl = (x**2 + y**2) ** 0.5

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
        return self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.DIPOLE


class QuadrupoleFieldComponent(FieldComponent):
    """Concrete implementation of a quadrupole field component."""
    
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

        Args:
            r_magn_frame: Position vector in the magnetic frame.

        Returns:
            np.ndarray: The magnetic field vector in the magnetic frame.
        """
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
        
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
        return self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.QUADRUPOLE


__all__ = ['DipoleFieldComponent', 'QuadrupoleFieldComponent']