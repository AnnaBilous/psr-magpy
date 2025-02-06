from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, Union, Optional, Any, List
import logging

import numpy as np
from scipy.spatial.transform import Rotation as rot

from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation

# Initialize logger
logger = logging.getLogger(__name__)


class MultipoleType(Enum):
    """Enumeration for multipole types."""
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


@dataclass
class FieldComponentParams:
    """Parameters for a field component.

    Attributes
    ----------
    moment_type : Union[MultipoleType, str]
        Type of the multipole moment (dipole or quadrupole).
    B_magnitude : float
        Magnitude of the magnetic field.
    moment_orientation_latitude_degree : float
        Latitude of the moment orientation in degrees.
    moment_orientation_longitude_degree : float
        Longitude of the moment orientation in degrees.
    offset_x_r_ns : float
        X offset in radius normalized units.
    offset_y_r_ns : float
        Y offset in radius normalized units.
    offset_z_r_ns : float
        Z offset in radius normalized units.
    component_number : int
        Identifier for the component.
    """

    moment_type: Union[MultipoleType, str]
    B_magnitude: float = 1.0
    moment_orientation_latitude_degree: float = 0.0
    moment_orientation_longitude_degree: float = 0.0
    offset_x_r_ns: float = 0.0
    offset_y_r_ns: float = 0.0
    offset_z_r_ns: float = 0.0
    component_number: int = 0

    @classmethod
    def get_parameter_names(cls) -> List[str]:
        """Get the names of the parameters defined in the dataclass.

        Returns
        -------
        List[str]
            A list of parameter names.
        """
        return list(cls.__dataclass_fields__.keys())

    def __post_init__(self) -> None:
        """Post-initialization processing to convert string to MultipoleType if necessary."""
        if isinstance(self.moment_type, str):
            try:
                self.moment_type = MultipoleType(self.moment_type)
            except ValueError:
                raise ValueError(
                    f"moment_type must be a MultipoleType, got {self.moment_type}"
                )


class FieldComponent(ABC):
    """Abstract base class for field components.

    Attributes
    ----------
    params : FieldComponentParams
        Parameters for the field component.
    moment_orientation : PhysicsSphericalRepresentation
        Orientation of the moment in spherical coordinates.
    moment_offset_vector : np.ndarray
        Offset vector for the moment in Cartesian coordinates.
    rotate_magn_to_spin_frame : Rotation
        Rotation object to transform from magnetic to spin frame.
    rotate_spin_to_magn_frame : Rotation
        Rotation object to transform from spin to magnetic frame.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize FieldComponent with given parameters.

        Args:
            **kwargs: Keyword arguments for FieldComponentParams.

        Raises:
            TypeError: If invalid parameters are provided.
        """
        logger.info("Initializing %s", self.__class__.__name__)

        # Convert kwargs to FieldComponentParams
        try:
            params = FieldComponentParams(**kwargs)
        except TypeError as e:
            logger.error("Failed to initialize %s: %s", self.__class__.__name__, e)
            raise

        logger.debug("Component parameters: %s", params)

        self.params = params
        self.moment_orientation = PhysicsSphericalRepresentation(
            phi=params.moment_orientation_longitude_degree * u.deg,
            theta=params.moment_orientation_latitude_degree * u.deg,
            r=0,
        )
        self.moment_offset_vector = np.array(
            [
                params.offset_x_r_ns,
                params.offset_y_r_ns,
                params.offset_z_r_ns,
            ]
        )

        logger.debug(
            "Moment orientation: phi=%s, theta=%s",
            self.moment_orientation.phi,
            self.moment_orientation.theta
        )
        logger.debug("Moment offset vector: %s", self.moment_offset_vector)

        self.rotate_magn_to_spin_frame = rot.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )
        self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()

    def __repr__(self) -> str:
        """Return string representation of the field component.

        Returns:
            str: String representation of the component.
        """
        return (
            f"{self.__class__.__name__}("
            f"B_magnitude={self.params.B_magnitude:.2f}, "
            f"lat={self.params.moment_orientation_latitude_degree:.1f}°, "
            f"lon={self.params.moment_orientation_longitude_degree:.1f}°, "
            f"offset=[{self.params.offset_x_r_ns:.2f}, "
            f"{self.params.offset_y_r_ns:.2f}, "
            f"{self.params.offset_z_r_ns:.2f}])"
        )

    @abstractmethod
    def get_field(self, r_spin_frame: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field at a given position in the spin frame.

        Args:
            r_spin_frame: Position vector in the spin frame.

        Returns:
            np.ndarray: The magnetic field vector in the spin frame.
        """
        pass

    def _transform_to_magnetic_frame(self, r_spin_frame: np.ndarray) -> np.ndarray:
        """Transform coordinates from spin frame to magnetic frame.

        Args:
            r_spin_frame: Position vector in the spin frame.

        Returns:
            np.ndarray: Position vector in the magnetic frame.
        """
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        r_magn_frame = self.rotate_spin_to_magn_frame.apply(r_offset_spin_frame)
        return r_magn_frame


class DipoleFieldComponent(FieldComponent):
    """Concrete implementation of a dipole field component."""

    def get_field(self, r_spin_frame: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field for a dipole at a given position."""
        logger.debug("Calculating dipole field at position %s", r_spin_frame)
        r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
        field = self._calculate_dipole_field(r_magn_frame)
        logger.debug("Calculated dipole field: %s", field)
        return field

    def _calculate_dipole_field(self, r_magn_frame: np.ndarray) -> np.ndarray:
        """Calculate dipole magnetic field in the magnetic frame.

        Args:
            r_magn_frame: Position vector in the magnetic frame.

        Returns:
            np.ndarray: The magnetic field vector in the magnetic frame.
        """
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        r_cyl = (x**2 + y**2) ** 0.5

        return (
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


class QuadrupoleFieldComponent(FieldComponent):
    """Concrete implementation of a quadrupole field component."""

    def get_field(self, r_spin_frame: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field for a quadrupole at a given position."""
        logger.debug("Calculating quadrupole field at position %s", r_spin_frame)
        r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
        field = self._calculate_quadrupole_field(r_magn_frame)
        logger.debug("Calculated quadrupole field: %s", field)
        return field

    def _calculate_quadrupole_field(self, r_magn_frame: np.ndarray) -> np.ndarray:
        """Calculate quadrupole magnetic field in the magnetic frame.

        Args:
            r_magn_frame: Position vector in the magnetic frame.

        Returns:
            np.ndarray: The magnetic field vector in the magnetic frame.
        """
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
        return (
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
