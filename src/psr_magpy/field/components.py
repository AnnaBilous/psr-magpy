import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Union, List, cast
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from pydantic import BaseModel, Field, model_validator, field_validator

from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyFieldError

logger = get_logger(__name__, console_format_brief=True)


class MultipoleType(Enum):
    """Enumeration for supported multipole field types."""

    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


class FieldComponentParams(BaseModel):
    """Parameters for magnetic field components."""

    B_magnitude: float = Field(
        default=1.0, ge=0.0, description="Field strength scaling factor"
    )
    moment_orientation_latitude_degree: float = Field(
        default=0.0,
        ge=0.0,
        le=180.0,
        description="Colatitude of magnetic axis (degrees)",
    )
    moment_orientation_longitude_degree: float = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Longitude of magnetic axis (degrees)",
    )
    offset_r_ns: Union[List[float], NDArray[np.float64]] = Field(
        default=[0.0, 0.0, 0.0], description="Offset vector [x, y, z] in NS radii"
    )
    moment_type: MultipoleType = Field(description="Type of multipole")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("offset_r_ns", mode="before")
    @classmethod
    def validate_offset_vector(cls, v):
        """Validate offset vector has exactly 3 components."""
        # Handle non-list/array types first to provide clear error message
        if not isinstance(v, (list, np.ndarray, str)):
            # Check if it's a type that could be converted
            try:
                v_array = np.asarray(v, dtype=np.float64)
                if v_array.shape != (3,):
                    raise ValueError(
                        f"Offset vector must have exactly 3 components, got shape {v_array.shape}"
                    )
                return v_array
            except (TypeError, ValueError):
                raise ValueError(f"Offset must be list or array, got {type(v)}")
        
        if isinstance(v, str):
            raise ValueError(f"Offset must be list or array, got {type(v)}")
            
        if isinstance(v, (list, np.ndarray)):
            v_array = np.asarray(v, dtype=np.float64)
            if v_array.shape != (3,):
                raise ValueError(
                    f"Offset vector must have exactly 3 components, got shape {v_array.shape}"
                )
            return v_array
        raise ValueError(f"Offset must be list or array, got {type(v)}")


class FieldComponent(ABC):
    """Abstract base class for magnetic field components.

    Provides common functionality for all field component types including
    coordinate transformations and parameter management.
    """

    params: FieldComponentParams

    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: Union[List[float], NDArray[np.float64]] = [0.0, 0.0, 0.0],
    ) -> None:
        """Initialize FieldComponent with given parameters."""
        logger.info(
            f"Initializing {self.__class__.__name__} with B={B_magnitude:.3g}, "
            f"lat={moment_orientation_latitude_degree:.2f}°, "
            f"lon={moment_orientation_longitude_degree:.2f}°, "
            f"offset={offset_r_ns}"
        )

        # Create and validate parameters using Pydantic
        try:
            self.params = FieldComponentParams(
                B_magnitude=B_magnitude,
                moment_orientation_latitude_degree=moment_orientation_latitude_degree,
                moment_orientation_longitude_degree=moment_orientation_longitude_degree,
                offset_r_ns=offset_r_ns,
                moment_type=self.get_moment_type(),
            )
        except Exception as e:
            raise PSRMagPyFieldError(f"Failed to initialize parameters: {e}") from e

        logger.debug(f"Successfully initialized {self.__class__.__name__}")

    @cached_property
    def moment_orientation(self) -> PhysicsSphericalRepresentation:
        """Get moment orientation in spherical coordinates (cached)."""
        return PhysicsSphericalRepresentation(
            phi=u.Quantity(self.params.moment_orientation_longitude_degree, u.deg),
            theta=u.Quantity(self.params.moment_orientation_latitude_degree, u.deg),
            r=0,
        )

    @cached_property
    def moment_offset_vector(self) -> NDArray[np.float64]:
        """Get moment offset vector (cached)."""
        return np.asarray(self.params.offset_r_ns, dtype=np.float64)

    @cached_property
    def numba_compatible_offset_vector(self) -> NDArray[np.float64]:
        """Get offset vector as numpy array (3-element) for Numba compatibility.
        
        Returns the same as moment_offset_vector property. Provided for consistency
        with Numba-compatible API.
        
        Returns
        -------
        NDArray[np.float64]
            3-element offset vector (float64).
        """
        return self.moment_offset_vector

    @cached_property
    def rotate_magn_to_spin_frame(self) -> Rotation:
        """Get rotation matrix from magnetic to spin frame (cached)."""
        return Rotation.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )

    @cached_property
    def rotate_spin_to_magn_frame(self) -> Rotation:
        """Get rotation matrix from spin to magnetic frame (cached)."""
        return self.rotate_magn_to_spin_frame.inv()

    @cached_property
    def numba_compatible_rotation_matrix(self) -> NDArray[np.float64]:
        """Get rotation matrix from magnetic to spin frame as numpy array (3x3).
        
        Returns the rotation matrix as a 3x3 numpy array (float64) for Numba compatibility.
        This is equivalent to rotate_magn_to_spin_frame but returns a matrix instead of
        a scipy Rotation object.
        
        Returns
        -------
        NDArray[np.float64]
            3x3 rotation matrix (float64) for transforming vectors from magnetic to spin frame.
        """
        return cast(NDArray[np.float64], self.rotate_magn_to_spin_frame.as_matrix().astype(np.float64))

    @cached_property
    def numba_compatible_rotation_matrix_inv(self) -> NDArray[np.float64]:
        """Get inverse rotation matrix from spin to magnetic frame as numpy array (3x3).
        
        Returns the inverse rotation matrix as a 3x3 numpy array (float64) for Numba compatibility.
        This is equivalent to rotate_spin_to_magn_frame but returns a matrix instead of
        a scipy Rotation object.
        
        Returns
        -------
        NDArray[np.float64]
            3x3 rotation matrix (float64) for transforming vectors from spin to magnetic frame.
        """
        return cast(NDArray[np.float64], self.rotate_spin_to_magn_frame.as_matrix().astype(np.float64))

    @abstractmethod
    def _calculate_field(
        self, r_magn_frame: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate magnetic field in the magnetic frame."""
        pass

    @abstractmethod
    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        pass

    def get_numba_params(self) -> dict:
        """Get Numba-compatible parameters as a dictionary.
        
        Returns a dictionary containing all parameters needed for Numba-compiled
        field calculations:
        - B_magnitude: float (field strength scaling factor)
        - rotation_matrix: NDArray[float64] (3x3, magn to spin frame)
        - rotation_matrix_inv: NDArray[float64] (3x3, spin to magn frame)
        - offset_vector: NDArray[float64] (3-element, offset in spin frame)
        - moment_type: str (enum value: "dipole" or "quadrupole")
        
        Returns
        -------
        dict
            Dictionary with keys: 'B_magnitude', 'rotation_matrix', 'rotation_matrix_inv',
            'offset_vector', 'moment_type'
        """
        return {
            "B_magnitude": float(self.params.B_magnitude),
            "rotation_matrix": self.numba_compatible_rotation_matrix,
            "rotation_matrix_inv": self.numba_compatible_rotation_matrix_inv,
            "offset_vector": self.numba_compatible_offset_vector,
            "moment_type": self.params.moment_type.value,
        }

    def _transform_to_magnetic_frame(
        self, r_spin_frame: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform coordinates from spin frame to magnetic frame."""
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        r_magn_frame = self.rotate_spin_to_magn_frame.apply(r_offset_spin_frame)
        return cast(NDArray[np.float64], r_magn_frame)

    def get_field(self, r_spin_frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the magnetic field at a given position in the spin frame."""
        # Input validation
        if not np.isfinite(r_spin_frame).all():
            raise PSRMagPyFieldError(
                f"Invalid position vector containing non-finite values: {r_spin_frame}"
            )

        try:
            r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
            field = self._calculate_field(r_magn_frame)

            # Validate output
            if not np.isfinite(field).all():
                raise PSRMagPyFieldError(f"Calculated field contains non-finite values: {field}")

            return field

        except Exception as e:
            raise PSRMagPyFieldError(f"Error calculating field: {e}") from e

    def __repr__(self) -> str:
        """Return string representation of the field component."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.params.moment_type}, "
            f"B={self.params.B_magnitude:.3g}, "
            f"lat={self.params.moment_orientation_latitude_degree:.2f}°, "
            f"lon={self.params.moment_orientation_longitude_degree:.2f}°, "
            f"offset={list(self.params.offset_r_ns)})"
        )


class DipoleFieldComponent(FieldComponent):
    """Dipole magnetic field component implementation."""

    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: Union[List[float], NDArray[np.float64]] = [0.0, 0.0, 0.0],
    ) -> None:
        """Initialize DipoleFieldComponent with given parameters."""
        super().__init__(
            B_magnitude=B_magnitude,
            moment_orientation_latitude_degree=moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=moment_orientation_longitude_degree,
            offset_r_ns=offset_r_ns,
        )

    def _calculate_field(
        self, r_magn_frame: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate dipole magnetic field in the magnetic frame."""
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5
        r_cyl = (x**2 + y**2) ** 0.5

        if r < 1e-10:  # Avoid singularity at origin
            logger.warning(f"Position too close to origin (r={r:.2e}), field undefined")
            return np.zeros_like(r_magn_frame, dtype=np.float64)

        try:
            B_magnetic_frame = (
                self.params.B_magnitude
                * r**-5
                * np.array(
                    [
                        1.5 * z * x,
                        1.5 * z * y,
                        z**2 - 0.5 * r_cyl**2,
                    ],
                    dtype=np.float64,
                )
            )

            field = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)
            return cast(NDArray[np.float64], field)

        except Exception as e:
            raise PSRMagPyFieldError(f"Error in dipole field calculation: {e}") from e

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.DIPOLE


class QuadrupoleFieldComponent(FieldComponent):
    """Quadrupole magnetic field component implementation."""

    def __init__(
        self,
        B_magnitude: float = 1.0,
        moment_orientation_latitude_degree: float = 0.0,
        moment_orientation_longitude_degree: float = 0.0,
        offset_r_ns: Union[List[float], NDArray[np.float64]] = [0.0, 0.0, 0.0],
    ) -> None:
        """Initialize QuadrupoleFieldComponent with given parameters."""
        super().__init__(
            B_magnitude=B_magnitude,
            moment_orientation_latitude_degree=moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=moment_orientation_longitude_degree,
            offset_r_ns=offset_r_ns,
        )

    def _calculate_field(
        self, r_magn_frame: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate quadrupole magnetic field in the magnetic frame."""
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2) ** 0.5

        if r < 1e-10:  # Avoid singularity at origin
            logger.warning(f"Position too close to origin (r={r:.2e}), field undefined")
            return np.zeros_like(r_magn_frame, dtype=np.float64)

        try:
            denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1

            B_magnetic_frame = (
                self.params.B_magnitude
                * r**-4
                * np.array(
                    [
                        x / denom,
                        y / denom,
                        z / denom - z / r,
                    ],
                    dtype=np.float64,
                )
            )

            field = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)
            return cast(NDArray[np.float64], field)

        except Exception as e:
            raise PSRMagPyFieldError(f"Error in quadrupole field calculation: {e}") from e

    def get_moment_type(self) -> MultipoleType:
        """Get the moment type for this component."""
        return MultipoleType.QUADRUPOLE
