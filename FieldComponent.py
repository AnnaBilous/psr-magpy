from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, Union, Optional, Any
import logging

import numpy as np
from scipy.spatial.transform import Rotation as rot

from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation

# Initialize logger
logger = logging.getLogger(__name__)

class MultipoleType(Enum):
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    
    
@dataclass
class FieldComponentParams:
    moment_type: Union[MultipoleType, str]
    B_magnitude: float = 1.0
    moment_orientation_latitude_degree: float = 0.0
    moment_orientation_longitude_degree: float = 0.0
    offset_x_r_ns: float = 0.0
    offset_y_r_ns: float = 0.0
    offset_z_r_ns: float = 0.0
    component_number: int = 0

    def __post_init__(self):
        if isinstance(self.moment_type, str):
            # Try to convert string to MultipoleType if provided as string
            try:
                self.moment_type = MultipoleType(self.moment_type)
            except ValueError:
                raise ValueError(f"moment_type must be a MultipoleType, got {self.moment_type}")
    

class FieldComponent:
    def __init__(self, **kwargs):
        logger.info("Initializing FieldComponent")
        
        # Convert kwargs to FieldComponentParams
        try:
            params = FieldComponentParams(**kwargs)
        except TypeError as e:
            logger.error(f"Invalid parameters: {e}")
            raise
            
        logger.debug(f"Component parameters: {params}")
        
        self.params = params
        self.moment_orientation = PhysicsSphericalRepresentation(
            phi=params.moment_orientation_longitude_degree * u.deg,
            theta=params.moment_orientation_latitude_degree * u.deg,
            r=0,
        )
        self.moment_offset_vector = np.array([
            params.offset_x_r_ns,
            params.offset_y_r_ns,
            params.offset_z_r_ns,
        ])

        self.rotate_magn_to_spin_frame = rot.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )
        self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()

    def get_field(self, r_spin_frame: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field at a given position in the spin frame.
        
        Args:
            r_spin_frame: Position vector in the spin frame
            
        Returns:
            The magnetic field vector in the spin frame
        """
        # Transform coordinates to magnetic frame
        r_magn_frame = self._transform_to_magnetic_frame(r_spin_frame)
        
        # Calculate field in magnetic frame
        if self.params.moment_type == MultipoleType.DIPOLE:
            B_magnetic_frame = self._calculate_dipole_field(r_magn_frame)
        elif self.params.moment_type == MultipoleType.QUADRUPOLE:
            B_magnetic_frame = self._calculate_quadrupole_field(r_magn_frame)
        
        # Transform back to spin frame
        return self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)

    def _transform_to_magnetic_frame(self, r_spin_frame: np.ndarray) -> np.ndarray:
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        
        # Rotate to magnetic frame
        r_magn_frame = self.rotate_spin_to_magn_frame.apply(r_offset_spin_frame)
         
        return r_magn_frame

    def _calculate_dipole_field(self, r_magn_frame: np.ndarray) -> np.ndarray:
        """Calculate dipole magnetic field in the magnetic frame."""
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2)**0.5
        r_cyl = (x**2 + y**2)**0.5
        
        return (
            self.params.B_magnitude
            * r**-5
            * np.array([
                1.5 * z * x,
                1.5 * z * y,
                z**2 - 0.5 * r_cyl**2,
            ])
        )

    def _calculate_quadrupole_field(self, r_magn_frame: np.ndarray) -> np.ndarray:
        """Calculate quadrupole magnetic field in the magnetic frame."""
        x, y, z = r_magn_frame
        r = (x**2 + y**2 + z**2)**0.5
        denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
        return (
            self.params.B_magnitude
            * r**-4
            * np.array([
                x / denom,
                y / denom,
                z / denom - z / r,
            ])
        )
