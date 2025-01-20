from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

@dataclass
class FieldLine:
    """Single field line and its properties."""
    # Required property
    footpoint_id: int
    
    # Optional path of the field line in 3D space
    r_line: Optional[NDArray[np.float64]] = None  # Shape: (n_points, 3)
    
    # Optional scalar properties at the initial point
    r_max: Optional[NDArray[np.float64]] = None
    r_cyl_max: Optional[NDArray[np.float64]] = None
    z_abs_max: Optional[NDArray[np.float64]] = None
    b_ini: Optional[NDArray[np.float64]] = None
    b_r_ini_angle: Optional[NDArray[np.float64]] = None
    r_ini: Optional[NDArray[np.float64]] = None

    @classmethod
    def create_template(cls, footpoint_id: int = 0, *, 
                       save_r_line: bool = False,
                       save_r_max: bool = False,
                       save_r_cyl_max: bool = False,
                       save_z_abs_max: bool = False,
                       save_b_ini: bool = False,
                       save_b_r_ini_angle: bool = False,
                       save_r_ini: bool = False) -> 'FieldLine':
        """Create a template FieldLine specifying which properties to calculate."""
        return cls(
            footpoint_id=footpoint_id,
            r_line=np.array([]) if save_r_line else None,
            r_max=np.array([]) if save_r_max else None,
            r_cyl_max=np.array([]) if save_r_cyl_max else None,
            z_abs_max=np.array([]) if save_z_abs_max else None,
            b_ini=np.array([]) if save_b_ini else None,
            b_r_ini_angle=np.array([]) if save_b_r_ini_angle else None,
            r_ini=np.empty((0, 2)) if save_r_ini else None  # Initialize as a 2D numpy array
        )

    def copy_structure(self) -> 'FieldLine':
        """Create a copy of the current FieldLine instance with the same structure."""
        return FieldLine(
            footpoint_id=self.footpoint_id,
            r_line=None,  # Copy structure only, set to None
            r_max=np.array([]) if self.r_max is not None else None,  # Maintain structure
            r_cyl_max=np.array([]) if self.r_cyl_max is not None else None,  # Maintain structure
            z_abs_max=np.array([]) if self.z_abs_max is not None else None,  # Maintain structure
            b_ini=np.array([]) if self.b_ini is not None else None,  # Maintain structure
            b_r_ini_angle=np.array([]) if self.b_r_ini_angle is not None else None,  # Maintain structure
            r_ini=np.empty((0, 2)) if self.r_ini is not None else None  # Maintain structure
        )

@dataclass
class FieldLineSet:
    """Collection of field lines."""
    lines: Sequence[FieldLine]

    @property
    def n_lines(self) -> int:
        return len(self.lines) 