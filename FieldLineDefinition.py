from dataclasses import dataclass, fields
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray
from sys import getsizeof


@dataclass
class FieldLine:
    """Single field line and its properties."""

    # Required property
    footpoint_id: int

    # Optional path of the field line in 3D space
    r_line: Optional[NDArray[np.float64]] = None  # Shape: (n_points, 3)

    # Optional scalar properties at the initial point
    r_max: Optional[float] = None
    r_cyl_max: Optional[float] = None
    z_abs_max: Optional[float] = None
    b_ini: Optional[float] = None
    b_r_ini_angle: Optional[float] = None

    def __init__(
        self,
        footpoint_id: int = -1,
        *,
        save_r_line: bool = False,
        save_r_max: bool = False,
        save_r_cyl_max: bool = False,
        save_z_abs_max: bool = False,
        save_b_ini: bool = False,
        save_b_r_ini_angle: bool = False
    ):
        self.footpoint_id = footpoint_id
        self.r_line = np.array([]) if save_r_line else None
        self.r_max = 0.0 if save_r_max else None
        self.r_cyl_max = 0.0 if save_r_cyl_max else None
        self.z_abs_max = 0.0 if save_z_abs_max else None
        self.b_ini = 0.0 if save_b_ini else None
        self.b_r_ini_angle = 0.0 if save_b_r_ini_angle else None

    # @property
    # def parameters(self):
    #    return [field.name for field in self.__dataclass_fields__.values() if getattr(self, field.name) is not None]

    def copy_structure(self) -> "FieldLine":
        """Create a copy of the current FieldLine instance with the same structure."""
        return FieldLine(
            footpoint_id=self.footpoint_id,
            save_r_line=False,  # Assuming you want to keep it None
            save_r_max=self.r_max is not None,
            save_r_cyl_max=self.r_cyl_max is not None,
            save_z_abs_max=self.z_abs_max is not None,
            save_b_ini=self.b_ini is not None,
            save_b_r_ini_angle=self.b_r_ini_angle is not None,
        )

    def memory_usage(self) -> int:
        """Calculate approximate memory usage of the field line in bytes.

        Returns
        -------
        int
            Estimated memory usage in bytes
        """
        total_size = 0

        # Get all instance attributes
        for field in fields(self):
            value = getattr(self, field.name)

            if value is None:
                continue

            if isinstance(value, np.ndarray):
                # For numpy arrays, get actual memory usage
                total_size += value.nbytes
            elif isinstance(value, (float, int, bool)):
                # For primitive types, use sys.getsizeof
                total_size += getsizeof(value)
            elif isinstance(value, str):
                # For strings, account for actual string size
                total_size += getsizeof(value)
            else:
                # For other types, use basic size
                total_size += getsizeof(value)

        return total_size


@dataclass
class FieldLineSet:
    """Collection of field lines with IO capabilities."""

    field_lines: List[FieldLine]

    def __init__(self, field_lines: Optional[List[FieldLine]] = None):
        """Initialize the FieldLineSet.

        Args:
            field_lines: List of FieldLine objects. If None, creates an empty list.
        """
        self.field_lines = field_lines if field_lines is not None else []

    def write_to_fits(self, filename: str) -> None:
        """Write field lines to a FITS file.

        Args:
            filename: Path to the output FITS file
        """
        from FitsIO import FieldLineFitsIO

        io = FieldLineFitsIO(self.field_lines)
        io.save_to_fits(filename)
