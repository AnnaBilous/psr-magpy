# Standard library imports
from dataclasses import dataclass, fields
import logging
from sys import getsizeof
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class FieldLine:
    """Single field line and its properties.

    A field line represents a magnetic field line in 3D space, along with various
    properties that can be calculated and stored.

    Attributes
    ----------
    footpoint_id : int
        Unique identifier for the field line's footpoint.
    r_line : Optional[NDArray[np.float32]]
        3D coordinates of points along the field line, shape (n_points, 3).
    r_max : Optional[float]
        Maximum radial distance reached by the field line.
    r_cyl_max : Optional[float]
        Maximum cylindrical radius reached by the field line.
    z_abs_max : Optional[float]
        Maximum absolute z-coordinate reached by the field line.
    b_ini : Optional[float]
        Magnetic field magnitude at the line footpoint.
    b_r_ini_angle : Optional[float]
        Angle between magnetic field and radial direction at the line footpoint.
    """

    # Required property
    footpoint_id: int = -1

    # Save flags
    save_r_line: bool = False
    save_r_max: bool = False
    save_r_cyl_max: bool = False
    save_z_abs_max: bool = False
    save_b_ini: bool = False
    save_b_r_ini_angle: bool = False

    # Optional properties
    r_line: Optional[NDArray[np.float32]] = None  # Shape: (n_points, 3)
    r_max: Optional[float] = None
    r_cyl_max: Optional[float] = None
    z_abs_max: Optional[float] = None
    b_ini: Optional[float] = None
    b_r_ini_angle: Optional[float] = None

    def copy_structure(self) -> "FieldLine":
        """Create a copy of the current FieldLine instance with the same structure.

        Returns:
            A new FieldLine instance with the same structure but empty values.
        """
        logger.debug("Creating copy of FieldLine structure for footpoint_id=%d", self.footpoint_id)
        new_line = FieldLine(footpoint_id=self.footpoint_id)
        new_line.save_r_line = self.save_r_line
        new_line.save_r_max = self.save_r_max
        new_line.save_r_cyl_max = self.save_r_cyl_max
        new_line.save_z_abs_max = self.save_z_abs_max
        new_line.save_b_ini = self.save_b_ini
        new_line.save_b_r_ini_angle = self.save_b_r_ini_angle
        return new_line

    def memory_usage(self) -> int:
        """Calculate approximate memory usage of the field line in bytes.

        Returns:
            Estimated memory usage in bytes.
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

        logger.debug("Memory usage for footpoint_id=%d: %d bytes", self.footpoint_id, total_size)
        return total_size

    def __repr__(self) -> str:
        """Return string representation of the field line.

        Returns:
            String representation including footpoint ID and saved properties.
        """
        saved_props = [
            name for name in ['r_line', 'r_max', 'r_cyl_max', 'z_abs_max', 'b_ini', 'b_r_ini_angle']
            if getattr(self, name) is not None
        ]
        return f"FieldLine(id={self.footpoint_id}, saved_properties={saved_props})"


@dataclass
class FieldLineSet:
    """Collection of field lines with I/O capabilities.

    A container class for managing multiple field lines, providing methods for
    concatenation and file I/O operations.

    Attributes
    ----------
    field_lines : List[FieldLine]
        List of FieldLine objects in the set.
    """

    field_lines: List[FieldLine]

    def __init__(self, field_lines: Optional[List[FieldLine]] = None) -> None:
        """Initialize the FieldLineSet.

        Args:
            field_lines: List of FieldLine objects. If None, creates an empty list.
        """
        self.field_lines = field_lines if field_lines is not None else []
        logger.debug("Initialized FieldLineSet with %d field lines", len(self.field_lines))

    def write_to_fits(self, filename: str) -> None:
        """Write field lines to a FITS file.

        Args:
            filename: Path to the output FITS file.
        """
        logger.info("Writing %d field lines to %s", len(self.field_lines), filename)
        from FitsIO import FieldLineFitsIO

        io = FieldLineFitsIO(self.field_lines)
        io.save_to_fits(filename)
        logger.debug("Successfully wrote field lines to %s", filename)

    def concat(self, other: 'FieldLineSet') -> 'FieldLineSet':
        """Concatenate another FieldLineSet to this one.

        Args:
            other: Another FieldLineSet instance to concatenate.

        Returns:
            A new FieldLineSet containing the combined field lines.

        Raises:
            TypeError: If the argument is not a FieldLineSet instance.
        """
        if not isinstance(other, FieldLineSet):
            logger.error("Attempted to concatenate with non-FieldLineSet object: %s", type(other))
            raise TypeError("Argument must be an instance of FieldLineSet")
        
        logger.debug("Concatenating FieldLineSets with %d and %d field lines",
                    len(self.field_lines), len(other.field_lines))
        combined_field_lines = self.field_lines + other.field_lines
        return FieldLineSet(combined_field_lines)

    def __repr__(self) -> str:
        """Return string representation of the field line set.

        Returns:
            String representation including the number of field lines.
        """
        return f"FieldLineSet(n_lines={len(self.field_lines)})"

    def __len__(self) -> int:
        """Return the number of field lines in the set.

        Returns:
            Number of field lines.
        """
        return len(self.field_lines)
