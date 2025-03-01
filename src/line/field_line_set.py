from typing import Any, List, Union
from sys import getsizeof
import numpy as np
from .field_line_base import FieldLineBase, PropertyName, PROPERTY_TYPES
from .field_line import FieldLine

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldLineSet(FieldLineBase):
    """Collection of field lines with array values"""
    def __init__(self, field_lines: Union[List[FieldLine], FieldLine]) -> None:
        """Initialize FieldLineSet, either empty or with provided field lines.
        
        Args:
            field_lines: Input that defines the structure:
                        - List[FieldLine]: Creates set with these field lines
                        - FieldLine: Uses as template for empty set
        """
        if isinstance(field_lines, FieldLine):
            # Initialize using template's structure but no data
            super().__init__(**field_lines.save_flags.__dict__)
            self._properties = {}
            self.n_field_lines = 0
            logger.debug("Initialized empty FieldLineSet with template: %s", repr(field_lines.save_flags))
        else:
            # Initialize using list of field lines
            template = field_lines[0]
            super().__init__(**template.save_flags.__dict__)
            logger.debug("Initialized FieldLineSet with field lines, template: %s", repr(template.save_flags))
            self.append(field_lines)
    
    def _set_property(self, prop: PropertyName, value: Any) -> None:
        """Set a property value, ensuring it's a collection."""
        self._validate_type(prop, value)
        if isinstance(value, (int, float)):
            raise TypeError(f"Expected collection for {prop.key_name}, got single value")
        self._properties[prop] = value 
        
           
    def append(self, field_lines: List[FieldLine]) -> None:
        """Append field lines to the set."""
        # Validate save flags compatibility
        for line in field_lines:
            if line.save_flags.__dict__ != self.save_flags.__dict__:
                raise ValueError("All field lines must have the same save flags as the set")
        
        # Initialize properties dict if not exists
        if not hasattr(self, '_properties'):
            self._properties = {}
        
        # Process each enabled property
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                continue
            
            # Get values from field lines
            values = [line._get_property(prop) for line in field_lines]
            
            if prop == PropertyName.R_LINE:
                # Special case for r_line which is a list of arrays
                if prop not in self._properties:
                    self._properties[prop] = values  # First append
                else:
                    self._properties[prop].extend(values)  # Subsequent appends
            else:
                # Convert to numpy array
                new_values = np.array(values)
                if prop not in self._properties:
                    self._properties[prop] = new_values  # First append
                else:
                    self._properties[prop] = np.concatenate([  # Subsequent appends
                        self._properties[prop],
                        new_values
                    ])
        
        self.n_field_lines = len(field_lines)
        logger.debug("Appended %d field lines", self.n_field_lines)
        
    def estimate_size(self) -> float:
        """Estimate the memory size of the FieldLineSet in bytes.
        
        Returns:
            float: Estimated size in bytes
        """
        
        total_bytes = 0
        
        # Get size of each enabled property
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                continue
            
            if prop not in self._properties:
                continue
            
            values = self._properties[prop]
            
            # Special case for r_line which is a list of arrays
            if prop == PropertyName.R_LINE:
                total_bytes += sum(r.nbytes for r in values)
            else:
                # For numpy arrays, use nbytes property
                if isinstance(values, np.ndarray):
                    total_bytes += values.nbytes
                # For scalar values, use getsizeof
                else:
                    total_bytes += getsizeof(values)
        
        logger.debug("Estimated size: %d bytes", total_bytes)
        return total_bytes
        
        
    def write_to_fits(self, filename: str) -> None:
        """Write field lines to a FITS file.

        Args:
            filename: Path to the output FITS file.
        """
        logger.info("Writing %d field lines to %s", len(self.footpoint_id), filename)
   
        # Import here to avoid circular dependency
        from src.line.io import FieldLineSetFitsIO
        io = FieldLineSetFitsIO(self)
        io.save_to_fits(filename)
        logger.debug("Successfully wrote field lines to %s", filename)

    def __repr__(self) -> str:
        """Return string representation of FieldLineSet."""
        enabled_props = [
            prop.key_name 
            for prop in PropertyName 
            if getattr(self.save_flags, prop.flag_name)
        ]
        
        sizes = {}
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                continue
            if prop not in self._properties:
                continue
                
            if prop == PropertyName.R_LINE:
                sizes[prop.key_name] = f"[{len(self._properties[prop])} lines]"
            elif isinstance(self._properties[prop], np.ndarray):
                sizes[prop.key_name] = f"[shape={self._properties[prop].shape}]"
            else:
                sizes[prop.key_name] = f"[{type(self._properties[prop]).__name__}]"
        
        props_str = ", ".join(f"{prop}: {sizes.get(prop, 'None')}" 
                            for prop in enabled_props)
        
        return f"FieldLineSet(n_lines={self.n_field_lines}, properties={{{props_str}}})"

    def _validate_type(self, prop: PropertyName, value: Any) -> Any:
        """Validate and convert value to match expected type for property."""
        value = super()._validate_type(prop, value)
        
        # Convert to array with proper dtype for FieldLineSet
        expected_type = PROPERTY_TYPES[prop]
        if expected_type.dtype is not None:
            if isinstance(value, np.ndarray):
                if value.dtype != expected_type.dtype:
                    value = value.astype(expected_type.dtype)
            elif isinstance(value, (int, float)):
                value = np.array(value, dtype=expected_type.dtype)
        
        return value

    def create_template(self) -> None:
        """Initialize properties dictionary based on current save flags."""
        self._properties = {}
        self.n_field_lines = 0
        logger.debug("Created template with save flags: %s", repr(self.save_flags))