from typing import List, Optional, Any, Union, Dict
from astropy.coordinates import CartesianRepresentation
from sys import getsizeof
import numpy as np

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName

if False:
    from .field_line_set import FieldLineSet

logger = get_logger(__name__, console_format_brief=True)


class FieldLineSet(FieldLine):
    """Collection of field lines with array values."""

    # List of properties that are stored as lists of arrays
    ARRAY_LIST_PROPERTIES = [PropertyName.R_LINE, PropertyName.PA_LINE]

    def __init__(
        self, field_lines: Union[List[FieldLine], FieldLine], copy_data: bool = True
    ) -> None:
        """Initialize a FieldLineSet from a list of field lines or a single field line."""
        logger.debug(f"Initializing FieldLineSet with {type(field_lines).__name__}")
        
        # Get template and validate input
        template = self._get_template(field_lines)
        
        # Use PropertySystem from parent class
        super().__init__(**template.property_system.get_flags_dict())

        self.n_field_lines = 0

        logger.debug(f"Initialized FieldLineSet with template: {self.save_flags}")
        if copy_data:
            logger.debug("Copying data from field lines")
            self.append(field_lines)
        else:
            logger.debug("Skipping data copy (copy_data=False)")

    def _get_template(self, field_lines: Union[List[FieldLine], FieldLine]) -> FieldLine:
        """Get template field line and validate input."""
        if isinstance(field_lines, list):
            if not field_lines:
                error_msg = "Empty field_lines list provided"
                logger.error(error_msg)
                raise ValueError(error_msg)
            return field_lines[0]
        elif isinstance(field_lines, FieldLine):
            return field_lines
        else:
            error_msg = f"Expected FieldLine or list of FieldLine, got {type(field_lines).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def append(
        self, field_lines: Union[List[FieldLine], FieldLine, "FieldLineSet"]
    ) -> None:
        """Append field lines to the set.

        Raises
        ------
        ValueError
            If field lines have incompatible save flags
        """
        logger.debug(f"Appending {type(field_lines).__name__} to FieldLineSet")
        
        # Handle FieldLineSet
        if isinstance(field_lines, FieldLineSet):
            self._append_field_line_set(field_lines)
            return

        # Convert single FieldLine to list
        if isinstance(field_lines, FieldLine):
            field_lines = [field_lines]
            logger.debug("Converted single FieldLine to list")

        # Handle list of FieldLine objects
        if isinstance(field_lines, list):
            self._append_field_line_list(field_lines)
        else:
            error_msg = f"Unexpected type: {type(field_lines).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _append_field_line_set(self, field_line_set: "FieldLineSet") -> None:
        """Append another FieldLineSet."""
        # Validate save flags compatibility using PropertySystem
        if (self.property_system.get_flags_dict() != 
            field_line_set.property_system.get_flags_dict()):
            error_msg = "FieldLineSet has incompatible save flags configuration"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Appending FieldLineSet with {field_line_set.n_field_lines} lines")
        
        # Append each property directly
        for prop in self.property_system.get_enabled_properties():
            source_data = field_line_set.property_system.get_property(prop)
            if source_data is None:
                logger.debug(f"Source data for {prop.key_name} is None, skipping")
                continue

            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                current_data = self.property_system.get_property(prop)
                if current_data is None:
                    logger.debug(f"Initializing {prop.key_name} with source data")
                    # Set property directly to avoid validation issues
                    self.property_system._properties[prop] = source_data.copy()
                else:
                    logger.debug(f"Extending {prop.key_name} with {len(source_data)} arrays")
                    current_data.extend(source_data)
                continue

            # For other properties, use numpy append
            current_data = self.property_system.get_property(prop)
            if current_data is None:
                logger.debug(f"Initializing {prop.key_name} with source data")
                # Set property directly to avoid validation issues with arrays
                self.property_system._properties[prop] = np.array(source_data)
            else:
                logger.debug(f"Appending to {prop.key_name}, adding {len(source_data)} items")
                # Filter out None values before appending
                valid_source_data = [x for x in source_data if x is not None]
                if valid_source_data:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.append(current_data, valid_source_data, axis=0)

        self.n_field_lines += field_line_set.n_field_lines
        logger.debug(
            f"Appended FieldLineSet with {field_line_set.n_field_lines} lines, total now: {self.n_field_lines}"
        )

    def _append_field_line_list(self, field_lines: List[FieldLine]) -> None:
        """Append a list of FieldLine objects."""
        # Validate save flags compatibility using PropertySystem
        for i, line in enumerate(field_lines):
            if (self.property_system.get_flags_dict() != 
                line.property_system.get_flags_dict()):
                error_msg = f"Field line {i} has incompatible save flags configuration"
                logger.error(error_msg)
                raise ValueError("All field lines must have the same save flags configuration")

        logger.debug(f"Appending {len(field_lines)} individual field lines")
        
        for prop in self.property_system.get_enabled_properties():
            values = [line.property_system.get_property(prop) for line in field_lines]
            logger.debug(f"Collected {len(values)} values for {prop.key_name}")

            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                current_data = self.property_system.get_property(prop)
                if current_data is None:
                    logger.debug(f"Initializing {prop.key_name} with collected values")
                    # Set property directly to avoid validation issues
                    self.property_system._properties[prop] = values
                else:
                    logger.debug(f"Extending {prop.key_name} with {len(values)} arrays")
                    current_data.extend(values)
                continue

            # For other properties, convert to numpy array
            current_data = self.property_system.get_property(prop)
            if current_data is None:
                logger.debug(f"Initializing {prop.key_name} with numpy array")
                # Filter out None values before creating array
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.array(valid_values)
            else:
                logger.debug(f"Appending to {prop.key_name} numpy array")
                # Filter out None values before appending
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.append(current_data, valid_values, axis=0)

        self.n_field_lines += len(field_lines)
        logger.debug(f"Appended {len(field_lines)} field lines, total now: {self.n_field_lines}")

    def estimate_size(self) -> float:
        """Estimate the memory size of the FieldLineSet in bytes."""
        logger.debug("Estimating memory size of FieldLineSet")
        total_bytes = 0

        # Get size of each enabled property using PropertySystem
        for prop in self.property_system.get_enabled_properties():
            values = self.property_system.get_property(prop)
            if values is None:
                logger.debug(f"Property {prop.key_name} is None, skipping")
                continue

            # Special case for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                prop_size = sum(r.nbytes for r in values if r is not None and hasattr(r, 'nbytes'))
                logger.debug(f"{prop.key_name} size: {prop_size} bytes from {len(values)} arrays")
            else:
                # For numpy arrays, use nbytes property
                if isinstance(values, np.ndarray):
                    prop_size = values.nbytes
                    logger.debug(f"{prop.key_name} size: {prop_size} bytes (numpy array)")
                # For scalar values, use getsizeof
                else:
                    prop_size = getsizeof(values)
                    logger.debug(f"{prop.key_name} size: {prop_size} bytes (scalar)")
            
            total_bytes += prop_size

        logger.debug(f"Total estimated size: {total_bytes} bytes ({total_bytes/1024/1024:.2f} MB)")
        return float(total_bytes)
        
    def get_property(self, property_name: Union[str, PropertyName]) -> Union[np.ndarray, List, None]:
        """Get property value with enabled check."""
        # Convert string to PropertyName if needed
        if isinstance(property_name, str):
            try:
                # First try using the string as an enum name (e.g., 'R_MAX')
                property_name = PropertyName[property_name.upper()]
            except KeyError:
                # If that fails, try to find by key_name (e.g., 'r_max')
                for prop in PropertyName:
                    if prop.key_name == property_name.lower():
                        property_name = prop
                        break
                else:
                    logger.warning(f"Unknown property name: {property_name}")
                    return None
        
        # Check if property is enabled
        if property_name not in self.property_system.get_enabled_properties():
            logger.debug(f"Property {property_name.key_name} is not enabled")
            return None
            
        return self.property_system.get_property(property_name)

    def __getattr__(self, name: str):
        """Dynamic property access for enabled properties."""
        # Try to get property if it exists and is enabled
        try:
            # First try using the name as an enum name (e.g., 'R_MAX')
            prop_name = PropertyName[name.upper()]
        except KeyError:
            # If that fails, try to find by key_name (e.g., 'r_max')
            prop_name = None
            for prop in PropertyName:
                if prop.key_name == name.lower():
                    prop_name = prop
                    break
        
        if prop_name and prop_name in self.property_system.get_enabled_properties():
            return self.property_system.get_property(prop_name)
        
        # Fallback to normal attribute error
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return string representation of the field line set."""
        enabled_props = self.property_system.get_enabled_property_names()
        return f"FieldLineSet(n_lines={self.n_field_lines}, enabled_properties={enabled_props})"
