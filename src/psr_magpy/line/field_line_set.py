import logging
from typing import List, Union
from sys import getsizeof

import numpy as np

from .field_line import FieldLine, PropertyName
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class FieldLineSet(FieldLine):
    """Collection of field lines with array values"""

    # List of properties that are stored as lists of arrays
    ARRAY_LIST_PROPERTIES = [PropertyName.R_LINE, PropertyName.CURVATURE_LINE, PropertyName.PA_LINE]

    def __init__(
        self, field_lines: Union[List[FieldLine], FieldLine], copy_data: bool = True
    ) -> None:
        """Initialize a FieldLineSet from a list of field lines or a single field line."""
        logger.debug(f"Initializing FieldLineSet with {type(field_lines).__name__}")
        
        # Handle both direct and imported class references
        if isinstance(field_lines, list):
            # Check if it's a list of FieldLine objects
            if not field_lines:
                raise ValueError("Empty field_lines list provided")
                
            # Check the first item's type
            first_item = field_lines[0]
            if not (isinstance(first_item, FieldLine) or 
                   first_item.__class__.__name__ == 'FieldLine'):
                error_msg = f"Expected FieldLine objects, got {type(first_item).__name__}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif isinstance(field_lines, FieldLine) or field_lines.__class__.__name__ == 'FieldLine':
            field_lines = [field_lines]
            logger.debug("Converted single FieldLine to list")
        else:
            error_msg = f"Expected FieldLine or list of FieldLine, got {type(field_lines).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        template = field_lines[0]
        # Get template's save flags and pass directly to parent init
        save_flags_dict = {
            flag: getattr(template.save_flags, flag)
            for flag in vars(template.save_flags)
        }
        super().__init__(**save_flags_dict)

        self.n_field_lines = 0

        logger.debug(f"Initialized FieldLineSet with template: {template.save_flags}")
        if copy_data:
            logger.debug("Copying data from field lines")
            self.append(field_lines)
        else:
            logger.debug("Skipping data copy (copy_data=False)")

    def append(
        self, field_lines: Union[List[FieldLine], FieldLine, "FieldLineSet"]
    ) -> None:
        """Append field lines to the set.

        Parameters
        ----------
        field_lines : Union[List[FieldLine], FieldLine, FieldLineSet]
            Field lines to append. Can be:
            - A single FieldLine instance
            - A list of FieldLine objects
            - Another FieldLineSet instance

        Raises
        ------
        ValueError
            If field lines have incompatible save flags
        """
        logger.debug(f"Appending {type(field_lines).__name__} to FieldLineSet")
        
        # Convert single FieldLine to list (checking both direct and by name)
        is_field_line = (isinstance(field_lines, FieldLine) or 
                         field_lines.__class__.__name__ == 'FieldLine')
        is_field_line_set = (isinstance(field_lines, FieldLineSet) or 
                            field_lines.__class__.__name__ == 'FieldLineSet')
        
        if is_field_line and not is_field_line_set:
            field_lines = [field_lines]
            logger.debug("Converted single FieldLine to list")

        # Handle FieldLineSet
        if is_field_line_set:
            # Validate save flags compatibility
            if vars(field_lines.save_flags) != vars(self.save_flags):
                error_msg = "FieldLineSet has incompatible save flags configuration"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"Appending FieldLineSet with {field_lines.n_field_lines} lines")
            
            # Append each property directly
            for prop in PropertyName:
                if not getattr(self.save_flags, prop.flag_name):
                    logger.debug(f"Skipping disabled property: {prop.key_name}")
                    continue

                source_data = field_lines._properties[prop]
                if source_data is None:
                    logger.debug(f"Source data for {prop.key_name} is None, skipping")
                    continue

                # Special handling for properties that are lists of arrays
                if prop in self.ARRAY_LIST_PROPERTIES:
                    if self._properties[prop] is None:
                        logger.debug(f"Initializing {prop.key_name} with source data")
                        self._properties[prop] = source_data.copy()  # Copy list of arrays
                    else:
                        logger.debug(f"Extending {prop.key_name} with {len(source_data)} arrays")
                        self._properties[prop].extend(source_data)
                    continue

                # For other properties, use numpy append
                if self._properties[prop] is None:
                    logger.debug(f"Initializing {prop.key_name} with source data")
                    self._properties[prop] = np.array(source_data)
                else:
                    logger.debug(f"Appending to {prop.key_name}, adding {len(source_data)} items")
                    self._properties[prop] = np.append(
                        self._properties[prop], source_data, axis=0
                    )

            self.n_field_lines += field_lines.n_field_lines
            logger.debug(
                f"Appended FieldLineSet with {field_lines.n_field_lines} lines, total now: {self.n_field_lines}"
            )
            return

        # Handle list of FieldLine objects (existing functionality)
        # Validate save flags compatibility
        for i, line in enumerate(field_lines):
            if vars(line.save_flags) != vars(self.save_flags):
                error_msg = f"Field line {i} has incompatible save flags configuration"
                logger.error(error_msg)
                raise ValueError(
                    "All field lines must have the same save flags configuration"
                )

        logger.debug(f"Appending {len(field_lines)} individual field lines")
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                logger.debug(f"Skipping disabled property: {prop.key_name}")
                continue

            values = [line.get_property(prop) for line in field_lines]
            logger.debug(f"Collected {len(values)} values for {prop.key_name}")

            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                if self._properties[prop] is None:
                    logger.debug(f"Initializing {prop.key_name} with collected values")
                    self._properties[prop] = values  # List of arrays
                else:
                    logger.debug(f"Extending {prop.key_name} with {len(values)} arrays")
                    self._properties[prop].extend(values)
                continue

            # For other properties, convert to numpy array
            if self._properties[prop] is None:
                logger.debug(f"Initializing {prop.key_name} with numpy array")
                self._properties[prop] = np.array(values)
            else:
                logger.debug(f"Appending to {prop.key_name} numpy array")
                self._properties[prop] = np.append(
                    self._properties[prop], values, axis=0
                )

        self.n_field_lines += len(field_lines)
        logger.debug(f"Appended {len(field_lines)} field lines, total now: {self.n_field_lines}")

    def estimate_size(self) -> float:
        """Estimate the memory size of the FieldLineSet in bytes.

        Returns:
            float: Estimated size in bytes
        """
        logger.debug("Estimating memory size of FieldLineSet")
        total_bytes = 0

        # Get size of each enabled property
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                logger.debug(f"Skipping disabled property: {prop.key_name}")
                continue

            if prop not in self._properties or self._properties[prop] is None:
                logger.debug(f"Property {prop.key_name} is None, skipping")
                continue

            values = self._properties[prop]

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
        return total_bytes
        
    def __repr__(self) -> str:
        """Return string representation of the field line set."""
        return f"FieldLineSet(n_lines={self.n_field_lines}, enabled_properties={[flag for flag in vars(self.save_flags) if getattr(self.save_flags, flag)]})"
