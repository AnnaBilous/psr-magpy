from typing import Any, List, Union
from sys import getsizeof
import numpy as np
from .field_line import FieldLine, PropertyName

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldLineSet(FieldLine):
    """Collection of field lines with array values"""
    def __init__(self, field_lines: Union[List[FieldLine], FieldLine], copy_data: bool = True) -> None:
        
        if isinstance(field_lines, FieldLine):
            field_lines = [field_lines]
        elif not all(isinstance(line, FieldLine) for line in field_lines):
            raise ValueError("All items in field_lines must be instances of FieldLine")
        
        template = field_lines[0]
        # Get template's save flags and pass directly to parent init
        save_flags_dict = {
            flag: getattr(template.save_flags, flag)
            for flag in vars(template.save_flags)
        }
        super().__init__(**save_flags_dict)
        
        self.n_field_lines = 0
        
        logger.debug("Initialized FieldLineSet with field lines, template: %s", repr(template.save_flags))
        if copy_data:
            self.append(field_lines)
    
   
           
    def append(self, field_lines: List[FieldLine]) -> None:
        """Append field lines to the set.
        
        Args:
            field_lines: List of FieldLine objects to append
            
        Raises:
            ValueError: If field lines have incompatible save flags
        """
        # Validate save flags compatibility
        for line in field_lines:
            if vars(line.save_flags) != vars(self.save_flags):
                raise ValueError("All field lines must have the same save flags configuration")
        
        for prop in PropertyName:
            if not getattr(self.save_flags, prop.flag_name):
                continue
            
            values = [line.get_property(prop) for line in field_lines]
            
            # Special handling for R_LINE which is a list of coordinate arrays
            if prop == PropertyName.R_LINE:
                if self._properties[prop] is None:
                    self._properties[prop] = values  # List of arrays
                else:
                    self._properties[prop].extend(values)
                continue
            
            # For other properties, convert to numpy array
            if self._properties[prop] is None:
                self._properties[prop] = np.array(values)
            else:
                self._properties[prop] = np.append(self._properties[prop], values, axis=0)
            
        logger.debug("Appended %d field lines to set", len(field_lines))
        self.n_field_lines += len(field_lines)
        
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
                total_bytes += sum(r.nbytes for r in values if r is not None)
            else:
                # For numpy arrays, use nbytes property
                if isinstance(values, np.ndarray):
                    total_bytes += values.nbytes
                # For scalar values, use getsizeof
                else:
                    total_bytes += getsizeof(values)
        
        logger.debug("Estimated size: %d bytes", total_bytes)
        return total_bytes
        
