# Standard library imports
from dataclasses import dataclass
import logging
from typing import Optional, List, Dict, Any, TypeVar, Union
from enum import Enum

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Initialize logger
logger = logging.getLogger(__name__)

class PropertyName(Enum):
    """Enum of all available properties"""
    FOOTPOINT_ID = "save_footpoint_id"
    R_LINE = "save_r_line"
    R_MAX = "save_r_max"
    R_CYL_MAX = "save_r_cyl_max"
    Z_ABS_MAX = "save_z_abs_max"
    B_INI = "save_b_ini"
    B_R_INI_ANGLE = "save_b_r_ini_angle"

    @property
    def flag_name(self) -> str:
        """Get the corresponding save flag name"""
        return self.value

    @property
    def key_name(self) -> str:
        """Get the property key name (lowercase)"""
        return self.name.lower()

@dataclass
class SaveFlags:
    """Class holding boolean flags for which properties to save.
    
    footpoint_id is always True and cannot be disabled.
    """
    def __init__(self, **kwargs):
        # Set default values for all properties from PropertyName enum
        for prop in PropertyName:
            # footpoint_id is always True, others default to False
            default_value = True if prop == PropertyName.FOOTPOINT_ID else False
            # Allow override from kwargs, but force footpoint_id to True
            value = kwargs.get(prop.flag_name, default_value)
            if prop == PropertyName.FOOTPOINT_ID:
                value = True
            setattr(self, prop.flag_name, value)


class FieldLine:
    """Base class for field lines with configurable properties.
    
    This class provides the basic structure for storing field line properties
    and managing which properties should be saved.
    """
    
    def __init__(self, **kwargs):
        """Initialize field line with optional save flags.
        
        Args:
            **kwargs: Boolean flags for which properties to save.
                     save_footpoint_id will always be True regardless of input.
        """
        # Initialize save flags, ensuring footpoint_id is True
        kwargs['save_footpoint_id'] = True
        self.save_flags = SaveFlags(**kwargs)
        logger.debug(f"Initialized save flags: {self.save_flags}")
        
        # Initialize properties dictionary dynamically from PropertyName enum
        self._properties = {prop: None for prop in PropertyName}
        
    def __repr__(self) -> str:
        """Return string representation of the field line."""
        enabled_flags = [
            flag for flag in vars(self.save_flags) 
            if getattr(self.save_flags, flag)
        ]
        return f"FieldLine(enabled_properties={enabled_flags})"

    def get_property(self, prop: Union[PropertyName, str]) -> Optional[Any]:
        """Get a property value if it's enabled.
        
        Args:
            prop: Property to get, either as PropertyName enum or string
                 (e.g., PropertyName.R_MAX or 'r_max')
            
        Returns:
            Property value or None if not enabled/set
            
        Raises:
            ValueError: If property is not enabled but accessed
            KeyError: If string property name is not valid
        """
        # Convert string to PropertyName if needed
        if isinstance(prop, str):
            try:
                prop = PropertyName[prop.upper()]
            except KeyError:
                logger.error(f"Invalid property name: {prop}")
                raise KeyError(f"Invalid property name: {prop}. Must be one of: {[p.key_name for p in PropertyName]}")
        
        if not getattr(self.save_flags, prop.flag_name):
            logger.warning(f"Attempting to access disabled property: {prop.key_name}")
            raise ValueError(f"Property {prop.key_name} is not enabled")
        return self._properties[prop]

    def set_property(self, **kwargs: Any) -> None:
        """Set multiple property values if they're enabled.
        
        Args:
            **kwargs: Property values to set, where keys are PropertyName.key_name
                     Example: set_property(r_line=coordinates, r_max=max_value)
            
        Raises:
            ValueError: If any property is not enabled but attempted to be set
            KeyError: If property name is not valid
        """
        for key, value in kwargs.items():
            try:
                # Convert key to PropertyName enum
                prop = PropertyName[key.upper()]
                
                # Check if property is enabled
                if not getattr(self.save_flags, prop.flag_name):
                    logger.warning(f"Attempting to set disabled property: {prop.key_name}")
                    raise ValueError(f"Property {prop.key_name} is not enabled")
                
                # Set the property
                self._properties[prop] = value
                logger.debug(f"Set property {prop.key_name}")
                
            except KeyError:
                logger.error(f"Invalid property name: {key}")
                raise KeyError(f"Invalid property name: {key}. Must be one of: {[p.key_name for p in PropertyName]}")

    def copy_structure(self) -> 'FieldLine':
        """Create a new instance with the same save flags configuration.
        
        Returns:
            A new FieldLineBase instance with the same save flags but empty properties
        """
        # Get the current save flags as a dictionary
        save_flags_dict = {
            flag: getattr(self.save_flags, flag)
            for flag in vars(self.save_flags)
        }
        
        # Create new instance with same flags
        logger.debug(f"Copying structure with flags: {save_flags_dict}")
        return self.__class__(**save_flags_dict)

