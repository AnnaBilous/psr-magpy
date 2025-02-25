# Standard library imports

import logging
from typing import  Any


# Third-party imports
import numpy as np

from .field_line_base import FieldLineBase, PropertyName

# Initialize logger
logger = logging.getLogger(__name__)

class FieldLine(FieldLineBase):
    """Single field line with scalar values"""
    
    def __init__(self, **save_flags) -> None:
        """Initialize single field line.
        
        Args:
            **save_flags: Boolean flags controlling which properties to save
        """
        super().__init__(**save_flags)
        # Initialize footpoint_id with default value since it's always enabled
        self._set_property(PropertyName.FOOTPOINT_ID, -1)
        logger.info(repr(self))
    
    def _set_property(self, prop: PropertyName, value: Any) -> None:
        """Set a property value, ensuring it's a single value except for r_line."""
        # Get validated value from base class
        value = super()._validate_type(prop, value)
        
        # Only check collection status - type validation is done in base class
        if prop != PropertyName.R_LINE and self._is_collection(value):
            raise TypeError(f"Expected single value for {prop.key_name}, got collection")
        
        self._properties[prop] = value

    def __repr__(self) -> str:
        """Return string representation of FieldLine."""
        enabled_props = [
            prop.key_name 
            for prop in PropertyName 
            if prop in self._properties
        ]
        
        props_str = []
        for prop_name in enabled_props:
            prop = PropertyName[prop_name.upper()]
            value = self._properties[prop]
            
            if prop == PropertyName.R_LINE:
                if value is not None:
                    props_str.append(f"{prop_name}: [points={len(value)}]")
            else:
                props_str.append(f"{prop_name}: {value}")
        
        return f"FieldLine({', '.join(props_str)})"

