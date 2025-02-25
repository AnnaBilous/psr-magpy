# Standard library imports
from dataclasses import dataclass
import logging
from typing import Optional, List, Dict, Any, TypeVar
from enum import Enum

# Third-party imports
import numpy as np

# Type definitions for single values
ScalarInt = int
ScalarFloat = float
ScalarArray = np.ndarray

# Type definitions for collection values
ArrayInt = np.ndarray
ArrayFloat = np.ndarray
ArrayList = List[np.ndarray]

# Generic type for properties
T = TypeVar('T')

class PropertyValue:
    """Base class for property values that can be either single or collection"""
    def __init__(self, single_type: type, collection_type: type, dtype: Optional[type] = None):
        self.single_type = single_type
        self.collection_type = collection_type
        self.dtype = dtype

class PropertyTypes:
    """Define property types and their collection equivalents"""
    INT_SCALAR_ARRAY = PropertyValue(ScalarInt, ArrayInt, np.int64)
    FLOAT_SCALAR_ARRAY = PropertyValue(ScalarFloat, ArrayFloat, np.float32)
    FLOAT_ARRAY_LIST = PropertyValue(ScalarArray, ArrayList, np.float32)

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

    @property
    def expected_type(self) -> PropertyValue:
        """Get the expected type for this property"""
        return PROPERTY_TYPES[self]

# Separate mapping for property types
PROPERTY_TYPES = {
    PropertyName.FOOTPOINT_ID: PropertyTypes.INT_SCALAR_ARRAY,
    PropertyName.R_LINE: PropertyTypes.FLOAT_ARRAY_LIST,
    PropertyName.R_MAX: PropertyTypes.FLOAT_SCALAR_ARRAY,
    PropertyName.R_CYL_MAX: PropertyTypes.FLOAT_SCALAR_ARRAY,
    PropertyName.Z_ABS_MAX: PropertyTypes.FLOAT_SCALAR_ARRAY,
    PropertyName.B_INI: PropertyTypes.FLOAT_SCALAR_ARRAY,
    PropertyName.B_R_INI_ANGLE: PropertyTypes.FLOAT_SCALAR_ARRAY,
}

@dataclass
class SaveFlags:
    """Flags controlling which properties should be saved/initialized"""
    def __init__(self, **kwargs):
        # Initialize all flags to False by default, except footpoint_id
        for prop in PropertyName:
            flag_name = prop.flag_name
            if not flag_name.startswith('save_'):
                raise ValueError(f"Invalid flag name: {flag_name}. Must start with 'save_'")
            
            default_value = True if prop == PropertyName.FOOTPOINT_ID else False
            value = kwargs.get(flag_name, default_value)
            setattr(self, flag_name, value)
            logger.debug(f"Initialized {flag_name} = {getattr(self, flag_name)}")

    def __repr__(self) -> str:
        """Return a string representation of enabled save flags."""
        enabled_flags = [
            prop.flag_name
            for prop in PropertyName
            if getattr(self, prop.flag_name)
        ]
        return f"SaveFlags(enabled={enabled_flags})"

class FieldLineBase:
    """Base class for field line data structures"""
    
    @classmethod
    def _create_property(cls, prop_name: PropertyName):
        """Factory method to create properties"""
        @property
        def prop(self): 
            if not getattr(self.save_flags, prop_name.flag_name):
                raise AttributeError(f"{prop_name.key_name} is not enabled in save_flags")
            return self._get_property(prop_name)
        return prop
    
    def __init__(self, **save_flags) -> None:
        """Initialize with save flags."""
        self.save_flags = SaveFlags(**save_flags)
        self._properties: Dict[PropertyName, Any] = {}
        
        for prop in PropertyName:
            if not hasattr(self.__class__, prop.key_name):
                setattr(self.__class__, prop.key_name, self._create_property(prop))
    
    def _get_property(self, prop: PropertyName) -> Any:
        """Get a property value from the properties dictionary."""
        if not getattr(self.save_flags, prop.flag_name):
            raise AttributeError(f"{prop.key_name} is not enabled in save_flags")        
        try:
            return self._properties[prop]
        except KeyError:
            raise KeyError(f"Property {prop.key_name} not found. Ensure it has been computed or updated.")
        
    def _is_collection(self, value: Any) -> bool:
        """Check if value is a collection type."""
        return isinstance(value, (np.ndarray, list))

    def _validate_type(self, prop: PropertyName, value: Any) -> Any:
        """Validate and convert value to match expected type for property."""
        expected_type = PROPERTY_TYPES[prop]
        
        # Handle numpy scalar types
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        
        # Check basic type
        if not isinstance(value, expected_type.single_type) and \
           not isinstance(value, expected_type.collection_type):
            raise TypeError(f"Expected {expected_type.single_type} or {expected_type.collection_type} "
                          f"for {prop.key_name}, got {type(value)}")
        
        # Don't convert to arrays in base class - let derived classes handle that
        return value

    def update(self, **kwargs) -> None:
        """Update field line data according to save flags.
        
        Args:
            **kwargs: Values to update. Keys should match property names (not flag names)
        """
        for prop in PropertyName:
            if prop.key_name in kwargs:  # Look for 'r_line' instead of 'save_r_line'
                self._set_property(prop, kwargs[prop.key_name])

    def copy_structure(self) -> 'FieldLineBase':
        """Create a new instance with the same save flags configuration."""
        save_flags_dict = {
            prop.flag_name: getattr(self.save_flags, prop.flag_name)
            for prop in PropertyName
        }
        return self.__class__(**save_flags_dict) 