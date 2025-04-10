import logging
from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, Set, List
from enum import Enum, auto

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class PropertyName(Enum):
    """Enum of all available properties"""

    FOOTPOINT_ID = "save_footpoint_id"
    R_LINE = "save_r_line"
    PA_LINE = "save_pa_line"
    CURVATURE_LINE = "save_curvature_line"
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


class PropertyGroup(Enum):
    """Enum of property groups for convenience"""
    
    SAVE_ALL = auto()
    SAVE_LINE_DATA = auto()
    SAVE_LINE_MAX = auto()
    SAVE_B_INI = auto()
    
    @property
    def properties(self) -> Set[PropertyName]:
        """Get the set of properties in this group"""
        if self == PropertyGroup.SAVE_ALL:
            return set(PropertyName)
        elif self == PropertyGroup.SAVE_LINE_DATA:
            return {PropertyName.R_LINE, PropertyName.PA_LINE, PropertyName.CURVATURE_LINE}
        elif self == PropertyGroup.SAVE_LINE_MAX:
            return {PropertyName.R_MAX, PropertyName.R_CYL_MAX, PropertyName.Z_ABS_MAX}
        elif self == PropertyGroup.SAVE_B_INI:
            return {PropertyName.B_INI, PropertyName.B_R_INI_ANGLE}
        return set()


@dataclass
class SaveFlags:
    """Class holding boolean flags for which properties to save.

    footpoint_id is always True and cannot be disabled.
    """

    def __init__(self, **kwargs):
        # Process property groups first
        property_flags = {}
        
        # Check for property groups
        for group in PropertyGroup:
            group_name = group.name.lower()
            if group_name in kwargs and kwargs[group_name]:
                # Enable all properties in this group
                for prop in group.properties:
                    property_flags[prop.flag_name] = True
                
        # Set default values for all properties from PropertyName enum
        for prop in PropertyName:
            # footpoint_id is always True, others default to False
            default_value = True if prop == PropertyName.FOOTPOINT_ID else False
            # Allow override from kwargs, but force footpoint_id to True
            value = kwargs.get(prop.flag_name, property_flags.get(prop.flag_name, default_value))
            if prop == PropertyName.FOOTPOINT_ID:
                value = True
            setattr(self, prop.flag_name, value)
        
    def __repr__(self) -> str:
        """Return string representation of save flags."""
        enabled_flags = [flag for flag in vars(self) if getattr(self, flag)]
        return f"SaveFlags(enabled={enabled_flags})"


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
                     
                     Property groups can be specified:
                     - save_all: Enable all properties
                     - save_line_data: Enable r_line, pa_line, curvature_line
                     - save_line_max: Enable r_max, r_cyl_max, z_abs_max
                     - save_b_ini: Enable b_ini, b_r_ini_angle
        """
        # Initialize save flags, ensuring footpoint_id is True
        kwargs["save_footpoint_id"] = True
        self.save_flags = SaveFlags(**kwargs)
        logger.debug(f"Initialized FieldLine with save flags: {self.save_flags}")

        # Initialize properties dictionary dynamically from PropertyName enum
        self._properties = {prop: None for prop in PropertyName}
        
        # Log enabled properties at debug level
        if logger.isEnabledFor(logging.DEBUG):
            enabled_props = [
                prop.key_name
                for prop in PropertyName
                if getattr(self.save_flags, prop.flag_name)
            ]
            logger.debug(f"Enabled properties: {enabled_props}")

    def __repr__(self) -> str:
        """Return string representation of the field line."""
        enabled_flags = [
            flag for flag in vars(self.save_flags) if getattr(self.save_flags, flag)
        ]
        
        # Add property values to the representation
        property_values = {}
        for prop in PropertyName:
            if getattr(self.save_flags, prop.flag_name):
                property_values[prop.key_name] = self._properties[prop]
        
        return (f"FieldLine(enabled_properties={enabled_flags}, "
                f"values={property_values})")

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
                logger.debug(f"Converted string '{prop}' to PropertyName enum")
            except KeyError:
                error_msg = f"Invalid property name: {prop}"
                logger.error(f"{error_msg}. Valid properties: {[p.key_name for p in PropertyName]}")
                raise KeyError(
                    f"{error_msg}. Must be one of: {[p.key_name for p in PropertyName]}"
                )

        if not getattr(self.save_flags, prop.flag_name):
            warning_msg = f"Attempting to access disabled property: {prop.key_name}"
            logger.warning(warning_msg)
            raise ValueError(f"Property {prop.key_name} is not enabled")
            
        #logger.debug(f"Getting property: {prop.key_name}")
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
        logger.debug(f"Setting properties: {list(kwargs.keys())}")
        
        for key, value in kwargs.items():
            try:
                # Convert key to PropertyName enum
                prop = PropertyName[key.upper()]

                # Check if property is enabled
                if not getattr(self.save_flags, prop.flag_name):
                    warning_msg = f"Attempting to set disabled property: {prop.key_name}"
                    logger.warning(warning_msg)
                    raise ValueError(f"Property {prop.key_name} is not enabled")

                # Set the property
                self._properties[prop] = value
                logger.debug(f"Set property {prop.key_name}")

            except KeyError:
                error_msg = f"Invalid property name: {key}"
                logger.error(f"{error_msg}. Valid properties: {[p.key_name for p in PropertyName]}")
                raise KeyError(
                    f"{error_msg}. Must be one of: {[p.key_name for p in PropertyName]}"
                )

    def copy_structure(self) -> "FieldLine":
        """Create a new instance with the same save flags configuration.

        Returns:
            A new FieldLineBase instance with the same save flags but empty properties
        """
        # Get the current save flags as a dictionary
        save_flags_dict = {
            flag: getattr(self.save_flags, flag) for flag in vars(self.save_flags)
        }

        # Create new instance with same flags
        logger.debug(f"Copying structure with flags: {save_flags_dict}")
        new_instance = self.__class__(**save_flags_dict)
        logger.debug(f"Created new {self.__class__.__name__} instance with same structure")
        return new_instance
