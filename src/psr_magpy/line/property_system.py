"""Unified property management system for field line data.

This module provides a consolidated property management system that combines
property names, validation, storage, and save flags management into a single cohesive interface.

Classes:
    PropertySystem: Unified property management with validation, storage, and save flags
    SaveFlags: Dataclass for holding boolean flags
"""

import logging
from typing import Optional, Any, Union, Dict, Set
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, ValidationError

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.grid.grid import HealpixGrid, SphericalGrid, RandomGrid, GridType

logger = get_logger(__name__)


@dataclass
class SaveFlags:
    """Class holding boolean flags for which properties to save.

    footpoint_id is always True and cannot be disabled.
    """
    
    save_footpoint_id: bool = True
    save_r_line: bool = False
    save_pa_line: bool = False
    save_r_max: bool = False
    save_r_cyl_max: bool = False
    save_z_abs_max: bool = False
    save_b_ini: bool = False
    save_b_r_ini_angle: bool = False
    
    def __repr__(self) -> str:
        """Return string representation of save flags."""
        enabled_flags = [flag for flag in vars(self) if getattr(self, flag)]
        return f"SaveFlags(enabled={enabled_flags})"


class PropertyName(Enum):
    """Enum of all available properties"""
    FOOTPOINT_ID = "save_footpoint_id"
    R_LINE = "save_r_line"
    PA_LINE = "save_pa_line"
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


class PropertyValidationError(ValueError):
    """Custom exception for property validation errors."""
    
    def __init__(self, message: str, property_name: Optional[str] = None, value: Any = None):
        self.message = message
        self.property_name = property_name
        self.value = value
        super().__init__(self.message)


class PropertyValue(BaseModel):
    """Pydantic model for validating individual property values."""
    
    property_name: PropertyName
    value: Any
    enabled: bool = True
    
    @field_validator('property_name')
    @classmethod
    def validate_property_name(cls, v):
        """Validate that property name is a valid PropertyName enum."""
        if not isinstance(v, PropertyName):
            try:
                if isinstance(v, str):
                    v = PropertyName[v.upper()]
                else:
                    raise PropertyValidationError(f"Invalid property name type: {type(v)}")
            except KeyError:
                raise PropertyValidationError(f"Invalid property name: {v}")
        return v
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v, info):
        """Validate property value based on property type."""
        property_name = info.data['property_name']
        if property_name is None:
            return v
        
        # Allow None values for unset properties
        if v is None:
            return v
        
        # Type-specific validation
        if property_name == PropertyName.FOOTPOINT_ID:
            if not isinstance(v, int):
                raise PropertyValidationError(
                    f"footpoint_id must be an integer, got {type(v)}",
                    property_name.key_name, v
                )
            if v < 0:
                raise PropertyValidationError(
                    f"footpoint_id must be non-negative, got {v}",
                    property_name.key_name, v
                )
        
        elif property_name in [PropertyName.R_LINE, PropertyName.PA_LINE]:
            # These should be numpy arrays or lists of coordinates
            if not hasattr(v, '__len__'):
                raise PropertyValidationError(
                    f"{property_name.key_name} must be an array-like object, got {type(v)}",
                    property_name.key_name, v
                )
        
        elif property_name in [PropertyName.R_MAX, PropertyName.R_CYL_MAX, PropertyName.Z_ABS_MAX]:
            # These should be numeric values (can be 0 for cylindrical coordinates)
            if not isinstance(v, (int, float)) or not (v >= 0):
                raise PropertyValidationError(
                    f"{property_name.key_name} must be a non-negative number, got {v}",
                    property_name.key_name, v
                )
        
        elif property_name in [PropertyName.B_INI, PropertyName.B_R_INI_ANGLE]:
            # These should be numeric values (can be negative)
            if not isinstance(v, (int, float)):
                raise PropertyValidationError(
                    f"{property_name.key_name} must be a number, got {type(v)}",
                    property_name.key_name, v
                )
        
        return v


class PropertySystem:
    """Unified property management system with validation, storage, and save flags.
    
    This class consolidates all property-related operations including:
    - Property name validation and conversion
    - Property value validation and storage
    - Property access control
    - Save flags management
    - Comprehensive error handling
    """
    
    def __init__(self, **kwargs):
        """Initialize property system with save flags.
        
        Parameters
        ----------
        **kwargs : bool
            Boolean flags for which properties to save.
            Examples:
            - save_r_line=True, save_r_max=True
            - save_all=True (enables all properties)
        """
        # Process save flags
        self.save_flags = self._process_flags(**kwargs)
        self.enabled_properties = self.get_enabled_properties()
        self._properties: Dict[PropertyName, Any] = {prop: None for prop in PropertyName}
        
        logger.debug(f"Initialized PropertySystem with flags: {self.save_flags}")
        logger.debug(f"Enabled properties: {self.get_enabled_property_names()}")
    
    def get_enabled_properties(self) -> Set[PropertyName]:
        """Get set of enabled PropertyName enums.
        
        Returns
        -------
        Set[PropertyName]
            Set of enabled property enums
        """
        enabled_properties = set()
        for prop in PropertyName:
            if getattr(self.save_flags, prop.flag_name):
                enabled_properties.add(prop)
        return enabled_properties
    
    def get_enabled_property_names(self) -> list:
        """Get list of enabled property key names.
        
        Returns
        -------
        list
            List of enabled property key names (lowercase)
        """
        return [prop.key_name for prop in self.get_enabled_properties()]
    
    def is_property_enabled(self, prop: Union[PropertyName, str]) -> bool:
        """Check if a specific property is enabled.
        
        Parameters
        ----------
        prop : Union[PropertyName, str]
            Property to check
            
        Returns
        -------
        bool
            True if property is enabled, False otherwise
        """
        try:
            property_enum = self._validate_property_name(prop)
            return getattr(self.save_flags, property_enum.flag_name, False)
        except (KeyError, ValueError):
            return False
    
    def get_flags_dict(self) -> Dict[str, bool]:
        """Get save flags as a dictionary.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping flag names to boolean values
        """
        return {
            flag: getattr(self.save_flags, flag) 
            for flag in vars(self.save_flags)
        }
    
    def validate_flags(self) -> None:
        """Validate that save flags are consistent.
        
        Raises
        ------
        ValueError
            If flags are inconsistent (e.g., footpoint_id disabled)
        """
        if not self.save_flags.save_footpoint_id:
            error_msg = "footpoint_id must always be enabled"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("Save flags validation passed")
    
    def _process_flags(self, **kwargs) -> SaveFlags:
        """Process kwargs to create SaveFlags instance.
        
        Parameters
        ----------
        **kwargs : bool
            Boolean flags for properties
            
        Returns
        -------
        SaveFlags
            Processed save flags instance
        """
        # Handle save_all flag
        if kwargs.get('save_all', False):
            # Enable all properties
            flags_dict = {prop.flag_name: True for prop in PropertyName}
        else:
            # Create flags dictionary with defaults
            flags_dict = {}
            for prop in PropertyName:
                # footpoint_id is always True, others default to False
                default_value = True if prop == PropertyName.FOOTPOINT_ID else False
                # Allow override from kwargs, but force footpoint_id to True
                value = kwargs.get(prop.flag_name, default_value)
                if prop == PropertyName.FOOTPOINT_ID:
                    value = True
                flags_dict[prop.flag_name] = value
        
        # Create SaveFlags instance
        save_flags = SaveFlags(**flags_dict)
        
        # Validate the flags
        self._validate_flags_internal(save_flags)
        
        return save_flags
    
    def _validate_flags_internal(self, save_flags: SaveFlags) -> None:
        """Internal validation of save flags.
        
        Parameters
        ----------
        save_flags : SaveFlags
            Save flags to validate
            
        Raises
        ------
        ValueError
            If flags are invalid
        """
        if not save_flags.save_footpoint_id:
            error_msg = "footpoint_id must always be enabled"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log enabled properties at debug level
        if logger.isEnabledFor(logging.DEBUG):
            enabled_props = [
                prop.key_name
                for prop in PropertyName
                if getattr(save_flags, prop.flag_name)
            ]
            logger.debug(f"Enabled properties: {enabled_props}")
    
    def get_property(self, prop: Union[PropertyName, str]) -> Optional[Any]:
        """Get a property value if it's enabled.
        
        Parameters
        ----------
        prop : Union[PropertyName, str]
            Property to get, either as PropertyName enum or string
            
        Returns
        -------
        Optional[Any]
            Property value or None if not set
            
        Raises
        ------
        PropertyValidationError
            If property is not enabled but accessed
        """
        try:
            property_enum = self._validate_property_name(prop)
            self._validate_property_access(property_enum)
            return self._properties[property_enum]
        except PropertyValidationError as e:
            logger.error(str(e))
            raise
    
    def set_property(self, **kwargs: Any) -> None:
        """Set multiple property values if they're enabled.
        
        Parameters
        ----------
        **kwargs : Any
            Property values to set, where keys are PropertyName.key_name
            
        Raises
        ------
        PropertyValidationError
            If any property is not enabled but attempted to be set
        """
        logger.debug(f"Setting properties: {list(kwargs.keys())}")
        for key, value in kwargs.items():
            try:
                property_enum = self._validate_property_name(key)
                self._validate_property_access(property_enum)
                # Always cast footpoint_id to Python int
                if property_enum.key_name == 'footpoint_id' and value is not None:
                    value = int(value)
                validated_value = self._validate_property_value(property_enum, value)
                self._properties[property_enum] = validated_value
                logger.debug(f"Set property {property_enum.key_name}")
            except PropertyValidationError as e:
                logger.error(str(e))
                raise
    
    def get_property_names(self) -> list:
        """Get list of enabled property key names.
        
        Returns
        -------
        list
            List of enabled property key names (lowercase)
        """
        return [prop.key_name for prop in self.enabled_properties]
    
    def get_property_values(self) -> Dict[str, Any]:
        """Get dictionary of property values for enabled properties.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping property key names to their values
        """
        return {
            prop.key_name: self._properties[prop]
            for prop in self.enabled_properties
        }
    
    def clear_properties(self) -> None:
        """Clear all property values (set to None)."""
        for prop in PropertyName:
            self._properties[prop] = None
        logger.debug("Cleared all property values")
    
    def validate_property_set(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate multiple property values for setting.
        
        Parameters
        ----------
        **kwargs : Any
            Property values to validate, where keys are PropertyName.key_name
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of validated property values
            
        Raises
        ------
        PropertyValidationError
            If any property or value is invalid
        """
        validated_properties = {}
        validation_errors = []
        
        for key, value in kwargs.items():
            try:
                # Validate property name
                property_enum = self._validate_property_name(key)
                
                # Validate property access
                self._validate_property_access(property_enum)
                
                # Validate property value
                validated_value = self._validate_property_value(property_enum, value)
                validated_properties[key] = validated_value
                
            except PropertyValidationError as e:
                validation_errors.append(str(e))
        
        if validation_errors:
            error_msg = f"Property validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise PropertyValidationError(error_msg)
        
        logger.debug(f"Validated {len(validated_properties)} properties successfully")
        return validated_properties
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation configuration.
        
        Returns
        -------
        Dict[str, Any]
            Summary of validation settings and enabled properties
        """
        return {
            'enabled_properties': [prop.key_name for prop in self.enabled_properties],
            'total_properties': len(PropertyName),
            'validation_rules': {
                'footpoint_id': 'non-negative integer',
                'r_line': 'array-like object',
                'pa_line': 'array-like object',
                'r_max': 'positive number',
                'r_cyl_max': 'positive number',
                'z_abs_max': 'positive number',
                'b_ini': 'number',
                'b_r_ini_angle': 'number'
            }
        }
    
    def _validate_property_name(self, prop: Union[PropertyName, str]) -> PropertyName:
        """Validate and convert property name to PropertyName enum.
        
        Parameters
        ----------
        prop : Union[PropertyName, str]
            Property name to validate
            
        Returns
        -------
        PropertyName
            Validated PropertyName enum
            
        Raises
        ------
        PropertyValidationError
            If property name is invalid
        """
        try:
            if isinstance(prop, PropertyName):
                return prop
            
            if isinstance(prop, str):
                property_enum = PropertyName[prop.upper()]
                logger.debug(f"Converted string '{prop}' to PropertyName enum")
                return property_enum
            
            raise PropertyValidationError(
                f"Property must be PropertyName enum or string, got {type(prop)}"
            )
            
        except KeyError:
            error_msg = f"Invalid property name: {prop}"
            logger.error(f"{error_msg}. Valid properties: {[p.key_name for p in PropertyName]}")
            raise PropertyValidationError(
                f"{error_msg}. Must be one of: {[p.key_name for p in PropertyName]}"
            )
    
    def _validate_property_access(self, prop: PropertyName) -> None:
        """Validate that a property is enabled for access.
        
        Parameters
        ----------
        prop : PropertyName
            Property to validate access for
            
        Raises
        ------
        PropertyValidationError
            If property is not enabled
        """
        if prop not in self.enabled_properties:
            warning_msg = f"Attempting to access disabled property: {prop.key_name}"
            logger.warning(warning_msg)
            raise PropertyValidationError(
                f"Property {prop.key_name} is not enabled",
                prop.key_name
            )
    
    def _validate_property_value(self, prop: PropertyName, value: Any) -> Any:
        """Validate a property value using Pydantic.
        
        Parameters
        ----------
        prop : PropertyName
            Property to validate value for
        value : Any
            Value to validate
            
        Returns
        -------
        Any
            Validated value
            
        Raises
        ------
        PropertyValidationError
            If value is invalid for the property
        """
        try:
            # Create PropertyValue model for validation
            property_value = PropertyValue(
                property_name=prop,
                value=value,
                enabled=prop in self.enabled_properties
            )
            return property_value.value
            
        except ValidationError as e:
            # Convert Pydantic validation error to our custom error
            error_details = []
            for error in e.errors():
                field = error.get('loc', ['unknown'])[-1]
                msg = error.get('msg', 'Validation error')
                error_details.append(f"{field}: {msg}")
            
            error_msg = f"Property validation failed for {prop.key_name}: {'; '.join(error_details)}"
            logger.error(error_msg)
            raise PropertyValidationError(error_msg, prop.key_name, value) 